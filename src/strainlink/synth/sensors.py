"""Synthetic sensor logs for the synthetic retaining-wall experiment (SYNTHETIC).

Mirrors the reference paper's instrumentation (Khan et al. 2026, Sec. 2):
actuator force and displacement, two horizontal LVDTs, a vertical LVDT for
wall uplift, and pressure transducers PT1..PT5 on the wall face, logged
against the data logger's clock. A calibration run (no soil) gives base
friction. The camera has its own clock, offset from the logger's.

Planted relationships (ground truth for the correlation engine):
  * PTi softens after band-1's tip passes the PTi level (deeper cells are
    passed first); cells never passed keep a plateau.
  * Passive force rises smoothly to Pp at delta_p, then softens
    (dense) or plateaus (loose/medium), with Pp = 0.5*Kp*gamma*H^2*w.
  * Wall uplift starts when the vertical component of the passive force
    (force * tan(delta_mob)) exceeds the wall weight.

Numbers are loosely based on the paper's Tables 1 and 3 [reference-paper
facts]. All output is SYNTHETIC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .wall import WallExperiment

CONCRETE_UNIT_WEIGHT_KN_M3 = 24.0
# Fraction of peak pressure kept after full softening (dense residual behaviour).
SOFTENED_FRACTION = 0.82
# Wall displacement over which a pressure cell softens once the band tip passes it (mm).
SOFTENING_SPAN_MM = 4.0
# Shape of the pre-peak rise P = Pp * (1 - (1 - d/dp)^n): stiff at first, zero
# slope exactly at the peak. n = 5 gives an initial stiffness of 5*Pp/dp.
RISE_EXPONENT = 5.0


@dataclass(frozen=True)
class SensorSpec:
    """Test protocol and instrumentation for a synthetic test."""

    unit_weight_kn_m3: float        # backfill unit weight gamma
    kp: float                       # passive earth pressure coefficient at peak
    delta_p_mm: float               # wall displacement at peak passive force
    softening: bool                 # dense behaviour: force softens after the peak
    wall_width_m: float = 0.70
    wall_thickness_m: float = 0.15
    wall_total_height_m: float = 0.45
    delta_mob_deg: float = 11.0     # mobilised wall friction angle
    push_rate_mm_s: float = 0.01
    max_push_mm: float = 60.0
    pre_push_s: float = 30.0        # logging before the actuator starts
    logger_rate_hz: float = 1.0
    camera_interval_s: float = 100.0      # one photo per 1 mm of push at 0.01 mm/s
    camera_clock_offset_s: float = 7.3    # camera clock minus logger clock
    pt_depths_fraction: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)   # of H, from the surface
    friction_kn: tuple[float, float] = (0.15, 0.004)   # base friction = a + b * push (kN, kN/mm)
    noise_force_kn: float = 0.02
    noise_disp_mm: float = 0.005
    noise_pressure_kpa: float = 0.3
    seed: int = 7

    @property
    def peak_force_kn(self) -> float:
        return 0.5 * self.kp * self.unit_weight_kn_m3 * self.height_m**2 * self.wall_width_m

    @property
    def height_m(self) -> float:
        return 0.39

    @property
    def wall_weight_kn(self) -> float:
        return CONCRETE_UNIT_WEIGHT_KN_M3 * self.wall_width_m * self.wall_thickness_m * self.wall_total_height_m


def sensor_preset(name: str) -> SensorSpec:
    """Values from the paper's sand tests (Tables 1 and 3): gamma, Kp, delta_p."""
    if name == "loose":
        return SensorSpec(unit_weight_kn_m3=13.55, kp=10.94, delta_p_mm=32.67, softening=False)
    if name == "medium":
        return SensorSpec(unit_weight_kn_m3=14.13, kp=11.76, delta_p_mm=15.71, softening=False)
    if name == "dense":
        return SensorSpec(unit_weight_kn_m3=14.90, kp=12.43, delta_p_mm=12.52, softening=True)
    raise ValueError(f"unknown preset {name!r}")


@dataclass
class SyntheticLogs:
    logger: pd.DataFrame            # time_s + channels, logger clock
    calibration: pd.DataFrame       # push_mm, force_kn (no soil)
    camera_times_s: np.ndarray      # camera clock
    frame_push_mm: np.ndarray       # true wall push at each photo
    pt_depths_mm: dict[str, float]  # depth below the soil surface
    truth: dict = field(default_factory=dict)


def _force_curve(push: np.ndarray, spec: SensorSpec) -> np.ndarray:
    """Passive force (kN): smooth rise to Pp at delta_p, then plateau or softening."""
    pp, dp = spec.peak_force_kn, spec.delta_p_mm
    x = np.clip(push / dp, 0.0, None)
    rise = pp * (1.0 - (1.0 - np.clip(x, 0, 1)) ** RISE_EXPONENT)
    force = np.where(x <= 1.0, rise, pp)
    if spec.softening:
        after = np.clip((push - dp) / (2.0 * dp), 0.0, 1.0)
        force = force * (1.0 - (1.0 - SOFTENED_FRACTION) * after)
    return force


def _tip_height(experiment: WallExperiment, push: np.ndarray) -> np.ndarray:
    """Height z (mm) of band-1's tip above the toe; -inf before localisation."""
    heights = np.full(push.shape, -np.inf)
    for i, d in enumerate(push):
        truth = experiment.truth(float(d))
        primary = next((b for b in truth.bands if b.label == "primary"), None)
        if primary is not None and len(primary.centreline):
            heights[i] = float(np.max(primary.centreline[:, 1]))
    return heights


def generate(experiment: WallExperiment, spec: SensorSpec) -> SyntheticLogs:
    rng = np.random.default_rng(spec.seed)
    duration = spec.pre_push_s + spec.max_push_mm / spec.push_rate_mm_s
    t = np.arange(0.0, duration + 1e-9, 1.0 / spec.logger_rate_hz)
    push = np.clip((t - spec.pre_push_s) * spec.push_rate_mm_s, 0.0, spec.max_push_mm)

    passive = _force_curve(push, spec)
    friction = np.where(push > 0, spec.friction_kn[0] + spec.friction_kn[1] * push, 0.0)
    actuator = passive + friction

    # Pressure cells: at rest K0*sigma_v, rising to the local passive pressure;
    # a cell softens once band-1's tip has passed its level.
    height_mm = spec.height_m * 1000.0
    k0 = 1.0 - np.sin(np.radians(experiment.spec.friction_angle_deg))
    # Tip height is sampled on a coarse push grid and interpolated (truth() is not vectorised).
    coarse = np.linspace(0.0, spec.max_push_mm, 121)
    coarse_tip = _tip_height(experiment, coarse)
    tip_z = np.interp(push, coarse, np.where(np.isfinite(coarse_tip), coarse_tip, -1e9))
    ratio = passive / spec.peak_force_kn if spec.peak_force_kn > 0 else passive
    pt_depths = {f"PT{i + 1}": f * height_mm for i, f in enumerate(spec.pt_depths_fraction)}
    pressures = {}
    pass_push = {}
    for name, depth in pt_depths.items():
        level = height_mm - depth
        sigma_v = spec.unit_weight_kn_m3 * depth / 1000.0            # kPa
        p_rest = k0 * sigma_v
        # Shallow cells mobilise more slowly (low confinement, paper Sec. 4.4).
        stiffness = 0.6 + 0.8 * depth / height_mm
        p = p_rest + (spec.kp * sigma_v - p_rest) * np.clip(ratio * stiffness, 0.0, 1.0)
        passed = tip_z >= level
        if passed.any():
            first = push[np.argmax(passed)]
            pass_push[name] = float(first)
            soften = np.clip((push - first) / SOFTENING_SPAN_MM, 0.0, 1.0)
            p = p * (1.0 - (1.0 - SOFTENED_FRACTION) * soften)
        else:
            pass_push[name] = float("nan")
        pressures[name] = p

    vertical = passive * np.tan(np.radians(spec.delta_mob_deg))
    uplift_start = push[np.argmax(vertical > spec.wall_weight_kn)] if (vertical > spec.wall_weight_kn).any() else np.nan
    uplift = np.where(vertical > spec.wall_weight_kn, 0.02 * (push - uplift_start), 0.0)

    def noisy(values: np.ndarray, sigma: float) -> np.ndarray:
        return values + rng.normal(0.0, sigma, values.shape)

    logger = pd.DataFrame({
        "time_s": t,
        "actuator_force_kN": noisy(actuator, spec.noise_force_kn),
        "actuator_disp_mm": noisy(push, spec.noise_disp_mm),
        "LVDT1_mm": noisy(push * 0.998, spec.noise_disp_mm),   # small frame compliance
        "LVDT2_mm": noisy(push * 0.997, spec.noise_disp_mm),
        "LVDT_uplift_mm": noisy(uplift, spec.noise_disp_mm),
        **{f"{k}_kPa": noisy(v, spec.noise_pressure_kpa) for k, v in pressures.items()},
    })
    cal_push = np.arange(0.0, spec.max_push_mm + 1e-9, 0.5)
    calibration = pd.DataFrame({
        "push_mm": cal_push,
        "force_kN": noisy(np.where(cal_push > 0, spec.friction_kn[0] + spec.friction_kn[1] * cal_push, 0.0),
                          spec.noise_force_kn / 2),
    })
    frame_push = np.arange(0.0, spec.max_push_mm + 1e-9, spec.camera_interval_s * spec.push_rate_mm_s)
    logger_times = spec.pre_push_s + frame_push / spec.push_rate_mm_s
    camera_times = logger_times + spec.camera_clock_offset_s
    peak_index = int(np.argmax(passive))
    truth = {
        "synthetic": True,
        "peak_force_kN": float(spec.peak_force_kn),
        "delta_p_mm": float(push[peak_index]) if not spec.softening else float(spec.delta_p_mm),
        "kp": spec.kp,
        "camera_clock_offset_s": spec.camera_clock_offset_s,
        "pt_band_tip_pass_mm": pass_push,
        "uplift_start_mm": float(uplift_start),
        "base_friction_kN": list(spec.friction_kn),
    }
    return SyntheticLogs(logger=logger, calibration=calibration, camera_times_s=camera_times,
                         frame_push_mm=frame_push, pt_depths_mm=pt_depths, truth=truth)
