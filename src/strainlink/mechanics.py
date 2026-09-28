"""Engineering results from the sensor records (automates the paper's Tables 3 and 5).

* Passive force = actuator force minus base friction from the calibration
  run (no soil), interpolated at the same push (paper Sec. 3.3).
* Peak passive force Pp, the push at the peak delta_p, and the passive earth
  pressure coefficient Kp = 2 Pp / (gamma H^2 w) (paper Eq. 1).
* Pressure cells: peak, push at peak, and whether the cell softens after
  the peak (with the onset) or holds a plateau (paper Sec. 4.4).
* Thrust from the pressure cells (trapezoidal integration over depth times
  wall width) against the actuator's passive force (paper Table 5).
* Wall uplift onset and the mobilised wall friction angle
  delta_mob = atan(W / P_h) at uplift (paper Sec. 4.3, after Duncan & Mokwa 2001).

Every value carries an uncertainty estimated from the record's own noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# Smoothing window for locating peaks, in push (mm): noise must not create peaks.
PEAK_SMOOTHING_MM = 0.5
# A record "softens" if it falls by more than this fraction of its peak AND by
# more than SOFTENING_NOISE_K times its noise after the peak.
SOFTENING_FRACTION = 0.05
SOFTENING_NOISE_K = 5.0
# Uplift has started once the uplift record exceeds this many noise standard
# deviations for UPLIFT_CONFIRM_MM of further push.
UPLIFT_NOISE_K = 5.0
UPLIFT_CONFIRM_MM = 0.5
# Push range over which the early uplift trend is fitted for back-extrapolation (mm).
UPLIFT_TREND_MM = 2.0
# A value within 2 sigma of its peak over more than this fraction of the push is a plateau.
PLATEAU_SPAN_FRACTION = 0.25


@dataclass
class PeakResult:
    peak: float
    peak_sigma: float
    push_at_peak_mm: float
    push_range_mm: tuple[float, float]   # pushes where the value is within 2 sigma of the peak
    softens: bool
    softening_onset_mm: float            # NaN if it does not soften
    residual: float                      # mean over the last 10% of the push
    plateau: bool = False                # flat top: push at peak = where the plateau starts


@dataclass
class CellResult(PeakResult):
    name: str = ""
    depth_mm: float = float("nan")


@dataclass
class MechanicsResults:
    push_mm: np.ndarray
    passive_kn: np.ndarray
    passive: PeakResult
    kp: float
    kp_sigma: float
    cells: list[CellResult] = field(default_factory=list)
    thrust_at_rest_kn: float = float("nan")
    thrust_from_cells_at_peak_kn: float = float("nan")
    thrust_mismatch_fraction: float = float("nan")
    uplift_onset_mm: float = float("nan")
    delta_mob_deg: float = float("nan")
    notes: list[str] = field(default_factory=list)


def noise_sigma(values: np.ndarray) -> float:
    """Measurement noise from successive differences (robust to trends): MAD(diff)/sqrt(2)."""
    d = np.diff(values[np.isfinite(values)])
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(d) else 0.0


def passive_force(push_mm: np.ndarray, actuator_kn: np.ndarray,
                  calibration_push_mm: np.ndarray | None = None,
                  calibration_kn: np.ndarray | None = None) -> np.ndarray:
    if calibration_push_mm is None or calibration_kn is None:
        return actuator_kn.copy()
    friction = np.interp(push_mm, calibration_push_mm, calibration_kn)
    return actuator_kn - friction


def _smooth(push: np.ndarray, values: np.ndarray) -> np.ndarray:
    step = np.nanmedian(np.diff(push[push > 0])) if (push > 0).sum() > 2 else 1.0
    size = max(int(round(PEAK_SMOOTHING_MM / max(step, 1e-9))), 1)
    return ndimage.uniform_filter1d(values, size=size, mode="nearest")


def analyse_peak(push: np.ndarray, values: np.ndarray) -> PeakResult:
    """Peak, push at peak (with range), softening and residual of one record."""
    moving = push > 0
    push, values = push[moving], values[moving]
    sigma = noise_sigma(values)
    smooth = _smooth(push, values)
    k = int(np.nanargmax(smooth))
    peak = float(smooth[k])
    near = np.flatnonzero(smooth >= peak - 2 * sigma)
    # On a flat top the maximum is wherever the noise happens to be highest.
    # If the value stays within noise of the peak over a long push range, the
    # meaningful "push at peak" is where that plateau starts.
    plateau = (push[near.max()] - push[near.min()]) > PLATEAU_SPAN_FRACTION * (push[-1] - push[0])
    at_peak = int(near.min()) if plateau else k
    tail = values[push >= push[-1] - 0.1 * (push[-1] - push[0])]
    residual = float(np.nanmean(tail))
    drop = peak - np.nanmin(smooth[k:]) if k < len(smooth) - 1 else 0.0
    softens = bool(drop > SOFTENING_FRACTION * abs(peak) and drop > SOFTENING_NOISE_K * sigma)
    onset = float("nan")
    if softens:
        below = np.flatnonzero(smooth[k:] < peak - max(SOFTENING_FRACTION * abs(peak) / 2, 2 * sigma))
        onset = float(push[k + below[0]]) if len(below) else float("nan")
    # Peak uncertainty: noise averaged over the smoothing window.
    window = max(int(round(PEAK_SMOOTHING_MM / max(np.nanmedian(np.diff(push)), 1e-9))), 1)
    return PeakResult(peak=peak, peak_sigma=float(sigma / np.sqrt(window)), push_at_peak_mm=float(push[at_peak]),
                      push_range_mm=(float(push[near.min()]), float(push[near.max()])),
                      softens=softens, softening_onset_mm=onset, residual=residual, plateau=bool(plateau and not softens))


def analyse(push_mm: np.ndarray, actuator_kn: np.ndarray, *,
            unit_weight_kn_m3: float, wall_height_mm: float, wall_width_mm: float,
            calibration: tuple[np.ndarray, np.ndarray] | None = None,
            cells: dict[str, tuple[float, np.ndarray]] | None = None,
            uplift_mm: np.ndarray | None = None, wall_weight_kn: float | None = None) -> MechanicsResults:
    """All sensor-side engineering results.

    cells: name -> (depth below the soil surface in mm, pressure record in kPa).
    """
    notes: list[str] = []
    cal_push, cal_force = calibration if calibration is not None else (None, None)
    if calibration is None:
        notes.append("No calibration run: base friction not removed; forces include friction.")
    passive = passive_force(push_mm, actuator_kn, cal_push, cal_force)
    peak = analyse_peak(push_mm, passive)
    h_m, w_m = wall_height_mm / 1000.0, wall_width_mm / 1000.0
    denominator = unit_weight_kn_m3 * h_m**2 * w_m
    kp = 2.0 * peak.peak / denominator
    kp_sigma = 2.0 * peak.peak_sigma / denominator

    results = MechanicsResults(push_mm=push_mm, passive_kn=passive, passive=peak, kp=kp, kp_sigma=kp_sigma, notes=notes)
    if cells:
        for name, (depth, pressure) in sorted(cells.items(), key=lambda item: item[1][0]):
            r = analyse_peak(push_mm, pressure)
            results.cells.append(CellResult(**r.__dict__, name=name, depth_mm=depth))
        results.thrust_at_rest_kn = _thrust(cells, push_mm <= 0, wall_width_mm, wall_height_mm)
        at_peak = np.abs(push_mm - peak.push_at_peak_mm) == np.min(np.abs(push_mm - peak.push_at_peak_mm))
        results.thrust_from_cells_at_peak_kn = _thrust(cells, at_peak, wall_width_mm, wall_height_mm)
        if peak.peak > 0:
            results.thrust_mismatch_fraction = (peak.peak - results.thrust_from_cells_at_peak_kn) / peak.peak
    if uplift_mm is not None:
        results.uplift_onset_mm = _uplift_onset(push_mm, uplift_mm)
        if wall_weight_kn and np.isfinite(results.uplift_onset_mm):
            p_h = float(np.interp(results.uplift_onset_mm, push_mm, _smooth(push_mm, passive)))
            if p_h > 0:
                results.delta_mob_deg = float(np.degrees(np.arctan(wall_weight_kn / p_h)))
        elif not wall_weight_kn:
            notes.append("Wall weight not given: mobilised wall friction angle not computed.")
    return results


def _thrust(cells: dict[str, tuple[float, np.ndarray]], rows: np.ndarray, wall_width_mm: float,
            wall_height_mm: float) -> float:
    """Resultant from pressure cells over the full wall height (kN).

    Trapezoidal integration from the soil surface (zero pressure) through the
    cells to the wall base, with the deepest pair extrapolated linearly to
    the base (as over the effective height in the paper, Sec. 4.5).
    """
    depths = np.array([d for d, _ in cells.values()])
    pressures = np.array([np.nanmean(p[rows]) for _, p in cells.values()])
    order = np.argsort(depths)
    depths, pressures = depths[order], pressures[order]
    if len(depths) >= 2 and depths[-1] < wall_height_mm:
        slope = (pressures[-1] - pressures[-2]) / (depths[-1] - depths[-2])
        depths = np.append(depths, wall_height_mm)
        pressures = np.append(pressures, pressures[-1] + slope * (wall_height_mm - depths[-2]))
    if depths[0] > 0:
        depths, pressures = np.insert(depths, 0, 0.0), np.insert(pressures, 0, 0.0)
    return float(np.trapezoid(pressures, depths / 1000.0) * wall_width_mm / 1000.0)


def _uplift_onset(push: np.ndarray, uplift: np.ndarray) -> float:
    before = uplift[push <= 0]
    baseline = float(np.nanmedian(before)) if len(before) else float(np.nanmedian(uplift[:10]))
    sigma = noise_sigma(uplift)
    above = (uplift - baseline) > UPLIFT_NOISE_K * max(sigma, 1e-9)
    for k in np.flatnonzero(above):
        confirm = (push >= push[k]) & (push <= push[k] + UPLIFT_CONFIRM_MM)
        if above[confirm].all():
            # The noise threshold is crossed after the real start: back-extrapolate
            # the early uplift trend to where it leaves the baseline.
            trend = (push >= push[k]) & (push <= push[k] + UPLIFT_TREND_MM)
            slope, intercept = np.polyfit(push[trend], uplift[trend] - baseline, 1)
            if slope > 0:
                return float(min(push[k], max(-intercept / slope, 0.0)))
            return float(push[k])
    return float("nan")
