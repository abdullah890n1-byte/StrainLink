"""Synthetic passive retaining-wall experiment with exact ground truth (SYNTHETIC).

Kinematic model (wall frame: origin at wall toe, x into backfill, z up, mm):

* The primary slip surface ("band-1") starts at the wall toe with lift-off
  angle alpha, follows a logarithmic spiral and then continues as a
  straight line at angle beta to the soil surface (composite), or is a
  straight line throughout. A log-spiral has a radius of curvature linear
  in arc length, rho(s) = rho0 + tan(phi) * s (its Cesaro equation), which
  is how it is generated here.
* Soil on the wall side of the band (the mobilised wedge) moves with the
  wall; soil beyond it is stationary. The transition across the band is a
  tanh profile, so the shear strain across the band has a sech^2 profile
  whose full width at half maximum is the band width.
* Progressive failure: only the part of the band behind its tip (which
  advances with wall displacement) is sharp. Ahead of the tip the
  transition is diffuse, which gives the near-uniform strain the reference
  paper reports before localisation.
* An optional radial band R1 runs from the wall top to band-1, and
  optional intermediate bands start on the wall (for the Log-Spiral-
  Hyperbolic question).

Presets are loosely calibrated to values reported by Khan et al. (2026),
Tables 1, 3 and 6 [reference-paper facts]. The generated data is
SYNTHETIC and is not a reproduction of the paper's experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import optimize
from matplotlib.path import Path
from scipy.spatial import cKDTree

ShapeLabel = Literal["linear", "log-spiral", "composite"]

# FWHM of sech^2(t) is 2*asinh(1) = 1.7627 in units of t; converts a band
# width (FWHM of shear strain) into the tanh length scale.
SECH2_FWHM = 2.0 * np.arcsinh(1.0)
CURVE_SAMPLING_MM = 0.5
# Angle difference below which the spiral part is dropped (straight band).
STRAIGHT_TOLERANCE_DEG = 1.0
# Width of the diffuse transition ahead of the band tip, as a multiple of band width.
DIFFUSE_WIDTH_FACTOR = 6.0
# Length scale of the sharp-to-diffuse blend around the band tip (mm).
TIP_BLEND_MM = 15.0
# Distance over which soil next to the wall is dragged horizontally (mm).
WALL_DRAG_MM = 60.0
# Length scale for averaging band tangents into a smooth slip-direction field
# (mm). Nearest-point tangents jump on the concave side of a curved band,
# which would create false strain lines. Weights decay with (distance -
# nearest distance) rather than squared distance, so the field stays smooth
# far from the band too.
DIRECTION_SMOOTHING_MM = 60.0
DIRECTION_SAMPLE_MM = 5.0
# Wall displacement over which a secondary band ramps up after it appears (mm).
SECONDARY_RAMP_MM = 3.0
# Points per chunk when evaluating the direction field (bounds memory).
FIELD_CHUNK = 200_000
# Stand-in mismatch for a curve that never reaches the surface (keeps brentq bracketed).
LENGTH_UNREACHABLE_MM = 1e6
# Span over which the measurable lift-off angle is defined (matches the detector default).
LIFT_OFF_SPAN_MM = 50.0
# A composite band is labelled "composite" only if its developed straight part is
# at least this many band widths long; shorter tails cannot be told from a
# pure log-spiral, so the honest ground-truth label is "log-spiral".
MIN_STRAIGHT_WIDTHS = 2.0


@dataclass(frozen=True)
class SecondaryBand:
    """A straight band between two points, active above a wall displacement."""

    name: str
    start: tuple[float, float]           # (x, z) mm
    end: tuple[float, float] | None      # None -> ends on band-1 (see end_on_primary)
    end_on_primary: float = 0.6          # fraction of band-1 arc length where it joins
    slip_fraction: float = 0.3           # slip across the band as a fraction of wall displacement
    active_from_mm: float = 0.0          # wall displacement at which it appears
    active_until_mm: float = np.inf      # slip stops growing after this


@dataclass(frozen=True)
class WallTestSpec:
    name: str
    wall_height_mm: float = 390.0
    friction_angle_deg: float = 41.6
    band_width_mm: float = 20.0
    target_length_mm: float = 650.0      # distance from wall to where band-1 meets the surface
    lift_off_start_deg: float = 10.0     # alpha when band-1 starts forming
    lift_off_end_deg: float = 25.0       # alpha at max_displacement_mm
    exit_angle_deg: float | None = None  # beta; None -> Rankine passive 45 - phi/2
    band_start_mm: float = 5.0           # wall displacement at which band-1 starts to localise
    band_full_mm: float = 15.0           # wall displacement at which band-1 reaches the surface
    max_displacement_mm: float = 60.0
    secondary: tuple[SecondaryBand, ...] = ()
    shape: ShapeLabel = "composite"

    @property
    def exit_angle(self) -> float:
        if self.exit_angle_deg is not None:
            return self.exit_angle_deg
        return 45.0 - self.friction_angle_deg / 2.0


@dataclass
class Curve:
    """Densely sampled polyline with arc length, tangents and normals."""

    points: np.ndarray    # (n, 2)
    s: np.ndarray         # (n,) arc length from the start, mm
    tangent: np.ndarray   # (n, 2) unit

    @classmethod
    def from_points(cls, points: np.ndarray) -> "Curve":
        seg = np.diff(points, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        tangent = np.gradient(points, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        return cls(points=points, s=s, tangent=tangent)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def upto(self, arc_length: float) -> np.ndarray:
        keep = self.s <= arc_length
        return self.points[keep]


@dataclass
class BandTruth:
    name: str
    label: Literal["primary", "radial", "intermediate"]
    centreline: np.ndarray       # developed (sharp) part only, (n, 2) mm, from its start
    reached_end: bool            # band-1: reached the surface; others: fully formed
    width_mm: float


@dataclass
class FrameTruth:
    wall_displacement_mm: float
    bands: list[BandTruth]
    lift_off_deg: float          # model tangent angle at the toe (generator parameter)
    lift_off_span_deg: float     # measurable definition: band direction over its first LIFT_OFF_SPAN_MM
    shape: ShapeLabel
    length_mm: float             # NaN until band-1 reaches the surface
    area_mm2: float              # NaN until band-1 reaches the surface
    primary_full: np.ndarray     # full band-1 geometry used for the kinematics


def _turning(alpha: float, beta: float, rho0: float, k: float, s: np.ndarray) -> np.ndarray:
    """Tangent angle (rad) along a log-spiral-then-straight curve."""
    direction = np.sign(beta - alpha)
    theta = alpha + direction * np.log1p(k * s / rho0) / k
    reached = direction * (theta - beta) >= 0
    theta[reached] = beta
    return theta


def primary_curve(
    alpha_deg: float, beta_deg: float, rho0: float, friction_deg: float, height: float,
) -> Curve:
    """Band-1 from the toe to the surface (z = height)."""
    alpha, beta = np.radians(alpha_deg), np.radians(beta_deg)
    if abs(alpha_deg - beta_deg) < STRAIGHT_TOLERANCE_DEG:
        beta = alpha
    k = np.tan(np.radians(friction_deg))
    max_length = 20.0 * height
    s = np.arange(0.0, max_length, CURVE_SAMPLING_MM)
    theta = _turning(alpha, beta, rho0, k, s)
    if np.all(np.sin(theta) <= 0):
        raise ValueError("band never rises to the surface")
    x = np.concatenate([[0.0], np.cumsum(np.cos(theta[:-1]) * CURVE_SAMPLING_MM)])
    z = np.concatenate([[0.0], np.cumsum(np.sin(theta[:-1]) * CURVE_SAMPLING_MM)])
    exit_index = int(np.argmax(z >= height))
    if z[exit_index] < height:
        raise ValueError("band does not reach the surface within the search length")
    points = np.column_stack([x[: exit_index + 1], z[: exit_index + 1]])
    points[-1] = _interpolate_to_height(points[-2], points[-1], height)
    return Curve.from_points(points)


def _interpolate_to_height(a: np.ndarray, b: np.ndarray, height: float) -> np.ndarray:
    t = (height - a[1]) / (b[1] - a[1])
    return a + t * (b - a)


def _extended_forward(curve: Curve, extension_mm: float) -> Curve:
    """Curve continued straight past its end (band-1 beyond the soil surface)."""
    after = curve.points[-1] + np.outer(np.arange(CURVE_SAMPLING_MM, extension_mm, CURVE_SAMPLING_MM),
                                        curve.tangent[-1])
    return Curve.from_points(np.vstack([curve.points, after]))


class WallExperiment:
    """Synthetic experiment: displacement field and ground truth at any wall displacement."""

    def __init__(self, spec: WallTestSpec):
        self.spec = spec
        self._rho_cache: dict[float, float] = {}

    # --- band-1 geometry -------------------------------------------------
    def lift_off(self, delta: float) -> float:
        s = self.spec
        t = np.clip((delta - s.band_start_mm) / (s.max_displacement_mm - s.band_start_mm), 0.0, 1.0)
        return s.lift_off_start_deg + t * (s.lift_off_end_deg - s.lift_off_start_deg)

    def _beta(self, alpha: float) -> float:
        return alpha if self.spec.shape == "linear" else self.spec.exit_angle

    def _rho0_for(self, alpha: float) -> float:
        """Initial radius of curvature that makes band-1 meet the surface at target_length_mm."""
        key = round(alpha, 6)
        if key in self._rho_cache:
            return self._rho_cache[key]
        s = self.spec
        beta = self._beta(alpha)
        if abs(alpha - beta) < STRAIGHT_TOLERANCE_DEG:
            rho = np.inf
        else:
            def mismatch(log_rho: float) -> float:
                try:
                    c = primary_curve(alpha, beta, np.exp(log_rho), s.friction_angle_deg, s.wall_height_mm)
                except ValueError:
                    # Curve never reaches the surface: effectively infinitely long.
                    return LENGTH_UNREACHABLE_MM
                return c.points[-1, 0] - s.target_length_mm
            rho = float(np.exp(optimize.brentq(mismatch, np.log(1.0), np.log(1e5), xtol=1e-6)))
        self._rho_cache[key] = rho
        return rho

    def primary(self, delta: float) -> Curve:
        s = self.spec
        alpha = self.lift_off(delta)
        rho0 = self._rho0_for(alpha)
        return primary_curve(alpha, self._beta(alpha), rho0 if np.isfinite(rho0) else 1.0,
                             s.friction_angle_deg, s.wall_height_mm)

    def spiral_length(self, alpha: float) -> float:
        """Arc length of the log-spiral part of band-1 (0 for a straight band)."""
        beta = self._beta(alpha)
        if abs(alpha - beta) < STRAIGHT_TOLERANCE_DEG:
            return 0.0
        k = np.tan(np.radians(self.spec.friction_angle_deg))
        rho0 = self._rho0_for(alpha)
        return float(rho0 * np.expm1(k * abs(np.radians(beta - alpha))) / k)

    def _developed_shape(self, alpha: float, developed_mm: float) -> ShapeLabel:
        spiral = self.spiral_length(alpha)
        if spiral == 0.0:
            return "linear"
        straight = developed_mm - spiral
        return "composite" if straight >= MIN_STRAIGHT_WIDTHS * self.spec.band_width_mm else "log-spiral"

    def tip_fraction(self, delta: float) -> float:
        s = self.spec
        return float(np.clip((delta - s.band_start_mm) / (s.band_full_mm - s.band_start_mm), 0.0, 1.0))

    # --- secondary bands -------------------------------------------------
    def _secondary_curve(self, band: SecondaryBand, primary: Curve) -> Curve:
        start = np.asarray(band.start, dtype=float)
        if band.end is not None:
            end = np.asarray(band.end, dtype=float)
        else:
            end = primary.points[np.searchsorted(primary.s, band.end_on_primary * primary.length)]
        n = max(int(np.hypot(*(end - start)) / CURVE_SAMPLING_MM), 2)
        return Curve.from_points(np.linspace(start, end, n))

    def _secondary_slip(self, band: SecondaryBand, delta: float) -> float:
        """Slip grows with the wall push once active, and stops growing after active_until_mm."""
        if delta <= band.active_from_mm:
            return 0.0
        ramp = min((delta - band.active_from_mm) / SECONDARY_RAMP_MM, 1.0)
        return band.slip_fraction * min(delta, band.active_until_mm) * ramp

    # --- kinematics ------------------------------------------------------
    def displacement(self, x: np.ndarray, z: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
        """Displacement (ux, uz) in mm at reference positions (x, z) for wall push delta."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        if delta <= 0:
            return np.zeros_like(x), np.zeros_like(z)
        spec = self.spec
        primary = self.primary(delta)
        signed = _signed_distance_to_primary(primary, x, z, spec.wall_height_mm)

        # Sharp behind the tip, diffuse ahead of it. The blend compares distances
        # to the developed and undeveloped parts: both are continuous in space,
        # so the field has no artificial jumps.
        tip_s = self.tip_fraction(delta) * primary.length
        developed = primary.points[primary.s <= tip_s]
        undeveloped = primary.points[primary.s > tip_s]
        if len(undeveloped) == 0:
            sharpness = np.ones_like(x)  # fully formed: sharp everywhere
        elif len(developed) == 0:
            sharpness = np.zeros_like(x)
        else:
            d_dev = _distance_to(developed, x, z)
            d_undev = _distance_to(undeveloped, x, z)
            sharpness = 0.5 * (1.0 + np.tanh((d_undev - d_dev) / TIP_BLEND_MM))
        sharp = spec.band_width_mm
        width = sharp + (DIFFUSE_WIDTH_FACTOR * sharp - sharp) * (1.0 - sharpness)
        mobilised = 0.5 * (1.0 + np.tanh(signed * SECH2_FWHM / width))

        # Direction: along the slip surface (smoothed), turning horizontal next to the wall.
        tangent = _smooth_direction(_extended_forward(primary, extension_mm=spec.wall_height_mm), x, z)
        drag = np.exp(-np.clip(x, 0.0, None) / WALL_DRAG_MM)
        direction = (1.0 - drag)[..., None] * tangent + drag[..., None] * np.array([1.0, 0.0])
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        ux = delta * mobilised * direction[..., 0]
        uz = delta * mobilised * direction[..., 1]

        for band in spec.secondary:
            slip = self._secondary_slip(band, delta)
            if slip <= 0:
                continue
            curve = self._secondary_curve(band, primary)
            start = curve.points[0]
            t = (curve.points[-1] - start) / np.linalg.norm(curve.points[-1] - start)
            # Straight band: exact signed distance to its line (positive on the left).
            d = t[0] * (z - start[1]) - t[1] * (x - start[0])
            # Secondary bands run between wedge boundaries (wall, surface, band-1),
            # so the block on one side slips as a whole; `mobilised` confines the
            # slip to the wedge.
            step = 0.5 * (1.0 + np.tanh(d * SECH2_FWHM / sharp)) * mobilised
            ux = ux + slip * step * t[..., 0]
            uz = uz + slip * step * t[..., 1]
        return ux, uz

    # --- ground truth ----------------------------------------------------
    def truth(self, delta: float) -> FrameTruth:
        spec = self.spec
        primary = self.primary(delta)
        fraction = self.tip_fraction(delta)
        bands: list[BandTruth] = []
        if fraction > 0:
            bands.append(BandTruth(
                name="band-1", label="primary",
                centreline=primary.upto(fraction * primary.length),
                reached_end=fraction >= 1.0, width_mm=spec.band_width_mm,
            ))
        for band in spec.secondary:
            if self._secondary_slip(band, delta) > 0:
                curve = self._secondary_curve(band, primary)
                label = "radial" if band.start[1] >= spec.wall_height_mm - 1e-6 else "intermediate"
                bands.append(BandTruth(name=band.name, label=label, centreline=curve.points,
                                       reached_end=True, width_mm=spec.band_width_mm))
        alpha = self.lift_off(delta)
        shape = self._developed_shape(alpha, fraction * primary.length)
        head = primary.upto(max(LIFT_OFF_SPAN_MM, 0.0))
        chord = _line_angle_deg(head)
        complete = fraction >= 1.0
        length = float(primary.points[-1, 0]) if complete else np.nan
        area = _wedge_area(primary.points, spec.wall_height_mm) if complete else np.nan
        return FrameTruth(
            wall_displacement_mm=delta, bands=bands, lift_off_deg=alpha, lift_off_span_deg=chord, shape=shape,
            length_mm=length, area_mm2=area, primary_full=primary.points,
        )


def _signed_distance_to_primary(primary: Curve, x: np.ndarray, z: np.ndarray, height: float) -> np.ndarray:
    """Distance to band-1, positive inside the mobilised wedge.

    The side is decided by a point-in-polygon test on the wedge (bounded by
    band-1 and closed outside the soil domain), not by the nearest point's
    tangent: nearest points can switch between distant parts of a curved
    band, which would flip the sign and create false strain.
    """
    far = 10.0 * height
    exit_point = primary.points[-1]
    polygon = np.vstack([
        primary.points,
        [[exit_point[0], height + far], [-far, height + far], [-far, 0.0]],
    ])
    inside = Path(polygon).contains_points(np.column_stack([x.ravel(), z.ravel()])).reshape(x.shape)
    distance = _distance_to(primary.points, x, z)
    return np.where(inside, distance, -distance)


def _distance_to(points: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    distance, _ = cKDTree(points).query(np.column_stack([x.ravel(), z.ravel()]))
    return distance.reshape(x.shape)


def _smooth_direction(curve: Curve, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Gaussian-weighted average of band tangents: a continuous direction field."""
    step = max(int(DIRECTION_SAMPLE_MM / CURVE_SAMPLING_MM), 1)
    pts, tan = curve.points[::step], curve.tangent[::step]
    flat_x, flat_z = x.ravel(), z.ravel()
    out = np.empty((flat_x.size, 2))
    for start in range(0, flat_x.size, FIELD_CHUNK):
        sl = slice(start, start + FIELD_CHUNK)
        dist = np.hypot(flat_x[sl, None] - pts[None, :, 0], flat_z[sl, None] - pts[None, :, 1])
        w = np.exp(-(dist - dist.min(axis=1, keepdims=True)) / DIRECTION_SMOOTHING_MM)
        v = w @ tan
        out[sl] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return out.reshape(x.shape + (2,))


def _line_angle_deg(points: np.ndarray) -> float:
    """Direction of the orthogonal-regression line through points, oriented along travel."""
    centre = points.mean(axis=0)
    direction = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    if np.dot(direction, points[-1] - points[0]) < 0:
        direction = -direction
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _wedge_area(centreline: np.ndarray, height: float) -> float:
    """Area enclosed by the wall (x = 0), the surface (z = height) and band-1."""
    polygon = np.vstack([centreline, [[0.0, height]]])
    x, z = polygon[:, 0], polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


def _radial_r1(height: float, active_from: float) -> SecondaryBand:
    # The paper reports R1 forms with band-1 and then stops developing (Sec. 5.2).
    return SecondaryBand(name="R1", start=(0.0, height), end=None, end_on_primary=0.55,
                         slip_fraction=0.3, active_from_mm=active_from, active_until_mm=active_from + 10.0)


def preset(name: Literal["loose", "medium", "dense"]) -> WallTestSpec:
    """Presets loosely based on the reference paper's sand tests (RW-LS, RW-MDS, RW-DS).

    Values from Khan et al. (2026) [reference-paper facts]: friction angles
    (Table 1), displacement at peak (Table 3), lift-off angles and impacted
    lengths (Table 6), and band onset/formation displacements (Sec. 5.2).
    The band width (20 mm) is an assumption, about 14 x D50 for this sand.
    Exit angles are assumptions chosen so that a concave-up band reaches
    the surface at the paper's impacted length: with the Rankine angle
    45 - phi/2 (about 24 deg) a concave-up band could not end closer than
    H / tan(24 deg) = 866 mm, yet the paper reports 590-680 mm.
    """
    height = 390.0
    if name == "loose":
        return WallTestSpec(
            name="SYNTH_loose_sand", friction_angle_deg=39.7, target_length_mm=634.0,
            lift_off_start_deg=-24.4, lift_off_end_deg=17.4, exit_angle_deg=50.0,
            band_start_mm=20.0, band_full_mm=60.0, shape="composite", secondary=(_radial_r1(height, 30.0),),
        )
    if name == "medium":
        return WallTestSpec(
            name="SYNTH_medium_sand", friction_angle_deg=41.6, target_length_mm=680.0,
            lift_off_start_deg=0.0, lift_off_end_deg=8.0, exit_angle_deg=42.0,
            band_start_mm=7.0, band_full_mm=15.0, shape="composite", secondary=(_radial_r1(height, 15.0),),
        )
    if name == "dense":
        return WallTestSpec(
            name="SYNTH_dense_sand", friction_angle_deg=43.8, target_length_mm=590.0,
            lift_off_start_deg=33.5, lift_off_end_deg=33.5, band_start_mm=2.0, band_full_mm=5.0,
            shape="linear", secondary=(_radial_r1(height, 5.0),),
        )
    raise ValueError(f"unknown preset {name!r}")


def with_intermediate_bands(spec: WallTestSpec, count: int = 2) -> WallTestSpec:
    """Add intermediate bands starting on the wall face between top and toe
    (the rotation-mode behaviour the LSH model assumes)."""
    h = spec.wall_height_mm
    extra = tuple(
        SecondaryBand(
            name=f"I{i + 1}", start=(0.0, h * (1 - (i + 1) / (count + 1))),
            end=(0.45 * spec.target_length_mm * (i + 1) / count, h),
            slip_fraction=0.3, active_from_mm=spec.band_start_mm * 0.5,
            active_until_mm=spec.band_full_mm,
        )
        for i in range(count)
    )
    return replace(spec, name=spec.name + "_with_intermediate", secondary=spec.secondary + extra)
