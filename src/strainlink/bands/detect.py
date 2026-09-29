"""Automatic shear-band detection and measurement for one frame.

Replaces the manual steps of reading bands off strain contours (see
ProblemStatement A5-A8): finds every localised band, labels it by where it
attaches (wall toe -> primary "band-1", wall top -> radial "R", wall face
-> intermediate "I"), measures its centreline, width and tip, classifies
the primary band's shape and measures lift-off angle, impacted length,
wedge area and surface heave. Every result carries flags so uncertain
measurements are sent for engineer review instead of being reported as
fact.

Coordinates: wall frame (mm), origin at the wall toe, x into the backfill,
z upward, soil surface at z = wall height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold

from . import centreline as cl
from .paths import extract_paths
from .ridges import FWHM_PER_SIGMA, fill_missing, ridge_maxima, ridge_strength, robust_noise
from .shape import ShapeResult, classify, lift_off_from_model

# Relative tolerance when checking that the grid spacing is the same in x and z.
ISOTROPY_TOLERANCE = 0.01
# Noise estimate of gamma uses deviations from a local median over this many grid points.
NOISE_MEDIAN_SIZE = 5
# Points at the ends of a band whose peak falls below this fraction of the band's
# median peak are not part of the localised band.
TIP_PEAK_FRACTION = 0.35
# Holes of at most this many grid points in the strain field are bridged.
SMALL_HOLE_POINTS = 6
# Percentile used to characterise a band's core (sharpest quarter of its points).
CORE_PERCENTILE = 25
# Interruptions of a band's localised run up to this many core widths are
# bridged (another band crossing it).
CROSSING_GAP_WIDTHS = 2.5
# Share of NaN strain around a band above which it is flagged as near a data gap.
DATA_GAP_FRACTION = 0.1
# A band whose continuation past an end meets missing data inside the soil
# within this many band widths is flagged: it may continue through the gap.
END_GAP_WIDTHS = 1.5
# A band whose centreline runs this much through refilled (lost-correlation)
# points is flagged: its position and slip are sound, its width there is an upper bound.
RECONSTRUCTED_FRACTION = 0.1
# The top ~2 band widths below the surface cannot be fully resolved (strain
# window and ridge filter reach past the data edge), so a tip there counts as
# having reached the surface. It is *confirmed* only once the band is traced
# to within this many band widths (at least SURFACE_CONFIRM_MIN_GRID grid
# steps) of the top of the data; before that the reach is flagged.
SURFACE_CONFIRM_WIDTHS = 0.5
SURFACE_CONFIRM_MIN_GRID = 2.0
# Lift-off uncertainty (1 sigma, degrees) above which the angle is flagged: at
# 1.5 deg the 95% range reaches the +-3 deg accuracy tolerance.
LIFT_OFF_SIGMA_LIMIT_DEG = 1.5
# Shape is judged away from the band's ends: within about one band width of
# the wall the wall-friction zone bends the band, and at the surface the free
# boundary does. (Lift-off is still extrapolated to the toe from the model.)
SHAPE_END_EXCLUSION_WIDTHS = 1.0
# Signal-to-noise ratio below which a band is flagged as low contrast.
MIN_SNR = 5.0
# A band whose median width is this close to the maximum accepted width is flagged.
WIDTH_LIMIT_FRACTION = 0.8
# Points used at the band end to extrapolate to the soil surface, in band widths.
END_FIT_WIDTHS = 2.0
# Attachment tolerance in band widths (lower bound on attach_tolerance_mm).
ATTACH_WIDTHS = 2.0
# A band end is considered to reach towards the wall if it is within this many
# band widths of the wall face (it is then extended back to the wall).
WALL_REACH_WIDTHS = 3.0
# Bands within this angle of vertical and next to the wall are wall-interface shear zones.
WALL_INTERFACE_ANGLE_DEG = 30.0
# End hooks: where a band meets the wall, the surface or another strain zone,
# its measured centreline curls over its last centimetres. A band cannot turn
# faster than its resolution allows (radius of curvature >= 2 widths turns
# ~30 deg per width), so an end whose direction over one band width differs
# from the next two widths by more than this is trimmed.
HOOK_ANGLE_DEG = 25.0
# Hooks range from a fraction of a band width to about one width, so ends are
# tested at two scales.
HOOK_WINDOW_WIDTHS = (0.5, 1.0)
# Skeletons of bands that cross the edge of the data run along that edge for a
# short way. A shear band cannot run parallel to the free soil surface, and the
# other non-wall edges are just the image limits, so a path lying mostly within
# this many grid steps of those edges is an artefact. (The wall face is kept:
# wall-friction zones are real.)
EDGE_BAND_GRID = 2.0
EDGE_ARTEFACT_FRACTION = 0.6
# A "band" whose width is mostly measured from one side (the other cut off by
# a data edge) is unverifiable: a broad diffuse zone cut by an edge looks
# narrow from one side. Such paths are rejected.
MAX_ONE_SIDED_FRACTION = 0.5
# A skeleton path can run from a real band straight on along a data edge (in
# real photos: the sand base, the surface, lost-correlation patches). Points
# measured from one side only along an edge are unreliable, so the band core
# is judged without such stretches when at least this many other points exist...
MIN_TWO_SIDED_CORE_POINTS = 5
# ...and a stretch that is more than EDGE_HUGGING_SHARE one-sided over this many
# core widths (i.e. one-sided for ~3.5 widths) hugs an edge and breaks the band
# there. A band leaving the surface or the wall-top corner at a shallow angle is
# legitimately one-sided for up to ~3 widths (radial band at ~28 deg), so it stays.
EDGE_HUGGING_WIDTHS = 5.0
EDGE_HUGGING_SHARE = 0.7
# End runs of centreline points separated from the rest by a gap larger than
# this (grid steps) and shorter than one band width are stray points.
STRAY_GAP_GRID = 3.0


@dataclass
class DetectionSettings:
    min_width_mm: float = 5.0         # narrowest band considered (also limited by grid spacing)
    max_width_mm: float = 80.0        # wider zones are diffuse deformation, not bands
    noise_threshold_k: float = 8.0    # ridge threshold, in robust noise standard deviations
    relative_floor: float = 0.03      # ...and at least this fraction of the strongest ridge
    hysteresis_ratio: float = 0.5     # low threshold / high threshold (keeps bands continuous)
    min_length_mm: float = 50.0       # shorter paths are ignored
    attach_tolerance_mm: float = 30.0 # how close an end must be to the toe / wall / top
                                      # (at least 2 band widths, see _label_bands)
    tip_width_factor: float = 2.0     # band ends where it is this much wider than its core
    lift_off_span_mm: float = 50.0    # lift-off angle = band direction over its first span


@dataclass
class DetectedBand:
    name: str
    label: str                  # primary | radial | intermediate | wall-interface | unattached
    points: np.ndarray          # (n, 2) centreline in mm, ordered from the attachment end
    width_mm: float             # median FWHM of shear strain across the band
    width_profile_mm: np.ndarray
    peak_gamma: float           # median peak shear strain above the local background
    snr: float
    reached_end: bool           # primary: reached the soil surface
    flags: list[str] = field(default_factory=list)

    @property
    def length_mm(self) -> float:
        seg = np.diff(self.points, axis=0)
        return float(np.hypot(seg[:, 0], seg[:, 1]).sum())

    @property
    def tip(self) -> np.ndarray:
        return self.points[-1]


@dataclass
class PrimaryMetrics:
    shape: ShapeResult
    lift_off_deg: float
    lift_off_sigma_deg: float
    length_mm: float            # distance from wall to where band-1 meets the surface
    area_mm2: float             # wedge area between wall, surface and band-1
    heave_mm: float             # max upward surface displacement over the wedge
    developed_length_mm: float  # arc length of the localised band-1
    flags: list[str] = field(default_factory=list)


@dataclass
class FrameBands:
    bands: list[DetectedBand]
    primary: PrimaryMetrics | None
    gamma_noise: float
    ridge_threshold: float

    @property
    def needs_review(self) -> bool:
        flagged = any(b.flags for b in self.bands)
        return flagged or bool(self.primary and self.primary.flags)

    def by_label(self, label: str) -> list[DetectedBand]:
        return [b for b in self.bands if b.label == label]


def _grid_spacing(x: np.ndarray, z: np.ndarray) -> float:
    dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
    dz = float(np.nanmedian(np.abs(np.diff(z, axis=0))))
    if abs(dx - dz) > ISOTROPY_TOLERANCE * dx:
        raise ValueError(f"grid spacing must be equal in x and z (got {dx} and {dz} mm)")
    return dx


def _to_mm(x: np.ndarray, z: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    coords = [rows, cols]
    return np.column_stack([
        ndimage.map_coordinates(x, coords, order=1, mode="nearest"),
        ndimage.map_coordinates(z, coords, order=1, mode="nearest"),
    ])


def _localised_run(line: cl.Centreline, max_width_grid: float, tip_width_factor: float) -> np.ndarray | None:
    """Indices of the band's localised points along a path.

    A path can run from a sharp band into the diffuse zone ahead of its tip.
    The band's core is characterised from its sharpest points; points much
    wider or weaker than that core are not part of the band. Short
    interruptions (up to CROSSING_GAP_WIDTHS core widths) are bridged: where
    another band crosses, the profile across this band includes the other
    one, so a few points look "not localised" although the band continues.
    Only the interrupted points themselves are dropped.
    """
    narrow = np.isfinite(line.fwhm_grid) & (line.fwhm_grid <= max_width_grid)
    if not narrow.any():
        return None
    # The core is the band's sharpest part: a path can be mostly diffuse zone
    # with a short sharp band at one end, so medians would describe the zone.
    core_width, core_peak = _core(line, narrow)
    # A stretch hugging a data edge (long, mostly one-sided) is not part of
    # the band, and its guessed one-sided widths must not set the core.
    hugging = _edge_hugging(line.one_sided, EDGE_HUGGING_WIDTHS * core_width)
    if hugging.any() and (narrow & ~hugging).sum() >= MIN_TWO_SIDED_CORE_POINTS:
        core_width, core_peak = _core(line, narrow & ~hugging)
        hugging = _edge_hugging(line.one_sided, EDGE_HUGGING_WIDTHS * core_width)
    good = narrow & (line.fwhm_grid <= tip_width_factor * core_width) & (line.peak >= TIP_PEAK_FRACTION * core_peak)
    good &= ~hugging
    runs = _runs(good)
    if not runs:
        return None
    max_gap = CROSSING_GAP_WIDTHS * core_width   # path points are about one grid step apart
    merged = [list(runs[0])]
    for start, stop in runs[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    best = max(merged, key=lambda r: good[r[0]:r[1]].sum())
    return best[0] + np.flatnonzero(good[best[0]:best[1]])


def _core(line: cl.Centreline, points: np.ndarray) -> tuple[float, float]:
    """Width and peak of a band's sharpest quarter, over the given points."""
    return (float(np.percentile(line.fwhm_grid[points], CORE_PERCENTILE)),
            float(np.percentile(line.peak[points], 100 - CORE_PERCENTILE)))


def _edge_hugging(one_sided: np.ndarray, window_points: float) -> np.ndarray:
    """Points in a stretch (about window_points long) that is mostly measured from one side."""
    window = max(int(round(window_points)), 1)
    if len(one_sided) < window:
        return np.zeros(len(one_sided), dtype=bool)
    # Beyond the path ends counts as two-sided, so a short one-sided end (a
    # band meeting the surface) is not mistaken for an edge-hugging stretch.
    share = ndimage.uniform_filter1d(one_sided.astype(float), window, mode="constant", cval=0.0)
    # A point hugs an edge if any window containing it is mostly one-sided.
    return (ndimage.maximum_filter1d(share, window, mode="nearest") > EDGE_HUGGING_SHARE) & one_sided


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of consecutive True values."""
    runs, start = [], None
    for i, ok in enumerate(np.append(flags, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    return runs


def detect_bands(
    gamma: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    wall_height_mm: float,
    settings: DetectionSettings | None = None,
    uz: np.ndarray | None = None,
    reconstructed: np.ndarray | None = None,
) -> FrameBands:
    """Detect and measure shear bands in one max-shear-strain field.

    gamma, x, z (and optional vertical displacement uz, mm) are grids of the
    same shape; x, z are reference positions in the wall frame (mm).
    reconstructed marks points whose displacement was refilled across lost
    correlation (strain.fill_displacement_gaps); bands through them are flagged.
    """
    settings = settings or DetectionSettings()
    spacing = _grid_spacing(x, z)
    # Isolated lost DIC points (typically right on a band's centreline, where
    # shear is highest) would punch holes in the ridge and break it into
    # loops. Small holes are bridged with interpolated strain; only large
    # gaps count as missing data.
    valid = _bridge_small_holes(np.isfinite(gamma))
    filled = fill_missing(gamma)
    min_width = max(settings.min_width_mm, 2.0 * spacing)
    ridge, sigma, normal = ridge_strength(gamma, spacing, min_width, settings.max_width_mm)
    ridge[~valid] = 0.0

    med, sd = robust_noise(ridge[valid])
    high = max(med + settings.noise_threshold_k * sd, settings.relative_floor * ridge.max())
    # Thin ridge lines (non-maximum suppression), then hysteresis along them.
    thin = np.where(ridge_maxima(ridge, normal), ridge, 0.0)
    mask = apply_hysteresis_threshold(thin, settings.hysteresis_ratio * high, high)
    band_scale = FWHM_PER_SIGMA * float(np.median(sigma[mask])) if mask.any() else 0.0
    paths = extract_paths(mask, settings.min_length_mm / spacing, band_scale)

    _, gamma_noise = robust_noise((filled - ndimage.median_filter(filled, NOISE_MEDIAN_SIZE))[valid])
    gamma_noise = max(gamma_noise, 1e-12)
    half_window = settings.max_width_mm / spacing
    z_top = float(np.nanmax(np.where(valid, z, np.nan)))

    bands: list[DetectedBand] = []
    for path in paths:
        line = cl.refine(filled, path, half_window, valid)
        keep = _localised_run(line, settings.max_width_mm / spacing, settings.tip_width_factor)
        if keep is None:
            continue
        if line.one_sided[keep].mean() > MAX_ONE_SIDED_FRACTION:
            continue
        rows, cols = line.rows[keep], line.cols[keep]
        width_profile = line.fwhm_grid[keep] * spacing
        width = float(np.nanmedian(width_profile))
        points = _to_mm(x, z, rows, cols)
        hook_free = _trim_hooks(points, width)
        points, width_profile = points[hook_free], width_profile[hook_free]
        rows, cols = rows[hook_free], cols[hook_free]
        connected = _drop_stray_ends(points, width, STRAY_GAP_GRID * spacing)
        points, width_profile = points[connected], width_profile[connected]
        rows, cols = rows[connected], cols[connected]
        if len(points) < 2 or _polyline_length(points) < settings.min_length_mm:
            continue
        if width > settings.max_width_mm:
            continue  # a diffuse zone, not a localised band
        if _along_data_edge(points, x, z, valid, spacing):
            continue
        peak = float(np.nanmedian(line.peak[keep]))
        band = DetectedBand(
            name="", label="", points=points, width_mm=width, width_profile_mm=width_profile,
            peak_gamma=peak, snr=peak / gamma_noise, reached_end=False,
        )
        _band_flags(band, valid, z, rows, cols, settings, width / spacing, reconstructed)
        bands.append(band)

    _label_bands(bands, wall_height_mm, settings.attach_tolerance_mm)
    primary_band = next((b for b in bands if b.label == "primary"), None)
    primary = None
    if primary_band is not None:
        primary = _primary_metrics(primary_band, x, z, uz, valid, wall_height_mm, z_top, spacing, settings)
    return FrameBands(bands=bands, primary=primary, gamma_noise=gamma_noise, ridge_threshold=high)


def _along_data_edge(points: np.ndarray, x: np.ndarray, z: np.ndarray, valid: np.ndarray, spacing: float) -> bool:
    margin = EDGE_BAND_GRID * spacing
    z_top = np.nanmax(np.where(valid, z, np.nan))
    z_bottom = np.nanmin(np.where(valid, z, np.nan))
    x_right = np.nanmax(np.where(valid, x, np.nan))
    near = (points[:, 1] >= z_top - margin) | (points[:, 1] <= z_bottom + margin) | (points[:, 0] >= x_right - margin)
    return bool(near.mean() > EDGE_ARTEFACT_FRACTION)


def _drop_stray_ends(points: np.ndarray, width: float, gap_mm: float) -> slice:
    """Remove short runs of points cut off from the rest of the band by a gap."""
    if len(points) < 3:
        return slice(0, len(points))
    seg = np.hypot(*np.diff(points, axis=0).T)
    breaks = np.flatnonzero(seg > gap_mm)
    start, stop = 0, len(points)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    for b in breaks:                      # leading stray run: points[:b+1]
        if s[b] - s[start] < width:
            start = b + 1
        else:
            break
    for b in breaks[::-1]:                # trailing stray run: points[b+1:]
        if b + 1 <= start:
            break
        if s[stop - 1] - s[b + 1] < width:
            stop = b + 1
        else:
            break
    return slice(start, stop)


def _chord_angle(points: np.ndarray) -> np.ndarray:
    return points[-1] - points[0]


def _end_is_hooked(points: np.ndarray, width: float) -> bool:
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    for scale in HOOK_WINDOW_WIDTHS:
        if s[-1] < (scale + 2) * width:
            continue
        head = points[s <= scale * width]
        body = points[(s >= scale * width) & (s <= (scale + 2) * width)]
        if len(head) < 2 or len(body) < 2:
            continue
        a, b = _chord_angle(head), _chord_angle(body)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) > HOOK_ANGLE_DEG:
            return True
    return False


def _trim_hooks(points: np.ndarray, width: float) -> slice:
    """Index range of a centreline with curled ends removed."""
    start, stop = 0, len(points)
    while stop - start > 2 and _end_is_hooked(points[start:stop], width):
        start += 1
    while stop - start > 2 and _end_is_hooked(points[start:stop][::-1], width):
        stop -= 1
    return slice(start, stop)


def _bridge_small_holes(valid: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(~valid)
    if n == 0:
        return valid
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    small = np.isin(labels, np.flatnonzero(sizes <= SMALL_HOLE_POINTS) + 1)
    return valid | small


def _polyline_length(points: np.ndarray) -> float:
    seg = np.diff(points, axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def _band_flags(band: DetectedBand, valid: np.ndarray, z: np.ndarray, rows, cols, settings: DetectionSettings,
                width_grid: float, reconstructed: np.ndarray | None = None) -> None:
    if band.snr < MIN_SNR:
        band.flags.append("low-contrast")
    if band.width_mm > WIDTH_LIMIT_FRACTION * settings.max_width_mm:
        band.flags.append("width-near-limit")
    neighbourhood = np.zeros(valid.shape, dtype=bool)
    r = np.clip(np.round(rows).astype(int), 0, valid.shape[0] - 1)
    c = np.clip(np.round(cols).astype(int), 0, valid.shape[1] - 1)
    neighbourhood[r, c] = True
    neighbourhood = ndimage.binary_dilation(neighbourhood, iterations=3)
    if (~valid[neighbourhood]).mean() > DATA_GAP_FRACTION:
        band.flags.append("near-data-gap")
    if _ends_at_data_gap(valid, z, np.asarray(rows, float), np.asarray(cols, float), width_grid):
        band.flags.append("ends-at-data-gap")
    if reconstructed is not None and reconstructed[r, c].mean() > RECONSTRUCTED_FRACTION:
        band.flags.append("through-lost-correlation")


def _ends_at_data_gap(valid: np.ndarray, z: np.ndarray, rows: np.ndarray, cols: np.ndarray, width_grid: float) -> bool:
    """Whether the band, continued past either end, runs into missing data inside the soil.

    Where grains inside a band rearrange faster than the photo rate, DIC
    loses correlation exactly on the band, so a band can seem to stop at a
    gap although it continues through it. Missing data above the soil
    surface (air) or below the analysed region is not a gap: those are the
    limits of the soil, where bands legitimately end.
    """
    top = np.where(valid, z, -np.inf).max(axis=0)
    bottom = np.where(valid, z, np.inf).min(axis=0)
    gaps = ~valid & (z < top[None, :]) & (z > bottom[None, :])
    if not gaps.any() or len(rows) < 2:
        return False
    path = np.column_stack([rows, cols])
    reach = END_GAP_WIDTHS * width_grid
    steps = np.arange(1.0, reach + 1e-9, 0.5)
    for end, inner in ((path[0], path[1:]), (path[-1], path[-2::-1])):
        # Direction of the end over about one band width.
        back = inner[np.linalg.norm(inner - end, axis=1) <= max(width_grid, 1.0)]
        away = end - (back[-1] if len(back) else inner[0])
        norm = np.linalg.norm(away)
        if norm == 0:
            continue
        ahead = end[None, :] + np.outer(steps, away / norm)
        r = np.round(ahead[:, 0]).astype(int)
        c = np.round(ahead[:, 1]).astype(int)
        inside = (r >= 0) & (r < gaps.shape[0]) & (c >= 0) & (c < gaps.shape[1])
        if gaps[r[inside], c[inside]].any():
            return True
    return False


def _wall_intercept(band: DetectedBand, from_start: bool, reach_mm: float) -> float | None:
    """z where the band, extended backwards from one end, meets the wall face x = 0.

    None if that end is farther than reach_mm from the wall or points away from it.
    """
    pts = band.points if from_start else band.points[::-1]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    head = pts[s <= max(END_FIT_WIDTHS * band.width_mm, s[min(2, len(s) - 1)])]
    if len(head) < 2 or pts[0, 0] > reach_mm:
        return None
    centre = head.mean(axis=0)
    direction = np.linalg.svd(head - centre, full_matrices=False)[2][0]
    if np.dot(direction, head[0] - head[-1]) < 0:
        direction = -direction           # now points out of the band, towards its end
    if direction[0] >= 0:
        return None                      # this end does not head towards the wall
    t = -pts[0, 0] / direction[0]
    return float(pts[0, 1] + t * direction[1])


def _is_wall_interface(band: DetectedBand, tol: float) -> bool:
    centre = band.points.mean(axis=0)
    direction = np.linalg.svd(band.points - centre, full_matrices=False)[2][0]
    near_vertical = abs(direction[1]) > np.cos(np.radians(WALL_INTERFACE_ANGLE_DEG))
    return bool(near_vertical and np.median(band.points[:, 0]) < tol)


def _label_bands(bands: list[DetectedBand], height: float, tol: float) -> None:
    """Label bands by where they meet the wall and orient them from that end.

    Each end is extended backwards along the band's own direction to the
    wall face; the intercept height decides the label: at the toe ->
    primary, at the top -> radial, in between -> intermediate. Using the
    extended line (not the raw end point) makes labels robust to the few
    centimetres by which measured band ends fall short of the wall.
    """
    base_tol = tol
    primaries = []
    for band in bands:
        # Band ends are measured to within about a band width, so tolerances
        # must be at least that.
        tol = max(base_tol, ATTACH_WIDTHS * band.width_mm)
        reach = max(base_tol, WALL_REACH_WIDTHS * band.width_mm)
        if _is_wall_interface(band, tol):
            band.label = "wall-interface"
            continue
        best = None                      # (priority, from_start)
        for from_start in (True, False):
            z_wall = _wall_intercept(band, from_start, reach)
            if z_wall is None or z_wall < -tol or z_wall > height + tol:
                continue
            if abs(z_wall) <= tol:
                candidate = (0, from_start, "primary")
            elif z_wall >= height - tol:
                candidate = (1, from_start, "radial")
            else:
                candidate = (2, from_start, "intermediate")
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            band.label = "unattached"
            from_start = band.points[0, 0] <= band.points[-1, 0]
        else:
            _, from_start, band.label = best
        if not from_start:
            band.points = band.points[::-1]
            band.width_profile_mm = band.width_profile_mm[::-1]
        if band.label == "primary":
            primaries.append(band)
    if len(primaries) > 1:
        primaries.sort(key=lambda b: b.length_mm, reverse=True)
        for extra in primaries[1:]:
            extra.label = "unattached"
            extra.flags.append("second-band-at-toe")

    counters: dict[str, int] = {}
    prefixes = {"primary": "band-", "radial": "R", "intermediate": "I",
                "wall-interface": "W", "unattached": "U"}
    ordered = sorted(bands, key=lambda b: (b.label, -b.points[0, 1]))
    for band in ordered:
        counters[band.label] = counters.get(band.label, 0) + 1
        band.name = f"{prefixes[band.label]}{counters[band.label]}"


def _primary_metrics(band, x, z, uz, valid, height, z_top, spacing, settings) -> PrimaryMetrics:
    flags: list[str] = []
    shape = classify(_interior(band.points, SHAPE_END_EXCLUSION_WIDTHS * band.width_mm), band.width_mm,
                     full_band=band.points)
    if shape.ambiguous:
        flags.append("shape-ambiguous")
    lift_off, lift_off_sigma, toe_gap = lift_off_from_model(
        shape, np.zeros(2), settings.lift_off_span_mm, band.width_mm)
    if not np.isfinite(lift_off_sigma) or lift_off_sigma > LIFT_OFF_SIGMA_LIMIT_DEG:
        flags.append("lift-off-uncertain")
    if toe_gap > max(settings.attach_tolerance_mm, ATTACH_WIDTHS * band.width_mm):
        flags.append("band-does-not-reach-toe")

    tip = band.points[-1]
    surface_tol = max(settings.attach_tolerance_mm, ATTACH_WIDTHS * band.width_mm)
    band.reached_end = bool(tip[1] >= z_top - surface_tol)
    confirm_tol = max(SURFACE_CONFIRM_MIN_GRID * spacing, SURFACE_CONFIRM_WIDTHS * band.width_mm)
    if band.reached_end and tip[1] < z_top - confirm_tol:
        flags.append("surface-reach-unconfirmed")
    length = area = np.nan
    if band.reached_end:
        exit_x = _surface_intersection(band.points, height, END_FIT_WIDTHS * band.width_mm)
        length = exit_x
        outline = np.vstack([[[0.0, 0.0]], band.points, [[exit_x, height], [0.0, height]]])
        area = _polygon_area(outline)
    else:
        flags.append("not-reached-surface")

    heave = np.nan
    if uz is not None:
        top_row = np.isclose(z, z_top, atol=0.5 * spacing) & valid & np.isfinite(uz)
        reach = (length if np.isfinite(length) else tip[0]) + band.width_mm
        over_wedge = top_row & (x >= 0) & (x <= reach)
        if over_wedge.any():
            heave = float(np.max(uz[over_wedge]))
    return PrimaryMetrics(
        shape=shape, lift_off_deg=lift_off, lift_off_sigma_deg=lift_off_sigma,
        length_mm=length, area_mm2=area, heave_mm=heave,
        developed_length_mm=band.length_mm, flags=flags,
    )


def _interior(points: np.ndarray, margin_mm: float) -> np.ndarray:
    """Points more than margin_mm (arc length) from both ends, if enough remain."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    inner = points[(s >= margin_mm) & (s <= s[-1] - margin_mm)]
    return inner if len(inner) >= max(len(points) // 2, 3) else points


def _surface_intersection(points: np.ndarray, height: float, fit_length_mm: float) -> float:
    """x where the band's end direction meets z = height (straight extrapolation)."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    from_end = np.concatenate([[0.0], np.cumsum(seg[::-1])])[::-1]
    tail = points[from_end <= max(fit_length_mm, seg[-1] * 2)]
    if len(tail) < 2:
        tail = points[-2:]
    centre = tail.mean(axis=0)
    direction = np.linalg.svd(tail - centre)[2][0]
    if direction[1] < 0:
        direction = -direction
    if abs(direction[1]) < 1e-9:
        return float(points[-1, 0])
    return float(centre[0] + (height - centre[1]) * direction[0] / direction[1])


def _polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
