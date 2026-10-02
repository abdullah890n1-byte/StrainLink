"""Markers: an independent check of DIC on a real test.

A marker is a small high-contrast dot fixed to the object (a coloured bead
or pin in the soil against the glass, a sticker on a specimen, a dark
pebble in sand). It is located in every photo from its own shape: the
centre of its dark (or bright) spot, weighted by how much it stands out
from the local background. That is a different measurement from DIC,
which matches the speckle pattern inside a square subset, so it shares
none of DIC's error sources (pattern, subset shape function,
interpolation). The difference between the two at the marker is a direct
accuracy measure on real images. See Markers.md.

1. detect: round blobs of marker size that are dark (bright) over a whole
   disc, and distinctly darker (brighter) than the object's own texture;
2. track: in each photo, look near the predicted position (previous
   position + previous motion) for a blob of similar size and contrast;
   take its weighted centroid;
3. compare: DIC displacement interpolated at the marker's first-photo
   position vs the marker's own displacement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# Marker cores must exceed the texture's own core values (this percentile over the
# object) by at least MIN_CONTRAST_GREY grey levels...
TEXTURE_CORE_PERCENTILE = 99.5
MIN_CONTRAST_GREY = 8.0
# ...where the texture's darkest (brightest) level is this local percentile.
TEXTURE_PERCENTILE = 5
# ...and markers are the group of candidates distinctly darker (brighter) than the rest.
SEPARATION_SIGMAS = 3.0
# Marker cores: the signal eroded by a disc of this fraction of the smallest marker diameter.
CORE_FRACTION = 0.35
# Minimum roundness: ratio of the blob's short to long axis (second moments; 1 = a disc).
MIN_ROUNDNESS = 0.5
# The local texture reference is taken over a window this many marker diameters wide,
# estimated on a copy reduced so a marker spans about BACKGROUND_REDUCTION_PX pixels.
BACKGROUND_DIAMETERS = 3.0
BACKGROUND_REDUCTION_PX = 4
# Search window around the predicted position, in marker diameters.
SEARCH_DIAMETERS = 1.5
# A marker is lost when its area changes by more than this factor (occluded, merged, crushed)...
AREA_CHANGE_LIMIT = 2.0
# ...or it keeps less than this share of its first-photo contrast (it is texture, not the marker).
LEVEL_KEPT = 0.6
MAX_MARKERS = 60
# A marker disagreeing with DIC by more than this many times the median (and at least
# OUTLIER_MIN_PX) is listed as possibly mis-tracked.
OUTLIER_FACTOR = 5.0
OUTLIER_MIN_PX = 1.0
# The centroid's reference is the median grey level in a ring this far (px) outside the marker.
RING_INNER_PX = 2
RING_OUTER_PX = 4
# The centre is refined inside a circle this much (px) wider than the marker, this many times.
APERTURE_MARGIN_PX = 1.5
APERTURE_ITERATIONS = 3


@dataclass
class MarkerTracks:
    x: np.ndarray            # (frames, markers) centroid column, px (NaN when lost)
    y: np.ndarray            # (frames, markers) centroid row, px
    diameter_px: np.ndarray  # (markers,) equivalent diameter in the first photo
    polarity: str            # "dark" or "bright"

    @property
    def count(self) -> int:
        return self.x.shape[1]


@dataclass
class MarkerComparison:
    per_marker_rms_px: np.ndarray            # (markers,) RMS DIC - marker difference over the test
    rms_px: float                            # over all markers and photos (outside zones)
    bias_px: tuple[float, float]             # mean difference (u, v)
    max_px: float
    markers_used: int
    markers_in_zones: list[int] = field(default_factory=list)
    rms_in_zones_px: float = float("nan")


# --- image measures ---------------------------------------------------------------

def _signal(image: np.ndarray, polarity: str, diameter_px: float) -> np.ndarray:
    """How far each pixel goes beyond the darkest (brightest) part of the object's own texture nearby.

    The reference is a local low (high) percentile of the grey levels: the
    texture's own dark (bright) phase sits at about zero, and only things
    darker (brighter) than anything in the texture, i.e. markers, are
    positive. Being local, it follows uneven lighting. It is estimated on a
    reduced copy (a full-size filter this wide took minutes per test).
    """
    image = np.asarray(image, dtype=float)
    factor = max(int(diameter_px // BACKGROUND_REDUCTION_PX), 1)
    small = image[::factor, ::factor]
    size = max(int(BACKGROUND_DIAMETERS * diameter_px / factor) | 1, 3)
    percentile = TEXTURE_PERCENTILE if polarity == "dark" else 100 - TEXTURE_PERCENTILE
    reference = ndimage.zoom(ndimage.percentile_filter(small, percentile, size=size),
                             (image.shape[0] / small.shape[0], image.shape[1] / small.shape[1]), order=1)
    reference = np.pad(reference, [(0, max(image.shape[0] - reference.shape[0], 0)),
                                    (0, max(image.shape[1] - reference.shape[1], 0))], mode="edge")
    reference = reference[:image.shape[0], :image.shape[1]]
    return reference - image if polarity == "dark" else image - reference


def _disc(radius: float) -> np.ndarray:
    r = max(int(np.ceil(radius)), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx**2 + yy**2) <= radius**2


def _robust_sigma(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values)))) or 1e-9


def _roundness(region: np.ndarray) -> float:
    rows, cols = np.nonzero(region)
    if len(rows) < 5:
        return 0.0
    eig = np.linalg.eigvalsh(np.cov(np.vstack([cols, rows])))
    return float(np.sqrt(max(eig[0], 0.0) / max(eig[1], 1e-12)))


def _centroid(image: np.ndarray, region: np.ndarray, polarity: str) -> tuple[float, float] | None:
    """Centroid of a marker, weighted by how much each pixel is darker (brighter) than its surroundings.

    The reference is one constant per marker, the median grey level of a thin
    ring around it, which moves with the material: a sloping reference
    (interpolated from the coarse local-texture map) tugged the centre as the
    object moved (0.07 px in testing).
    """
    grown = ndimage.binary_dilation(region, iterations=1)
    ring = ndimage.binary_dilation(region, iterations=RING_OUTER_PX) & ~ndimage.binary_dilation(region, iterations=RING_INNER_PX)
    if not ring.any():
        return None
    base = float(np.median(image[ring]))
    contrast = np.clip(base - image if polarity == "dark" else image - base, 0, None)
    weights = np.where(grown, contrast, 0.0)
    if weights.sum() <= 0:
        return None
    rows, cols = np.nonzero(grown)
    w = weights[rows, cols]
    cx, cy = float((cols * w).sum() / w.sum()), float((rows * w).sum() / w.sum())
    # Refine inside a fixed circle around the centre: a threshold-shaped region's outline
    # flickers from photo to photo, a circle of fixed size doesn't.
    radius = np.sqrt(region.sum() / np.pi) + APERTURE_MARGIN_PX
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    for _ in range(APERTURE_ITERATIONS):
        # Soft-edged aperture: pixel weight falls from 1 to 0 across the circle's edge.
        aperture = np.clip(radius + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0)
        w = aperture * contrast
        total = w.sum()
        if total <= 0:
            return None
        cx, cy = float((xx * w).sum() / total), float((yy * w).sum() / total)
    return cx, cy


def _core(signal: np.ndarray, min_diameter: float) -> np.ndarray:
    return ndimage.grey_erosion(signal, footprint=_disc(CORE_FRACTION * min_diameter))


def _threshold(signal: np.ndarray, min_diameter: float, allowed: np.ndarray | None = None) -> float:
    """Marker cores must beat the texture's own highest core values by a clear margin.

    Markers cover well under 1% of the object, so a high percentile of the
    core values over the object describes the texture, not the markers.
    """
    core = _core(signal, min_diameter)
    values = core[allowed] if allowed is not None else core
    return max(MIN_CONTRAST_GREY, float(np.percentile(values, TEXTURE_CORE_PERCENTILE)) + MIN_CONTRAST_GREY)


def _candidates(signal: np.ndarray, min_diameter: float, max_diameter: float, threshold: float,
                allowed: np.ndarray | None = None):
    """Round blobs of marker size, dark (bright) over a whole disc: (region, diameter, mean signal).

    Speckle and sand grains are small, so eroding the signal with a disc of
    CORE_FRACTION of the smallest marker diameter removes most of them; each
    remaining core is grown back to its blob's outline (half its peak).
    """
    core = ndimage.grey_erosion(signal, footprint=_disc(CORE_FRACTION * min_diameter))
    cores = core > threshold
    if allowed is not None:
        cores &= allowed
    labels, n = ndimage.label(cores)
    found = []
    for i in range(1, n + 1):
        seed = labels == i
        peak = float(signal[seed].max())
        grown = ndimage.binary_propagation(seed, mask=signal > 0.5 * peak)
        diameter = 2.0 * np.sqrt(grown.sum() / np.pi)
        if min_diameter <= diameter <= max_diameter and _roundness(grown) >= MIN_ROUNDNESS:
            found.append((grown, diameter, float(signal[grown].mean())))
    return found


def _markers_only(candidates: list) -> list:
    """The candidates distinctly darker (brighter) than the object's own texture.

    A textured object has marker-sized dark clusters of its own; markers are
    chosen (or are, like pebbles in sand) much darker. The candidates' mean
    contrasts are split in two (Otsu); the upper group is kept when the gap
    is clear (SEPARATION_SIGMAS pooled standard deviations), else all are kept.
    """
    if len(candidates) < 3:
        return candidates
    from skimage.filters import threshold_otsu
    means = np.array([c[2] for c in candidates])
    cut = threshold_otsu(means)
    high, low = means[means > cut], means[means <= cut]
    if len(high) and len(low):
        pooled = np.sqrt((high.var() * len(high) + low.var() * len(low)) / len(means)) or 1e-9
        if (high.mean() - low.mean()) / pooled >= SEPARATION_SIGMAS:
            return [c for c in candidates if c[2] > cut]
    return candidates


# --- detection and tracking -------------------------------------------------------

def detect(image: np.ndarray, diameter_px: tuple[float, float] = (6.0, 40.0), polarity: str = "dark",
           mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marker centres (N, 2) as (col, row), diameters (N,) and mean contrasts (N,) in the first photo."""
    signal = _signal(image, polarity, diameter_px[1])
    threshold = _threshold(signal, diameter_px[0], mask)
    regions = sorted(_markers_only(_candidates(signal, *diameter_px, threshold, allowed=mask)), key=lambda r: -r[2])
    centres, diameters, levels = [], [], []
    image = np.asarray(image, dtype=float)
    for region, diameter, level in regions[:MAX_MARKERS]:
        c = _centroid(image, region, polarity)
        if c is not None:
            centres.append(c)
            diameters.append(diameter)
            levels.append(level)
    return np.asarray(centres).reshape(-1, 2), np.asarray(diameters), np.asarray(levels)


def track(images, centres: np.ndarray, diameters: np.ndarray, levels: np.ndarray, polarity: str = "dark",
          diameter_px: tuple[float, float] = (6.0, 40.0), progress=None) -> MarkerTracks:
    """Follow each marker through the photos by its centroid (predicted from its previous motion)."""
    n_frames, n = len(images), len(centres)
    xs = np.full((n_frames, n), np.nan)
    ys = np.full((n_frames, n), np.nan)
    if n == 0:
        return MarkerTracks(xs, ys, np.empty(0), polarity)
    xs[0], ys[0] = centres[:, 0], centres[:, 1]
    for k in range(1, n_frames):
        image = np.asarray(images[k], dtype=float)
        signal = _signal(image, polarity, diameter_px[1])
        threshold = _threshold(signal, diameter_px[0])
        for m in range(n):
            if not np.isfinite(xs[k - 1, m]):
                continue
            moving = k >= 2 and np.isfinite(xs[k - 2, m])
            px = xs[k - 1, m] + (xs[k - 1, m] - xs[k - 2, m] if moving else 0.0)
            py = ys[k - 1, m] + (ys[k - 1, m] - ys[k - 2, m] if moving else 0.0)
            found = _find_near(image, signal, polarity, px, py, diameters[m], levels[m], diameter_px, threshold)
            if found is not None:
                xs[k, m], ys[k, m] = found
        if progress:
            progress(k, n_frames - 1)
    return MarkerTracks(xs, ys, np.asarray(diameters), polarity)


def _find_near(image: np.ndarray, signal: np.ndarray, polarity: str, px: float, py: float, diameter: float, level: float,
               diameter_px: tuple[float, float], threshold: float):
    """The marker near its predicted position: similar size, similar contrast, nearest."""
    r = int(np.ceil(SEARCH_DIAMETERS * diameter + diameter_px[1]))
    r0, r1 = max(int(py) - r, 0), min(int(py) + r + 1, signal.shape[0])
    c0, c1 = max(int(px) - r, 0), min(int(px) + r + 1, signal.shape[1])
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    window = signal[r0:r1, c0:c1]
    image_window = image[r0:r1, c0:c1]
    best, best_d = None, np.inf
    for region, found_diameter, found_level in _candidates(window, diameter_px[0], diameter_px[1], threshold):
        if not (diameter / AREA_CHANGE_LIMIT ** 0.5 <= found_diameter <= diameter * AREA_CHANGE_LIMIT ** 0.5):
            continue
        if found_level < LEVEL_KEPT * level:
            continue
        c = _centroid(image_window, region, polarity)
        if c is None:
            continue
        d = np.hypot(c[0] + c0 - px, c[1] + r0 - py)
        if d < best_d:
            best, best_d = (c[0] + c0, c[1] + r0), d
    return best if best is not None and best_d <= SEARCH_DIAMETERS * diameter else None


# --- comparison with DIC ------------------------------------------------------------

def compare(tracks: MarkerTracks, dic_u_px: np.ndarray, dic_v_px: np.ndarray, grid_cols: np.ndarray,
            grid_rows: np.ndarray, zone_mask: np.ndarray | None = None) -> MarkerComparison:
    """DIC displacement at each marker's first-photo position vs the marker's own displacement.

    dic_u_px, dic_v_px: (frames, ny, nx) DIC displacement (px, image axes) on a regular grid
    of first-photo positions grid_cols (nx,), grid_rows (ny,). zone_mask (ny, nx) marks
    points inside localisation zones, where a point marker and a DIC subset (which averages
    over its window) may legitimately differ; those markers are reported separately.
    """
    from .bands.ridges import fill_missing
    col_idx = (tracks.x[0] - grid_cols[0]) / (grid_cols[1] - grid_cols[0])
    row_idx = (tracks.y[0] - grid_rows[0]) / (grid_rows[1] - grid_rows[0])
    coords = np.stack([row_idx, col_idx])
    inside_grid = (row_idx >= 0) & (row_idx <= len(grid_rows) - 1) & (col_idx >= 0) & (col_idx <= len(grid_cols) - 1)
    in_zone = np.zeros(tracks.count, dtype=bool)
    if zone_mask is not None and zone_mask.any():
        in_zone = ndimage.map_coordinates(zone_mask.astype(float), coords, order=0, mode="nearest") > 0.5
    du, dv = [], []
    for k in range(1, tracks.x.shape[0]):
        measured = np.isfinite(dic_u_px[k])
        if not measured.any():
            du.append(np.full(tracks.count, np.nan))
            dv.append(np.full(tracks.count, np.nan))
            continue
        # DIC at the marker: interpolated only where the DIC points around it were measured.
        near = ndimage.map_coordinates(measured.astype(float), coords, order=1, mode="constant") > 0.99
        u = ndimage.map_coordinates(fill_missing(dic_u_px[k]), coords, order=1, mode="nearest")
        v = ndimage.map_coordinates(fill_missing(dic_v_px[k]), coords, order=1, mode="nearest")
        ok = near & inside_grid & np.isfinite(tracks.x[k])
        du.append(np.where(ok, u - (tracks.x[k] - tracks.x[0]), np.nan))
        dv.append(np.where(ok, v - (tracks.y[k] - tracks.y[0]), np.nan))
    du, dv = np.array(du), np.array(dv)
    err = np.hypot(du, dv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # markers never compared stay NaN, by design
        per_marker = np.sqrt(np.nanmean(err**2, axis=0))
    use = np.isfinite(per_marker) & ~in_zone
    zone_used = np.isfinite(per_marker) & in_zone
    sel = err[:, use]
    finite = np.isfinite(sel).any()
    return MarkerComparison(
        per_marker_rms_px=per_marker,
        rms_px=float(np.sqrt(np.nanmean(sel**2))) if finite else float("nan"),
        bias_px=(float(np.nanmean(du[:, use])) if finite else float("nan"),
                 float(np.nanmean(dv[:, use])) if finite else float("nan")),
        max_px=float(np.nanmax(sel)) if finite else float("nan"),
        markers_used=int(use.sum()),
        markers_in_zones=[int(i) for i in np.flatnonzero(zone_used)],
        rms_in_zones_px=float(np.sqrt(np.nanmean(err[:, zone_used] ** 2))) if zone_used.any() else float("nan"),
    )


class ColourBrightness:
    """Photos seen as the brightness of their brightest colour (max of R, G, B).

    On coloured objects (pink and blue sand) grey conversion makes coloured
    grains as dark as a grey pebble; in this channel coloured grains stay
    bright and only things dark in every colour (pebbles, black beads) are dark.
    """

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, k: int) -> np.ndarray:
        from PIL import Image
        with Image.open(self.paths[k]) as im:
            return np.asarray(im.convert("RGB"), dtype=float).max(axis=2)


def check_dic(images, object_mask: np.ndarray | None, grid_cols: np.ndarray, grid_rows: np.ndarray,
              u_px: np.ndarray, v_px: np.ndarray, zone_mask: np.ndarray | None, mm_per_px: float,
              polarity: str = "dark", diameter_px: tuple[float, float] = (6.0, 40.0)) -> tuple[dict, MarkerTracks]:
    """Detect and track the markers, compare them with DIC, and summarise (px and mm)."""
    centres, diameters, levels = detect(images[0], diameter_px, polarity, object_mask)
    tracks = track(images, centres, diameters, levels, polarity, diameter_px)
    result = {"found": int(tracks.count), "polarity": polarity, "diameter_px": [float(d) for d in diameter_px]}
    if tracks.count == 0:
        return result, tracks
    cmp = compare(tracks, u_px, v_px, grid_cols, grid_rows, zone_mask)
    per = cmp.per_marker_rms_px
    outside = np.array([np.isfinite(v) and i not in cmp.markers_in_zones for i, v in enumerate(per)])
    median = float(np.median(per[outside])) if outside.any() else float("nan")
    # A marker far from its neighbours' agreement was most likely mis-tracked (it jumped to a
    # nearby dark grain, or a natural pebble rolled); it is listed, not hidden.
    outliers = [int(i) for i in np.flatnonzero(outside & (per > max(OUTLIER_FACTOR * median, OUTLIER_MIN_PX)))]
    result.update({
        "median_px": median, "median_mm": median * mm_per_px, "disagreeing": outliers,
        "followed_to_end": int(np.isfinite(tracks.x[-1]).sum()), "used": cmp.markers_used,
        "in_zones": cmp.markers_in_zones, "rms_px": cmp.rms_px, "rms_mm": cmp.rms_px * mm_per_px,
        "max_px": cmp.max_px, "max_mm": cmp.max_px * mm_per_px, "bias_px": list(cmp.bias_px),
        "rms_in_zones_px": cmp.rms_in_zones_px,
        "per_marker_rms_px": [None if not np.isfinite(v) else float(v) for v in cmp.per_marker_rms_px],
        "first_positions_px": tracks.x[0].round(2).tolist(), "first_rows_px": tracks.y[0].round(2).tolist(),
        "diameters_px": np.round(tracks.diameter_px, 1).tolist(),
    })
    return result, tracks
