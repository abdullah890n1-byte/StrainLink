"""Sub-grid centreline and width of a band from strain profiles across it.

For each point on a skeleton path, the shear-strain profile is sampled
along the local normal (cubic interpolation). The width is the full width
at half maximum above the local background, and the band centre is the
midpoint of the two half-maximum crossings. That midpoint is unbiased for
any symmetric profile and does not snap to grid nodes the way a peak
estimate on interpolated data does. Skeleton pixels are only accurate to
about half a grid step; this refinement is what makes centreline errors
small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Profile sampling step across the band (grid units).
PROFILE_STEP_GRID = 0.25
# Smoothing of the path before taking tangents (grid units, Gaussian sigma).
TANGENT_SMOOTHING_GRID = 2.0


@dataclass
class Centreline:
    rows: np.ndarray      # sub-grid centre positions
    cols: np.ndarray
    fwhm_grid: np.ndarray # band width at each point (grid units, NaN if undefined)
    peak: np.ndarray      # peak shear strain above background
    background: np.ndarray
    one_sided: np.ndarray # width measured from one side only (other side cut off by an edge or gap)


def _tangents(path: np.ndarray) -> np.ndarray:
    smooth = ndimage.gaussian_filter1d(path.astype(float), TANGENT_SMOOTHING_GRID, axis=0, mode="nearest")
    tangent = np.gradient(smooth, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.where(norm > 0, norm, 1.0)


def refine(
    gamma_filled: np.ndarray, path: np.ndarray, half_window_grid: float, valid: np.ndarray | None = None,
) -> Centreline:
    """Refine an ordered (row, col) path.

    gamma_filled must be hole-free (for interpolation); `valid` marks real
    data. Profile samples outside the grid or on invalid data are ignored,
    so bands next to the soil surface or a data gap are measured from the
    side that has data.
    """
    tangent = _tangents(path)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offsets = np.arange(-half_window_grid, half_window_grid + 1e-9, PROFILE_STEP_GRID)
    rows = path[:, 0, None] + normal[:, 0, None] * offsets[None, :]
    cols = path[:, 1, None] + normal[:, 1, None] * offsets[None, :]
    profiles = ndimage.map_coordinates(gamma_filled, [rows.ravel(), cols.ravel()],
                                       order=3, mode="nearest").reshape(rows.shape)
    inside = (rows >= 0) & (rows <= gamma_filled.shape[0] - 1) & (cols >= 0) & (cols <= gamma_filled.shape[1] - 1)
    if valid is not None:
        nearest_valid = ndimage.map_coordinates(valid.astype(float), [rows.ravel(), cols.ravel()],
                                                order=0, mode="nearest").reshape(rows.shape) > 0.5
        inside &= nearest_valid
    profiles = np.where(inside, profiles, np.nan)

    n = len(path)
    centre_offset = np.zeros(n)
    fwhm = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    background = np.full(n, np.nan)
    one_sided = np.zeros(n, dtype=bool)
    near = np.abs(offsets) <= 2.0
    for i, profile in enumerate(profiles):
        # Search the peak near the skeleton point (within 2 grid units) so a
        # neighbouring band cannot capture the profile.
        candidates = np.flatnonzero(near & np.isfinite(profile))
        if len(candidates) == 0:
            continue
        k = int(candidates[np.nanargmax(profile[candidates])])
        k = int(np.clip(k, 1, len(offsets) - 2))
        y0, y1, y2 = profile[k - 1: k + 2]
        curvature = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / curvature if np.isfinite(curvature) and curvature < 0 else 0.0
        centre_offset[i] = offsets[k] + np.clip(shift, -1, 1) * PROFILE_STEP_GRID
        base = np.nanmin(profile)
        height = profile[k] - base
        background[i], peak[i] = base, height
        if not np.isfinite(height) or height <= 0:
            continue
        left, right = _crossings(offsets, profile - base, k, 0.5 * height)
        fwhm[i] = _fwhm(offsets[k], left, right)
        if np.isfinite(left) and np.isfinite(right):
            centre_offset[i] = 0.5 * (left + right)
        else:
            one_sided[i] = True

    return Centreline(
        rows=path[:, 0] + normal[:, 0] * centre_offset,
        cols=path[:, 1] + normal[:, 1] * centre_offset,
        fwhm_grid=fwhm, peak=peak, background=background, one_sided=one_sided,
    )


def _crossing(offsets: np.ndarray, profile: np.ndarray, k: int, half: float, step: int) -> float:
    """Offset of the half-maximum crossing on one side of index k (NaN if none)."""
    j = k
    while 0 <= j + step < len(profile) and np.isfinite(profile[j + step]) and profile[j] > half:
        j += step
    if profile[j] > half:
        return np.nan  # ran into the window edge, the data edge or a gap
    return float(np.interp(half, sorted([profile[j], profile[j - step]]),
                           [offsets[j], offsets[j - step]] if profile[j] <= profile[j - step]
                           else [offsets[j - step], offsets[j]]))


def _crossings(offsets: np.ndarray, profile: np.ndarray, k: int, half: float) -> tuple[float, float]:
    return _crossing(offsets, profile, k, half, -1), _crossing(offsets, profile, k, half, +1)


def _fwhm(peak_offset: float, left: float, right: float) -> float:
    """Full width at half maximum from the two crossings.

    If only one side has a crossing (band next to a data edge), the band is
    assumed symmetric and twice that half-width is returned.
    """
    if np.isfinite(left) and np.isfinite(right):
        return right - left
    one_side = abs(peak_offset - left) if np.isfinite(left) else abs(right - peak_offset)
    return 2.0 * one_side if np.isfinite(one_side) else np.nan
