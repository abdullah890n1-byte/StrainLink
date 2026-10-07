"""How accurate each result of an analysis is, stated with the result (95% uncertainty).

The random part is measured from the analysis itself: the fine-scale
scatter of a field about its own smooth trend. Calibrated against the
Vic-2D task suite (20 cases with exact answers, `scripts/validate_accuracy.py`):

- per point, 2.5 x the local scatter contains the true strain error for at
  least 95% of points in every case (clean, noisy, low-contrast, blurred,
  compressed, every file format): the multiple needed was 1.65-2.63;
- displacement noise follows from the strain scatter through the strain
  window (a plane fit over w x w points turns displacement noise s into
  strain noise s / (spacing * sqrt(sum of x^2))); 2 x that covers at least
  95% of true displacement errors on lossless images (needed: 0.8-1.8),
  and strongly compressed images with their allowance.

The systematic part cannot be seen in the data, so validated allowances
are added: the largest bias of a whole-object strain in the suite was
24 microstrain on lossless images (38 at 30% stretch, 0.01% of the value)
and 1.2% of the value with strong JPEG compression.

Not included, because the images cannot show them: movement towards or
away from the camera, an uncorrected lens, a wrong scale, and features
narrower than the subset and strain window. These are listed with every
statement (`NOT_INCLUDED`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Multiple of the local scatter that contained 95% of true errors in every suite case.
COVERAGE_FACTOR = 2.5
# The same for displacement derived from the strain scatter (needed: 0.8-1.8 on lossless images).
DISPLACEMENT_FACTOR = 2.0
# Neighbourhood (grid points) over which the smooth trend and the local scatter are taken:
# three strain windows, so the noise (correlated over about one window) is not taken for trend.
SCATTER_WINDOW = 15
# Rings of points next to the edge of the measured area that take their own scatter (about one
# strain window plus the subset), and the fewest points a ring needs for that.
EDGE_RINGS = 6
MIN_RING_POINTS = 12
# Systematic allowances for whole-object values (largest biases found in the suite, rounded up).
BIAS_FLOOR_STRAIN = 25e-6
BIAS_FRACTION_LOSSLESS = 0.003
BIAS_FRACTION_COMPRESSED = 0.015
# Displacement in compressed images: JPEG's blocks bias the matching by a share of the movement
# (strong JPEG, quality 60: 95% of errors within 2.2% of it once interpolation error was removed).
DISPLACEMENT_FRACTION_COMPRESSED = 0.03
# Whole-object averages: neighbouring points share their noise over about one strain window.
DEFAULT_STRAIN_WINDOW = 5
# A value is "small compared with the noise" when its uncertainty exceeds this share of it.
POOR_RELATIVE = 0.2

NOT_INCLUDED = (
    "movement towards or away from the camera (2D DIC reads it as strain: 0.5% closer shows as 0.5% strain)",
    "lens distortion, unless a calibration-grid photo was given",
    "an error in the scale (it changes displacements in mm, not strains)",
    "features narrower than the subset and strain window, which are smoothed",
)


@dataclass
class Scatter:
    local: np.ndarray      # scatter around each point (same shape as the field; NaN where no data)
    typical: float         # robust typical scatter of the whole field


def scatter(field: np.ndarray, window: int = SCATTER_WINDOW) -> Scatter:
    """Fine-scale scatter of a field about its own smooth trend (local median)."""
    known = np.isfinite(field)
    if known.sum() < 9:
        return Scatter(np.full(field.shape, np.nan), float("nan"))
    size = int(max(3, min(window, *(2 * (n // 2) - 1 for n in field.shape))))      # odd, fits the grid
    filled = np.where(known, field, np.nanmedian(field))
    trend = ndimage.median_filter(filled, size=size, mode="nearest")
    residual = np.where(known, field - trend, 0.0)
    weight = ndimage.uniform_filter(known.astype(float), size, mode="constant")
    local = np.sqrt(np.maximum(ndimage.uniform_filter(residual**2, size, mode="constant"), 0.0) / np.maximum(weight, 1e-9))
    values = residual[known]
    typical = float(1.4826 * np.median(np.abs(values - np.median(values))))
    # Never below the field's typical scatter: a quiet patch is not more accurate than the method.
    local = np.maximum(local, typical)
    # Points next to the edge of the measured area are noisier than their inward neighbours
    # (partial subsets, one-sided strain fits, material leaving the photo), and the local
    # scatter around them is mostly taken from those quieter neighbours. Each ring of edge
    # points therefore takes at least the scatter of that ring itself.
    depth = ndimage.distance_transform_cdt(np.pad(known, 1), metric="chessboard")[1:-1, 1:-1]
    for ring in range(1, EDGE_RINGS + 1):
        members = known & (depth == ring)
        if members.sum() >= MIN_RING_POINTS:
            local[members] = np.maximum(local[members], float(np.sqrt(np.mean(residual[members] ** 2))))
    return Scatter(np.where(known, local, np.nan), typical)


def point_uncertainty(field: np.ndarray, compressed: bool = False) -> np.ndarray:
    """95% uncertainty of each point of a strain-like field (same unit as the field)."""
    random = COVERAGE_FACTOR * scatter(field).local
    if not compressed:
        return random
    return np.sqrt(random**2 + (BIAS_FRACTION_COMPRESSED * np.abs(field)) ** 2)


def window_factor(window: int = DEFAULT_STRAIN_WINDOW) -> float:
    """sqrt(sum of x^2) over a window x window plane fit, in grid steps."""
    return float(np.sqrt(window * window * (window * window - 1) / 12.0))


def displacement_uncertainty(strain_along: np.ndarray, displacement: np.ndarray, spacing: float,
                             window: int = DEFAULT_STRAIN_WINDOW, compressed: bool = False) -> np.ndarray:
    """95% uncertainty of a displacement component, in the unit of `spacing` (the grid spacing).

    strain_along: the normal strain along that component (its scatter carries the displacement noise).
    """
    found = scatter(strain_along)
    # Edge points can have a displacement but no strain: they take the field's typical scatter.
    local = np.where(np.isfinite(found.local), found.local, found.typical)
    random = np.where(np.isfinite(displacement), DISPLACEMENT_FACTOR * local * abs(spacing) * window_factor(window), np.nan)
    if not compressed:
        return random
    return np.sqrt(random**2 + (DISPLACEMENT_FRACTION_COMPRESSED * np.abs(displacement)) ** 2)


def average_uncertainty(value: float, typical_scatter: float, points: int, compressed: bool = False,
                        window: int = DEFAULT_STRAIN_WINDOW) -> float:
    """95% uncertainty of a strain averaged (median or mean) over `points` grid points."""
    if not np.isfinite(value) or not np.isfinite(typical_scatter) or points < 1:
        return float("nan")
    independent = max(points / float(window * window), 1.0)
    random = COVERAGE_FACTOR * typical_scatter / np.sqrt(independent)
    fraction = BIAS_FRACTION_COMPRESSED if compressed else BIAS_FRACTION_LOSSLESS
    return float(np.sqrt(random**2 + BIAS_FLOOR_STRAIN**2 + (fraction * value) ** 2))


def relative(value: float, uncertainty: float) -> float | None:
    """Uncertainty as a share of the value (None when the value is too small for that to mean anything)."""
    if not (np.isfinite(value) and np.isfinite(uncertainty)) or abs(value) <= uncertainty:
        return None
    return float(uncertainty / abs(value))


def describe(value: float, uncertainty: float) -> str:
    """'within 1.2% of the value', or how the noise compares when the value is too small."""
    share = relative(value, uncertainty)
    if share is None:
        return "smaller than the measurement noise"
    return f"within {100 * share:.1f}% of the value" if share >= 0.001 else "within 0.1% of the value"
