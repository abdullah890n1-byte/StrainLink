"""Ridge (band) enhancement on a shear-strain field.

A shear band is a narrow ridge of high shear strain. Ridges are found from
the Hessian of the strain field: across a bright ridge the second
derivative is strongly negative. The response is scale-normalised
(sigma^2 * |lambda|) and maximised over scales up to the widest band we
accept, so narrow localised bands respond strongly while broad diffuse
zones (pre-localisation deformation) respond weakly.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# Converts a Gaussian sigma to its full width at half maximum.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
# Number of scales tried between the smallest and largest band width.
N_SCALES = 5
# Smallest Hessian scale in grid units; below this the discrete Hessian is unreliable.
MIN_SIGMA_GRID = 0.8


def fill_missing(field: np.ndarray) -> np.ndarray:
    """Replace NaNs with the nearest valid value (for filtering only)."""
    missing = ~np.isfinite(field)
    if not missing.any():
        return field
    if missing.all():
        raise ValueError("field has no valid values")
    _, (rows, cols) = ndimage.distance_transform_edt(missing, return_indices=True)
    return field[rows, cols]


def ridge_strength(
    gamma: np.ndarray, spacing_mm: float, min_width_mm: float, max_width_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale-normalised ridge strength and the best scale (sigma, grid units) per point."""
    filled = fill_missing(gamma)
    sigma_lo = max(min_width_mm / FWHM_PER_SIGMA / spacing_mm, MIN_SIGMA_GRID)
    sigma_hi = max(max_width_mm / FWHM_PER_SIGMA / spacing_mm, sigma_lo)
    sigmas = np.geomspace(sigma_lo, sigma_hi, N_SCALES)
    best = np.zeros_like(filled)
    best_sigma = np.full_like(filled, sigmas[0])
    for sigma in sigmas:
        elements = hessian_matrix(filled, sigma=sigma, mode="nearest", order="rc",
                                  use_gaussian_derivatives=True)
        # Eigenvalues sorted in decreasing order; the last is the most negative.
        smallest = hessian_matrix_eigvals(elements)[-1]
        response = sigma**2 * np.clip(-smallest, 0.0, None)
        better = response > best
        best[better] = response[better]
        best_sigma[better] = sigma
    return best, best_sigma


def robust_noise(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based standard deviation (robust to the few band points)."""
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, 1.4826 * mad
