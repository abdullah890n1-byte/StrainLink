"""Strain fields from gridded displacement.

Displacement gradients come from a local least-squares plane fit over a
square window of grid points (the "strain window", equivalent to Vic-2D's
filter size). Missing points (NaN) are simply left out of each local fit,
so holes from lost DIC points do not spread.

Finite-strain measures: Green-Lagrange E = (C - I)/2 and logarithmic
(Hencky) principal strains 0.5*ln(eigenvalues of C). Maximum shear strain
follows the reference paper's definition gamma_max = e1 - e2 (Khan et al.
2026, Eq. 2), on Hencky principal strains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# A local fit needs at least this share of its window populated to be trusted.
MIN_WINDOW_FILL = 0.6
# Relative determinant floor for the 3x3 normal equations (degenerate layouts).
MIN_RELATIVE_DETERMINANT = 1e-9


@dataclass
class StrainField:
    dudx: np.ndarray
    dudy: np.ndarray
    dvdx: np.ndarray
    dvdy: np.ndarray
    exx: np.ndarray        # Green-Lagrange
    eyy: np.ndarray
    exy: np.ndarray        # tensor shear component
    e1: np.ndarray         # Hencky major principal strain
    e2: np.ndarray         # Hencky minor principal strain
    gamma_max: np.ndarray  # e1 - e2
    volumetric: np.ndarray # e1 + e2 = ln(J), positive = dilation

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.gamma_max)


def displacement_gradients(
    field: np.ndarray, spacing_x: float, spacing_y: float, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """d(field)/dx and d(field)/dy on a regular grid.

    spacing_x: coordinate increment per column; spacing_y: per row (may be
    negative, e.g. when rows run downward but y points upward).
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    mask = np.isfinite(field).astype(np.float64)
    values = np.where(mask > 0, field, 0.0)

    def local_sum(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return ndimage.correlate(data, kernel, mode="constant", cval=0.0)

    ones = np.ones_like(ox)
    s1 = local_sum(mask, ones)
    sx = local_sum(mask, ox)
    sy = local_sum(mask, oy)
    sxx = local_sum(mask, ox * ox)
    sxy = local_sum(mask, ox * oy)
    syy = local_sum(mask, oy * oy)
    f1 = local_sum(values, ones)
    fx = local_sum(values, ox)
    fy = local_sum(values, oy)

    normal = np.stack([
        np.stack([s1, sx, sy], -1),
        np.stack([sx, sxx, sxy], -1),
        np.stack([sy, sxy, syy], -1),
    ], -2)
    rhs = np.stack([f1, fx, fy], -1)
    full_window_det = np.linalg.det(np.array([
        [ones.sum(), 0, 0], [0, (ox * ox).sum(), 0], [0, 0, (oy * oy).sum()],
    ]))
    det = np.linalg.det(normal)
    usable = (
        (s1 >= MIN_WINDOW_FILL * window * window)
        & (np.abs(det) > MIN_RELATIVE_DETERMINANT * full_window_det)
        & (mask > 0)
    )
    solution = np.full(rhs.shape, np.nan)
    solution[usable] = np.linalg.solve(normal[usable], rhs[usable][..., None])[..., 0]
    return solution[..., 1] / spacing_x, solution[..., 2] / spacing_y


def compute_strain(
    u: np.ndarray, v: np.ndarray, spacing_x: float, spacing_y: float, window: int = 5
) -> StrainField:
    """Strain from displacement components u (along x) and v (along y) on a grid."""
    dudx, dudy = displacement_gradients(u, spacing_x, spacing_y, window)
    dvdx, dvdy = displacement_gradients(v, spacing_x, spacing_y, window)

    # Right Cauchy-Green tensor C = F^T F with F = I + grad(u).
    f11, f12, f21, f22 = 1.0 + dudx, dudy, dvdx, 1.0 + dvdy
    c11 = f11 * f11 + f21 * f21
    c22 = f12 * f12 + f22 * f22
    c12 = f11 * f12 + f21 * f22
    exx, eyy, exy = 0.5 * (c11 - 1.0), 0.5 * (c22 - 1.0), 0.5 * c12

    mean = 0.5 * (c11 + c22)
    radius = np.sqrt((0.5 * (c11 - c22)) ** 2 + c12**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        e1 = 0.5 * np.log(mean + radius)
        e2 = 0.5 * np.log(mean - radius)
    return StrainField(
        dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy, exx=exx, eyy=eyy, exy=exy,
        e1=e1, e2=e2, gamma_max=e1 - e2, volumetric=e1 + e2,
    )
