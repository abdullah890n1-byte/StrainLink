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
# Edge-aware fit (order 2): a quadratic fit is used where the window is cut by an edge, if its
# normal equations are at least this well determined relative to a window cut in half by a
# straight edge; otherwise the plane fit stays.
MIN_QUADRATIC_DETERMINANT = 0.25


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
    field: np.ndarray, spacing_x: float, spacing_y: float, window: int = 5, order: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """d(field)/dx and d(field)/dy on a regular grid.

    spacing_x: coordinate increment per column; spacing_y: per row (may be
    negative, e.g. when rows run downward but y points upward).

    order 2 (edge-aware): where the window is cut by an edge or a gap, a
    plane fit returns the gradient at the middle of the points it has, not
    at the point itself, which smooths strain next to holes and edges. A
    quadratic fit removes that bias. On a full window its gradient equals
    the plane fit's exactly (odd and even terms are orthogonal there), so
    only cut windows change.
    """
    dx, dy = _plane_gradients(field, window)
    if order == 2:
        qx, qy, cut = _quadratic_gradients(field, window)
        dx, dy = np.where(cut & np.isfinite(dx), qx, dx), np.where(cut & np.isfinite(dx), qy, dy)
    return dx / spacing_x, dy / spacing_y


def _plane_gradients(field: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Plane-fit gradients per grid step."""
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
    return solution[..., 1], solution[..., 2]


def _quadratic_gradients(field: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadratic-fit gradients per grid step where the window is cut, and where that fit is usable."""
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    terms = [np.ones_like(ox), ox, oy, ox * ox, ox * oy, oy * oy]
    known = np.isfinite(field)
    mask = known.astype(np.float64)
    values = np.where(known, field, 0.0)

    def local_sum(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return ndimage.correlate(data, kernel, mode="constant", cval=0.0)

    cut = known & (local_sum(mask, terms[0]) < window * window)
    rows, cols = np.nonzero(cut)
    if rows.size == 0:
        return np.full(field.shape, np.nan), np.full(field.shape, np.nan), cut
    normal = np.empty((rows.size, 6, 6))
    for i in range(6):
        for j in range(i, 6):
            normal[:, i, j] = normal[:, j, i] = local_sum(mask, terms[i] * terms[j])[rows, cols]
    rhs = np.stack([local_sum(values, t)[rows, cols] for t in terms], -1)
    # Reference: the same window cut in half by a straight edge (rows 0..half kept).
    keep = oy >= 0
    reference = np.array([[np.sum((terms[i] * terms[j])[keep]) for j in range(6)] for i in range(6)])
    usable = np.linalg.det(normal) > MIN_QUADRATIC_DETERMINANT * np.linalg.det(reference)
    gx, gy = np.full(field.shape, np.nan), np.full(field.shape, np.nan)
    if usable.any():
        sol = np.linalg.solve(normal[usable], rhs[usable][..., None])[..., 0]
        gx[rows[usable], cols[usable]], gy[rows[usable], cols[usable]] = sol[:, 1], sol[:, 2]
    cut[rows[~usable], cols[~usable]] = False
    return gx, gy, cut


def compute_strain(
    u: np.ndarray, v: np.ndarray, spacing_x: float, spacing_y: float, window: int = 5, order: int = 1
) -> StrainField:
    """Strain from displacement components u (along x) and v (along y) on a grid.

    order 2: edge-aware gradients (see displacement_gradients).
    """
    dudx, dudy = displacement_gradients(u, spacing_x, spacing_y, window, order)
    dvdx, dvdy = displacement_gradients(v, spacing_x, spacing_y, window, order)

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


# Lost-correlation gaps are filled only this close to measured data (grid
# steps): a band's lost strip is a few steps wide, wider holes stay empty.
MAX_GAP_FILL_DISTANCE_GRID = 4.0


def fill_displacement_gaps(
    u: np.ndarray, v: np.ndarray, lost: np.ndarray, max_distance_grid: float = MAX_GAP_FILL_DISTANCE_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill narrow lost-correlation gaps in the displacement from the measured points around them.

    Inside an active shear band the grains rearrange between photos, so DIC
    loses exactly the points on the band, and strain there is undefined.
    The soil on both sides is still measured, and the difference between
    the sides is the band's slip; interpolating the *displacement* across
    the gap (harmonic, i.e. the smoothest field matching both sides) puts
    that slip back into the strain field, so a band is not cut where it is
    most active. The strain inside the filled strip is spread over the
    strip's width, so a band's width there is an upper bound; callers flag
    it (returned mask).

    lost: points that are soil (tracked before) but have no displacement
    now. Only those within max_distance_grid of measured points are filled.
    Returns (u_filled, v_filled, filled_mask).
    """
    measured = np.isfinite(u) & np.isfinite(v)
    if not lost.any() or not measured.any():
        return u, v, np.zeros(u.shape, dtype=bool)
    distance = ndimage.distance_transform_edt(~measured)
    target = lost & ~measured & (distance <= max_distance_grid)
    if not target.any():
        return u, v, target
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    index = -np.ones(u.shape, dtype=int)
    rows_t, cols_t = np.nonzero(target)
    index[rows_t, cols_t] = np.arange(len(rows_t))
    i_list, j_list, values = [], [], []
    rhs_u = np.zeros(len(rows_t))
    rhs_v = np.zeros(len(rows_t))
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = rows_t + dr, cols_t + dc
        inside = (r >= 0) & (r < u.shape[0]) & (c >= 0) & (c < u.shape[1])
        k = np.flatnonzero(inside)
        rr, cc = r[k], c[k]
        unknown = index[rr, cc] >= 0
        known = measured[rr, cc]
        # Each neighbour that is unknown or measured contributes to the discrete Laplacian;
        # neighbours with no data at all (air, wall) are left out (a free edge).
        use = unknown | known
        i_list.append(k[use]); j_list.append(k[use]); values.append(np.ones(use.sum()))
        i_list.append(k[unknown]); j_list.append(index[rr[unknown], cc[unknown]]); values.append(-np.ones(unknown.sum()))
        np.add.at(rhs_u, k[known], u[rr[known], cc[known]])
        np.add.at(rhs_v, k[known], v[rr[known], cc[known]])
    n = len(rows_t)
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(i_list), np.concatenate(j_list))), shape=(n, n)).tocsr()
    # Unknowns with no usable neighbour at all would make the system singular; pin them weakly.
    matrix = matrix + coo_matrix((np.full(n, 1e-9), (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
    u_new, v_new = u.copy(), v.copy()
    u_new[rows_t, cols_t] = spsolve(matrix, rhs_u)
    v_new[rows_t, cols_t] = spsolve(matrix, rhs_v)
    return u_new, v_new, target


def measures(strain: StrainField) -> dict[str, np.ndarray]:
    """Further strain measures engineers use (Vic-2D offers them too), from the same gradients.

    - engineering strains: change in length of a line along x (z) per unit
      length, sqrt(C) - 1, and the engineering shear angle (change of a right angle);
    - Euler-Almansi strain (current configuration): e = (I - B^-1) / 2, B = F F^T;
    - von Mises equivalent strain from the Hencky principal strains, with the
      out-of-plane strain from volume conservation (e3 = -(e1 + e2)):
      e_vm = (2 / sqrt(3)) sqrt(e1^2 + e1 e2 + e2^2);
    - rotation of the material (polar decomposition), degrees, counter-clockwise
      in the x-z frame.
    """
    f11, f12, f21, f22 = 1.0 + strain.dudx, strain.dudy, strain.dvdx, 1.0 + strain.dvdy
    c11 = f11**2 + f21**2
    c22 = f12**2 + f22**2
    c12 = f11 * f12 + f21 * f22
    with np.errstate(invalid="ignore", divide="ignore"):
        exx_eng = np.sqrt(c11) - 1.0
        eyy_eng = np.sqrt(c22) - 1.0
        shear_eng = np.arcsin(np.clip(c12 / np.sqrt(c11 * c22), -1.0, 1.0))
        b11 = f11**2 + f12**2
        b22 = f21**2 + f22**2
        b12 = f11 * f21 + f12 * f22
        det_b = b11 * b22 - b12**2
        inv11, inv22, inv12 = b22 / det_b, b11 / det_b, -b12 / det_b
        e1, e2 = strain.e1, strain.e2
        return {
            "exx_eng": exx_eng, "eyy_eng": eyy_eng, "shear_eng": shear_eng,
            "almansi_xx": 0.5 * (1.0 - inv11), "almansi_yy": 0.5 * (1.0 - inv22), "almansi_xy": -0.5 * inv12,
            "von_mises": (2.0 / np.sqrt(3.0)) * np.sqrt(e1**2 + e1 * e2 + e2**2),
            "rotation_deg": np.degrees(np.arctan2(f21 - f12, f11 + f22)),
        }
