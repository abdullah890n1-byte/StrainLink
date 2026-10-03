"""Batched inverse-compositional Gauss-Newton (IC-GN) sub-pixel registration.

First-order subset shape function, ZNSSD criterion (Pan 2013, "An evaluation
of convergence criteria for digital image correlation using inverse
compositional Gauss-Newton algorithm"). All subsets in a chunk iterate
together as NumPy arrays; IC-GN suits batching because the Hessian depends
only on the reference subset and is computed once.

Parameter vector per subset: p = [u, du/dx, du/dy, v, dv/dx, dv/dy].
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .image import SplineImage

N_PARAMS = 6
# Subsets per batch. Each batch needs ~25 MB of working arrays for 21x21
# subsets and WORKERS batches run at once. 1500 (~90 MB each, ~720 MB with 8
# workers) is suspected of pushing an 8 GB Mac into swapping (multi-minute
# stalls in testing; fix not yet confirmed, see Bugs.md).
CHUNK_SIZE = 400
# A subset with (almost) uniform intensity has no texture to correlate.
MIN_INTENSITY_SPREAD = 1e-6
# Chunks are solved in parallel threads: the heavy work (spline sampling,
# einsum) runs in C and releases the GIL, measured 3.3x faster on 8 cores.
WORKERS = max(1, (os.cpu_count() or 1))


@dataclass
class ICGNSettings:
    max_iterations: int = 50
    # Convergence on ||dp|| with gradient terms scaled by subset radius,
    # i.e. the largest pixel movement anywhere in the subset (px).
    convergence_px: float = 1e-3


@dataclass
class ICGNResult:
    params: np.ndarray        # (N, 6)
    zncc: np.ndarray          # (N,) zero-normalised cross-correlation, 1 = perfect
    iterations: np.ndarray    # (N,)
    converged: np.ndarray     # (N,) bool


def subset_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel offsets (dx, dy) of a square subset of side 2*radius+1, flattened."""
    span = np.arange(-radius, radius + 1, dtype=np.float64)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    return dx.ravel(), dy.ravel()


def _warp_matrices(p: np.ndarray) -> np.ndarray:
    """Homogeneous 3x3 affine matrices for first-order warps, shape (N, 3, 3)."""
    n = p.shape[0]
    m = np.zeros((n, 3, 3))
    m[:, 0, 0] = 1.0 + p[:, 1]
    m[:, 0, 1] = p[:, 2]
    m[:, 0, 2] = p[:, 0]
    m[:, 1, 0] = p[:, 4]
    m[:, 1, 1] = 1.0 + p[:, 5]
    m[:, 1, 2] = p[:, 3]
    m[:, 2, 2] = 1.0
    return m


def _invert_affine(m: np.ndarray) -> np.ndarray:
    """Exact inverse of homogeneous 2D affine matrices (N, 3, 3), without LAPACK.

    Called every iteration for every subset; a closed form is faster and keeps
    linear-algebra library threads out of the parallel workers.
    """
    a, b, tx = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    c, d, ty = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    det = a * d - b * c
    out = np.zeros_like(m)
    out[:, 0, 0], out[:, 0, 1] = d / det, -b / det
    out[:, 1, 0], out[:, 1, 1] = -c / det, a / det
    out[:, 0, 2] = -(out[:, 0, 0] * tx + out[:, 0, 1] * ty)
    out[:, 1, 2] = -(out[:, 1, 0] * tx + out[:, 1, 1] * ty)
    out[:, 2, 2] = 1.0
    return out


def _params_from_matrices(m: np.ndarray) -> np.ndarray:
    return np.stack(
        [m[:, 0, 2], m[:, 0, 0] - 1.0, m[:, 0, 1], m[:, 1, 2], m[:, 1, 0], m[:, 1, 1] - 1.0],
        axis=1,
    )


def _zero_normalise(values: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean values and their spread; with weights, over the weighted pixels only (others become 0)."""
    if weights is None:
        centred = values - values.mean(axis=1, keepdims=True)
    else:
        mean = (values * weights).sum(axis=1, keepdims=True) / np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        centred = (values - mean) * weights
    spread = np.sqrt((centred**2).sum(axis=1))
    return centred, spread


def refine(
    reference: SplineImage,
    current: SplineImage,
    centres_x: np.ndarray,
    centres_y: np.ndarray,
    initial_params: np.ndarray,
    radius: int,
    settings: ICGNSettings | None = None,
    subset_mask=None,
    point_ids: np.ndarray | None = None,
) -> ICGNResult:
    """Refine displacement parameters for subsets centred at (centres_x, centres_y)
    in the reference image. Centres may be non-integer (incremental tracking).

    subset_mask (a dic.mask.SubsetMask) with point_ids (each subset's point index)
    makes the subsets edge-aware: only pixels on the object take part.
    """
    settings = settings or ICGNSettings()
    n = centres_x.size
    params = np.array(initial_params, dtype=np.float64, copy=True)
    zncc = np.full(n, np.nan)
    iterations = np.zeros(n, dtype=int)
    converged = np.zeros(n, dtype=bool)
    slices = [slice(s, min(s + CHUNK_SIZE, n)) for s in range(0, n, CHUNK_SIZE)]

    def solve(sl: slice):
        weights = None if subset_mask is None else subset_mask.chunk_weights(point_ids[sl])
        return sl, _refine_chunk(reference, current, centres_x[sl], centres_y[sl], params[sl], radius, settings, weights)

    with ThreadPoolExecutor(max_workers=min(WORKERS, max(len(slices), 1))) as pool:
        for sl, (p, z, it, ok) in pool.map(solve, slices):
            params[sl], zncc[sl], iterations[sl], converged[sl] = p, z, it, ok
    return ICGNResult(params=params, zncc=zncc, iterations=iterations, converged=converged)


def _refine_chunk(reference, current, cx, cy, params, radius, settings, weights=None):
    dx, dy = subset_offsets(radius)
    ref_x = cx[:, None] + dx[None, :]
    ref_y = cy[:, None] + dy[None, :]

    f = reference.sample(ref_x, ref_y)
    fx, fy = reference.sample_gradients(ref_x, ref_y)
    f_centred, f_spread = _zero_normalise(f, weights)
    textured = f_spread > MIN_INTENSITY_SPREAD
    if weights is not None:
        # Pixels off the object contribute nothing to the Hessian or the residual.
        fx, fy = fx * weights, fy * weights

    # Steepest-descent images: grad(f) * dW/dp, shape (N, S, 6).
    jac = np.stack([fx, fx * dx, fx * dy, fy, fy * dx, fy * dy], axis=2)
    hessian = np.einsum("nsi,nsj->nij", jac, jac)
    # Singular Hessians (flat subsets) are excluded instead of crashing the batch.
    solvable = textured & (np.linalg.cond(hessian) < 1e12)
    hessian_inv = np.zeros_like(hessian)
    hessian_inv[solvable] = np.linalg.inv(hessian[solvable])

    n = cx.size
    active = solvable.copy()
    iterations = np.zeros(n, dtype=int)
    converged = np.zeros(n, dtype=bool)
    warps = _warp_matrices(params)

    for iteration in range(1, settings.max_iterations + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        m = warps[idx]
        wx = cx[idx, None] + m[:, 0, 0, None] * dx + m[:, 0, 1, None] * dy + m[:, 0, 2, None]
        wy = cy[idx, None] + m[:, 1, 0, None] * dx + m[:, 1, 1, None] * dy + m[:, 1, 2, None]
        g = current.sample(wx, wy)
        g_centred, g_spread = _zero_normalise(g, None if weights is None else weights[idx])
        g_spread = np.where(g_spread > MIN_INTENSITY_SPREAD, g_spread, np.nan)

        residual = f_centred[idx] - (f_spread[idx] / g_spread)[:, None] * g_centred
        gradient = np.einsum("nsi,ns->ni", jac[idx], residual)
        dp = -np.einsum("nij,nj->ni", hessian_inv[idx], gradient)

        bad = ~np.all(np.isfinite(dp), axis=1)
        dp[bad] = 0.0
        # Inverse-compositional update: W(p) <- W(p) o W(dp)^-1.
        warps[idx] = m @ _invert_affine(_warp_matrices(dp))

        step = np.sqrt(
            dp[:, 0] ** 2 + dp[:, 3] ** 2
            + (radius * dp[:, [1, 2, 4, 5]]) ** 2 @ np.ones(4)
        )
        iterations[idx] = iteration
        done = (step < settings.convergence_px) & ~bad
        converged[idx[done]] = True
        active[idx[done | bad]] = False

    params_out = _params_from_matrices(warps)
    zncc = _final_zncc(current, cx, cy, warps, dx, dy, f_centred, f_spread, weights)
    zncc[~solvable] = np.nan
    return params_out, zncc, iterations, converged


def _final_zncc(current, cx, cy, warps, dx, dy, f_centred, f_spread, weights=None):
    wx = cx[:, None] + warps[:, 0, 0, None] * dx + warps[:, 0, 1, None] * dy + warps[:, 0, 2, None]
    wy = cy[:, None] + warps[:, 1, 0, None] * dx + warps[:, 1, 1, None] * dy + warps[:, 1, 2, None]
    g_centred, g_spread = _zero_normalise(current.sample(wx, wy), weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (f_centred * g_centred).sum(axis=1) / (f_spread * g_spread)
