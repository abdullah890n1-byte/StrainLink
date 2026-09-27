"""Batched inverse-compositional Gauss-Newton (IC-GN) sub-pixel registration.

First-order subset shape function, ZNSSD criterion (Pan 2013, "An evaluation
of convergence criteria for digital image correlation using inverse
compositional Gauss-Newton algorithm"). All subsets in a chunk iterate
together as NumPy arrays; IC-GN suits batching because the Hessian depends
only on the reference subset and is computed once.

Parameter vector per subset: p = [u, du/dx, du/dy, v, dv/dx, dv/dy].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image import SplineImage

N_PARAMS = 6
# Subsets per batch: bounds peak memory (chunk * subset_pixels * 8 bytes
# per working array) to a few hundred MB for 41x41 subsets.
CHUNK_SIZE = 1500
# A subset with (almost) uniform intensity has no texture to correlate.
MIN_INTENSITY_SPREAD = 1e-6


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


def _params_from_matrices(m: np.ndarray) -> np.ndarray:
    return np.stack(
        [m[:, 0, 2], m[:, 0, 0] - 1.0, m[:, 0, 1], m[:, 1, 2], m[:, 1, 0], m[:, 1, 1] - 1.0],
        axis=1,
    )


def _zero_normalise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = values - values.mean(axis=1, keepdims=True)
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
) -> ICGNResult:
    """Refine displacement parameters for subsets centred at (centres_x, centres_y)
    in the reference image. Centres may be non-integer (incremental tracking)."""
    settings = settings or ICGNSettings()
    n = centres_x.size
    params = np.array(initial_params, dtype=np.float64, copy=True)
    zncc = np.full(n, np.nan)
    iterations = np.zeros(n, dtype=int)
    converged = np.zeros(n, dtype=bool)
    for start in range(0, n, CHUNK_SIZE):
        sl = slice(start, min(start + CHUNK_SIZE, n))
        p, z, it, ok = _refine_chunk(
            reference, current, centres_x[sl], centres_y[sl], params[sl], radius, settings
        )
        params[sl], zncc[sl], iterations[sl], converged[sl] = p, z, it, ok
    return ICGNResult(params=params, zncc=zncc, iterations=iterations, converged=converged)


def _refine_chunk(reference, current, cx, cy, params, radius, settings):
    dx, dy = subset_offsets(radius)
    ref_x = cx[:, None] + dx[None, :]
    ref_y = cy[:, None] + dy[None, :]

    f = reference.sample(ref_x, ref_y)
    fx, fy = reference.sample_gradients(ref_x, ref_y)
    f_centred, f_spread = _zero_normalise(f)
    textured = f_spread > MIN_INTENSITY_SPREAD

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
        g_centred, g_spread = _zero_normalise(g)
        g_spread = np.where(g_spread > MIN_INTENSITY_SPREAD, g_spread, np.nan)

        residual = f_centred[idx] - (f_spread[idx] / g_spread)[:, None] * g_centred
        gradient = np.einsum("nsi,ns->ni", jac[idx], residual)
        dp = -np.einsum("nij,nj->ni", hessian_inv[idx], gradient)

        bad = ~np.all(np.isfinite(dp), axis=1)
        dp[bad] = 0.0
        # Inverse-compositional update: W(p) <- W(p) o W(dp)^-1.
        warps[idx] = m @ np.linalg.inv(_warp_matrices(dp))

        step = np.sqrt(
            dp[:, 0] ** 2 + dp[:, 3] ** 2
            + (radius * dp[:, [1, 2, 4, 5]]) ** 2 @ np.ones(4)
        )
        iterations[idx] = iteration
        done = (step < settings.convergence_px) & ~bad
        converged[idx[done]] = True
        active[idx[done | bad]] = False

    params_out = _params_from_matrices(warps)
    zncc = _final_zncc(current, cx, cy, warps, dx, dy, f_centred, f_spread)
    zncc[~solvable] = np.nan
    return params_out, zncc, iterations, converged


def _final_zncc(current, cx, cy, warps, dx, dy, f_centred, f_spread):
    wx = cx[:, None] + warps[:, 0, 0, None] * dx + warps[:, 0, 1, None] * dy + warps[:, 0, 2, None]
    wy = cy[:, None] + warps[:, 1, 0, None] * dx + warps[:, 1, 1, None] * dy + warps[:, 1, 2, None]
    g_centred, g_spread = _zero_normalise(current.sample(wx, wy))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (f_centred * g_centred).sum(axis=1) / (f_spread * g_spread)
