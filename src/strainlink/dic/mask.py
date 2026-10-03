"""Edge-aware subsets: each subset uses only the pixels that lie on the object.

A square subset that reaches past the object's outline (a hole, a crack
face, the specimen's edge) also sees background that does not move with
the material, which pulls its displacement towards zero and smooths the
strain next to the edge. With an object mask, every subset is weighted by
the mask, so only material pixels take part in the matching.

The mask is drawn on the first photo. Under incremental tracking the
subset is centred on the point's position in a later reference photo, so
each subset pixel is mapped back to the first photo through the point's
accumulated deformation (the linear part of its subset warps, composed at
each reference update) before the mask is read. A subset deep inside the
object has every weight equal to one and is computed exactly as without a
mask.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .icgn import subset_offsets

# Pixels this close to the outline are left out: their grey level can be a blend of
# object and background (lens blur), so they move with neither. The automatic outline
# already errs inside the object (general.region), so one pixel is enough there.
EDGE_MARGIN_PX = 1
# A subset needs at least this share of its pixels on the object (a point on a
# straight edge has about half).
MIN_FILL = 0.4


class SubsetMask:
    """Per-subset weights (0 or 1) from an object mask on the first photo."""

    def __init__(self, mask: np.ndarray, x0: np.ndarray, y0: np.ndarray, radius: int):
        mask = np.asarray(mask, dtype=bool)
        self.mask = ndimage.binary_erosion(mask, iterations=EDGE_MARGIN_PX) if EDGE_MARGIN_PX else mask
        self.x0, self.y0 = np.asarray(x0, float), np.asarray(y0, float)
        self.radius = radius
        self.dx, self.dy = subset_offsets(radius)
        n = self.x0.size
        # Accumulated deformation of each point since the first photo (linear part, 2x2).
        self.linear = np.tile(np.eye(2), (n, 1, 1))
        # Distance from each point to the outline in the first photo: points far inside skip the
        # pixel-by-pixel weights (see full()).
        clearance = ndimage.distance_transform_edt(self.mask)
        finite = np.isfinite(self.x0) & np.isfinite(self.y0)
        rows = np.clip(np.round(np.where(finite, self.y0, 0.0)).astype(int), 0, mask.shape[0] - 1)
        cols = np.clip(np.round(np.where(finite, self.x0, 0.0)).astype(int), 0, mask.shape[1] - 1)
        self._clearance = np.where(finite, clearance[rows, cols], 0.0)

    def update(self, params: np.ndarray, moved: np.ndarray) -> None:
        """Compose the warps of a new reference photo (params from the old reference, (N, 6))."""
        grad = np.zeros((params.shape[0], 2, 2))
        grad[:, 0, 0], grad[:, 0, 1] = 1.0 + params[:, 1], params[:, 2]
        grad[:, 1, 0], grad[:, 1, 1] = params[:, 4], 1.0 + params[:, 5]
        ok = moved & np.all(np.isfinite(params), axis=1)
        self.linear[ok] = grad[ok] @ self.linear[ok]

    def full(self, idx: np.ndarray) -> np.ndarray:
        """Points whose subset certainly lies wholly on the object (no weights needed)."""
        # The subset's corner reaches radius*sqrt(2) in the reference; mapped back it reaches at most
        # that times the largest stretch of the inverse deformation.
        inv_stretch = 1.0 / np.maximum(_smallest_singular(self.linear[idx]), 1e-6)
        return self._clearance[idx] > self.radius * np.sqrt(2.0) * inv_stretch + 1.0

    def weights(self, idx: np.ndarray) -> np.ndarray:
        """Weights (len(idx), S) of the subsets of points idx, in their current reference photo."""
        inv = np.linalg.inv(self.linear[idx])
        bx = self.x0[idx, None] + inv[:, 0, 0, None] * self.dx + inv[:, 0, 1, None] * self.dy
        by = self.y0[idx, None] + inv[:, 1, 0, None] * self.dx + inv[:, 1, 1, None] * self.dy
        rows = np.round(by).astype(int)
        cols = np.round(bx).astype(int)
        h, w = self.mask.shape
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.zeros(bx.shape)
        out[inside] = self.mask[rows[inside], cols[inside]]
        return out

    def chunk_weights(self, idx: np.ndarray) -> np.ndarray | None:
        """Weights for a batch of points, or None when every subset in it is wholly on the object."""
        full = self.full(idx)
        if full.all():
            return None
        out = np.ones((idx.size, self.dx.size))
        out[~full] = self.weights(idx[~full])
        return out

    def fill(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Share of each subset's pixels on the object."""
        idx = np.arange(self.x0.size) if idx is None else np.asarray(idx)
        out = np.ones(idx.size)
        partial = ~self.full(idx)
        if partial.any():
            out[partial] = self.weights(idx[partial]).mean(axis=1)
        return out


def _smallest_singular(m: np.ndarray) -> np.ndarray:
    """Smallest singular value of 2x2 matrices (N, 2, 2), closed form."""
    a, b, c, d = m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1]
    s1 = a * a + b * b + c * c + d * d
    s2 = np.sqrt(np.maximum((a * a + b * b - c * c - d * d) ** 2 + 4 * (a * c + b * d) ** 2, 0.0))
    return np.sqrt(np.maximum(0.5 * (s1 - s2), 0.0))
