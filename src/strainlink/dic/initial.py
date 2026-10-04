"""Integer-pixel initial guesses for IC-GN.

IC-GN converges only within a few pixels of the true displacement, so each
subset first gets an integer-pixel estimate from normalised
cross-correlation (OpenCV matchTemplate, TM_CCOEFF_NORMED = ZNCC) inside a
search window around a prior guess.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.spatial import cKDTree

# OpenCV's matchTemplate releases the GIL, so subsets are searched in parallel threads.
WORKERS = max(1, (os.cpu_count() or 1))
# An edge subset borrows the integer estimate of a fully-on-object point within this many subset radii.
BORROW_RADII = 3.0


def integer_search(
    reference: np.ndarray,
    current: np.ndarray,
    centres_x: np.ndarray,
    centres_y: np.ndarray,
    radius: int,
    search_radius: int,
    prior_u: np.ndarray | None = None,
    prior_v: np.ndarray | None = None,
    subset_mask=None,
    point_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, peak_zncc) integer displacements per subset.

    Subsets whose template or search window leaves the image get NaN.
    Centres are rounded to the nearest pixel for the template. With
    subset_mask (dic.mask.SubsetMask) and point_ids, a subset that reaches
    past the object's outline is matched on its object pixels only (a
    template that is half background would lock onto the still background).
    """
    n = centres_x.size
    prior_u = np.zeros(n) if prior_u is None else np.nan_to_num(prior_u)
    prior_v = np.zeros(n) if prior_v is None else np.nan_to_num(prior_v)
    ref32 = reference.astype(np.float32)
    cur32 = current.astype(np.float32)
    height, width = current.shape
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    # Edge subsets (partly off the object): a masked search is exact but slow (several FFT
    # correlations each), so it is kept for the ones with no fully-on-object point nearby; the
    # others borrow their neighbour's result afterwards (displacement is continuous, and IC-GN
    # refines from there).
    template_masks: dict[int, np.ndarray] = {}
    partial = np.zeros(n, dtype=bool)
    borrow_from = np.full(n, -1)
    if subset_mask is not None:
        partial = ~subset_mask.full(point_ids)
        full_idx = np.flatnonzero(~partial)
        if partial.any() and full_idx.size:
            tree = cKDTree(np.column_stack([centres_x[full_idx], centres_y[full_idx]]))
            distance, nearest = tree.query(np.column_stack([centres_x[partial], centres_y[partial]]))
            close = distance <= BORROW_RADII * radius
            borrow_from[np.flatnonzero(partial)[close]] = full_idx[nearest[close]]
        alone = np.flatnonzero(partial & (borrow_from < 0))
        if alone.size:
            side = 2 * radius + 1
            weights = subset_mask.weights(point_ids[alone]).astype(np.float32).reshape(-1, side, side)
            template_masks = dict(zip(alone.tolist(), weights))

    def search(indices: range) -> None:
        for i in indices:
            _search_one(i)

    def _search_one(i: int) -> None:
        if borrow_from[i] >= 0:
            return
        tx, ty = int(round(centres_x[i])), int(round(centres_y[i]))
        if not (radius <= tx < reference.shape[1] - radius and radius <= ty < reference.shape[0] - radius):
            return
        template = ref32[ty - radius: ty + radius + 1, tx - radius: tx + radius + 1]
        sx = tx + int(round(prior_u[i]))
        sy = ty + int(round(prior_v[i]))
        x0, x1 = sx - radius - search_radius, sx + radius + search_radius + 1
        y0, y1 = sy - radius - search_radius, sy + radius + search_radius + 1
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x1c - x0c < template.shape[1] or y1c - y0c < template.shape[0]:
            return
        template_mask = template_masks.get(i)
        if template_mask is None:
            response = cv2.matchTemplate(cur32[y0c:y1c, x0c:x1c], template, cv2.TM_CCOEFF_NORMED)
        else:
            response = np.nan_to_num(cv2.matchTemplate(cur32[y0c:y1c, x0c:x1c], template, cv2.TM_CCOEFF_NORMED,
                                                       mask=template_mask), nan=-1.0)
        _, best, _, loc = cv2.minMaxLoc(response)
        u[i] = x0c + loc[0] + radius - tx
        v[i] = y0c + loc[1] + radius - ty
        peak[i] = best

    blocks = [range(s, min(s + max(n // WORKERS, 1), n)) for s in range(0, n, max(n // WORKERS, 1))]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(search, blocks))
    borrowers = np.flatnonzero(borrow_from >= 0)
    if borrowers.size:
        source = borrow_from[borrowers]
        # The neighbour's movement relative to its own prior, added to this point's prior.
        u[borrowers] = np.round(prior_u[borrowers]) + (u[source] - np.round(prior_u[source]))
        v[borrowers] = np.round(prior_v[borrowers]) + (v[source] - np.round(prior_v[source]))
        peak[borrowers] = peak[source]
        # The template must still fit inside the photo for IC-GN's start to be meaningful.
        tx, ty = np.round(centres_x[borrowers]), np.round(centres_y[borrowers])
        outside = ~((tx >= radius) & (tx < reference.shape[1] - radius) & (ty >= radius) & (ty < reference.shape[0] - radius))
        u[borrowers[outside]] = v[borrowers[outside]] = peak[borrowers[outside]] = np.nan
    return u, v, peak
