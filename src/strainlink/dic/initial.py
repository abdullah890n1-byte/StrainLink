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

# OpenCV's matchTemplate releases the GIL, so subsets are searched in parallel threads.
WORKERS = max(1, (os.cpu_count() or 1))


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
    template_masks: dict[int, np.ndarray] = {}
    if subset_mask is not None:
        partial = np.flatnonzero(~subset_mask.full(point_ids))
        if partial.size:
            side = 2 * radius + 1
            weights = subset_mask.weights(point_ids[partial]).astype(np.float32).reshape(-1, side, side)
            template_masks = dict(zip(partial.tolist(), weights))

    def search(indices: range) -> None:
        for i in indices:
            _search_one(i)

    def _search_one(i: int) -> None:
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
    return u, v, peak
