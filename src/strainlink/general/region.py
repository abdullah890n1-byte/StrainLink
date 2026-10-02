"""Where the object is in the photo: the textured, in-focus area DIC can measure.

DIC needs a random pattern (speckle, grains). Plain background, air, the
loading frame and defocused regions have little fine texture, so the
object is found as the connected area of high local texture: the standard
deviation of the Laplacian over a subset-sized window. This replaced
drawing an area by hand (Vic-2D's AOI); a rectangle can still limit it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Window (px) over which local texture is measured; about one DIC subset.
TEXTURE_WINDOW_PX = 31
# Texture counts as "object" above this fraction of the typical object texture...
TEXTURE_FRACTION = 0.35
# ...where "typical" is this percentile of texture over the photo (the object
# is assumed to fill at least ~10% of it).
TYPICAL_PERCENTILE = 90
# Smallest detail of the outline kept (px); grain-scale gaps are closed.
OUTLINE_SMOOTHING_PX = 5
# Gaps inside the object smaller than this many texture windows (~subsets) are filled; larger ones
# are real holes (a drilled hole, a notch) and stay out of the measured area.
MAX_FILLED_HOLE_WINDOWS = 2.0


def texture(image: np.ndarray) -> np.ndarray:
    lap = ndimage.laplace(ndimage.gaussian_filter(np.asarray(image, float), 1.0))
    mean = ndimage.uniform_filter(lap, TEXTURE_WINDOW_PX)
    return np.sqrt(np.maximum(ndimage.uniform_filter(lap**2, TEXTURE_WINDOW_PX) - mean**2, 0.0))


def object_mask(image: np.ndarray, roi_px: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Boolean mask of the measurable object (largest textured area), optionally within a rectangle."""
    level = texture(image)
    allowed = np.ones(level.shape, dtype=bool)
    if roi_px is not None:
        x0, y0, x1, y1 = roi_px
        allowed[:] = False
        allowed[max(y0, 0):y1, max(x0, 0):x1] = True
    typical = float(np.percentile(level[allowed], TYPICAL_PERCENTILE))
    mask = (level > TEXTURE_FRACTION * typical) & allowed
    mask = ndimage.binary_opening(mask, iterations=OUTLINE_SMOOTHING_PX)
    labels, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("no textured object found: DIC needs a speckle pattern or natural texture on the object")
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    mask = labels == (1 + int(np.argmax(sizes)))
    # Close grain-scale gaps only: a real hole in the specimen (bigger than a few subsets) is
    # not part of the object and must stay out (filling every hole measured inside it).
    from skimage.morphology import remove_small_holes
    closed = ndimage.binary_closing(mask, iterations=OUTLINE_SMOOTHING_PX)
    return remove_small_holes(closed, max_size=int(MAX_FILLED_HOLE_WINDOWS * TEXTURE_WINDOW_PX**2))
