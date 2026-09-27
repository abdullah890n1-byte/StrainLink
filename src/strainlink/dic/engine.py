"""High-level DIC: correlate image pairs and track material points through a sequence.

Tracking is Lagrangian: a fixed set of material points defined in the
first image is followed through every frame. With incremental reference
updating the reference image is replaced by a later frame whenever
correlation quality degrades, which keeps subsets correlatable under the
large, localised deformations typical of granular soil (the same idea as
GeoPIV-RG's "leapfrog" updating).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

from .icgn import ICGNSettings, refine
from .image import SplineImage
from .initial import integer_search

ReferenceMode = Literal["fixed", "incremental"]


@dataclass
class DICSettings:
    subset_radius: int = 15          # px; 15 -> 31x31 subset, as in the reference paper
    step: int = 5                    # px grid spacing between subset centres
    search_radius: int = 10          # px integer search around the prior guess
    min_zncc: float = 0.8            # points below this are reported invalid
    reference_mode: ReferenceMode = "incremental"
    # Incremental mode: update the reference when median ZNCC falls below this…
    update_below_zncc: float = 0.95
    # …or after this many frames regardless (0 disables the frame limit).
    update_every: int = 0
    icgn: ICGNSettings = field(default_factory=ICGNSettings)


@dataclass
class Grid:
    """Regular grid of subset centres in the first (undeformed) image, px."""

    x: np.ndarray   # (ny, nx)
    y: np.ndarray   # (ny, nx)

    @classmethod
    def over_roi(cls, roi: tuple[int, int, int, int], step: int, margin: int) -> "Grid":
        """roi = (x0, y0, x1, y1) in px, exclusive end; margin keeps subsets inside."""
        x0, y0, x1, y1 = roi
        xs = np.arange(x0 + margin, x1 - margin, step, dtype=np.float64)
        ys = np.arange(y0 + margin, y1 - margin, step, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        return cls(x=gx, y=gy)

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape  # type: ignore[return-value]


@dataclass
class PairResult:
    u: np.ndarray        # (N,) px
    v: np.ndarray        # (N,) px
    zncc: np.ndarray     # (N,)
    params: np.ndarray   # (N, 6) first-order shape parameters
    valid: np.ndarray    # (N,) bool


@dataclass
class TrackingResult:
    grid: Grid
    x: np.ndarray            # (frames, N) tracked positions, px
    y: np.ndarray            # (frames, N)
    zncc: np.ndarray         # (frames, N)
    valid: np.ndarray        # (frames, N) bool
    reference_frames: list[int]  # frame index used as reference for each frame

    @property
    def u(self) -> np.ndarray:
        """Total displacement since frame 0, px, shape (frames, N)."""
        return self.x - self.x[0]

    @property
    def v(self) -> np.ndarray:
        return self.y - self.y[0]

    def field(self, frame: int, component: Literal["u", "v"]) -> np.ndarray:
        """Displacement component on the grid for one frame, NaN where invalid."""
        values = (self.u if component == "u" else self.v)[frame].copy()
        values[~self.valid[frame]] = np.nan
        return values.reshape(self.grid.shape)


def correlate_pair(
    reference: SplineImage,
    current: SplineImage,
    centres_x: np.ndarray,
    centres_y: np.ndarray,
    settings: DICSettings,
    prior_u: np.ndarray | None = None,
    prior_v: np.ndarray | None = None,
) -> PairResult:
    """Sub-pixel displacement of subsets centred at the given reference positions.

    Centres that are NaN (points lost earlier) are returned invalid.
    """
    n = centres_x.size
    known = np.isfinite(centres_x) & np.isfinite(centres_y)
    if not np.all(known):
        sub = correlate_pair(
            reference, current, centres_x[known], centres_y[known], settings,
            None if prior_u is None else prior_u[known],
            None if prior_v is None else prior_v[known],
        )
        return _scatter(sub, known, n)

    radius = settings.subset_radius
    u0, v0, _ = integer_search(
        reference.pixels, current.pixels, centres_x, centres_y,
        radius, settings.search_radius, prior_u, prior_v,
    )
    searchable = np.isfinite(u0)
    # Rounded template centres shift the integer estimate by the fractional part;
    # correct for it so IC-GN starts as close as possible.
    u0 = u0 + (np.round(centres_x) - centres_x)
    v0 = v0 + (np.round(centres_y) - centres_y)

    initial = np.zeros((centres_x.size, 6))
    initial[:, 0] = np.nan_to_num(u0)
    initial[:, 3] = np.nan_to_num(v0)
    result = refine(reference, current, centres_x, centres_y, initial, radius, settings.icgn)

    zncc = np.where(searchable, result.zncc, np.nan)
    valid = searchable & result.converged & (np.nan_to_num(zncc) >= settings.min_zncc)
    return PairResult(
        u=result.params[:, 0], v=result.params[:, 3], zncc=zncc, params=result.params, valid=valid
    )


def _scatter(sub: PairResult, mask: np.ndarray, n: int) -> PairResult:
    def full(values: np.ndarray, fill) -> np.ndarray:
        out = np.full((n,) + values.shape[1:], fill, dtype=values.dtype)
        out[mask] = values
        return out

    return PairResult(
        u=full(sub.u, np.nan), v=full(sub.v, np.nan), zncc=full(sub.zncc, np.nan),
        params=full(sub.params, np.nan), valid=full(sub.valid, False),
    )


def track_sequence(
    images: Sequence[np.ndarray],
    grid: Grid,
    settings: DICSettings | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> TrackingResult:
    """Track the grid's material points through all images (images[0] = undeformed)."""
    settings = settings or DICSettings()
    n_frames = len(images)
    n_points = grid.x.size
    xs = np.full((n_frames, n_points), np.nan)
    ys = np.full((n_frames, n_points), np.nan)
    zncc = np.full((n_frames, n_points), np.nan)
    valid = np.zeros((n_frames, n_points), dtype=bool)
    xs[0], ys[0] = grid.x.ravel(), grid.y.ravel()
    zncc[0], valid[0] = 1.0, True
    reference_frames = [0]

    ref_index = 0
    reference = SplineImage.from_array(images[0])
    for k in range(1, n_frames):
        current = SplineImage.from_array(images[k], with_gradients=False)
        ref_x, ref_y = xs[ref_index], ys[ref_index]
        # Prior = motion accumulated since the reference (previous frame's result).
        prior_u = xs[k - 1] - ref_x
        prior_v = ys[k - 1] - ref_y
        pair = correlate_pair(reference, current, ref_x, ref_y, settings, prior_u, prior_v)
        still_tracked = valid[ref_index] & pair.valid
        xs[k] = np.where(still_tracked, ref_x + pair.u, np.nan)
        ys[k] = np.where(still_tracked, ref_y + pair.v, np.nan)
        zncc[k] = pair.zncc
        valid[k] = still_tracked
        reference_frames.append(ref_index)

        if settings.reference_mode == "incremental" and _should_update(k, ref_index, pair, settings):
            ref_index = k
            reference = SplineImage.from_array(images[k])
        if progress:
            progress(k, n_frames - 1)

    return TrackingResult(
        grid=grid, x=xs, y=ys, zncc=zncc, valid=valid, reference_frames=reference_frames
    )


def _should_update(k: int, ref_index: int, pair: PairResult, settings: DICSettings) -> bool:
    if not np.any(pair.valid):
        return False
    if np.nanmedian(pair.zncc[pair.valid]) < settings.update_below_zncc:
        return True
    return settings.update_every > 0 and (k - ref_index) >= settings.update_every
