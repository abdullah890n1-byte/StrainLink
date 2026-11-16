"""Measure a scene with StrainLink's own DIC, as a real analysis would (same subsets, grid and settings)."""

from __future__ import annotations

import numpy as np

from ..dic import DICSettings, Grid, track_sequence
from ..dic.mask import MIN_FILL, SubsetMask
from ..general.pipeline import DEFAULT_STEP_PX, DEFAULT_SUBSET_RADIUS_PX
from .features import noise_of
from .scenes import DETECTABLE_SNR, Scene

SETTINGS = DICSettings(subset_radius=DEFAULT_SUBSET_RADIUS_PX, step=DEFAULT_STEP_PX, search_radius=12,
                       reference_mode="fixed")


def measure(scene: Scene) -> dict:
    """DIC of the scene's two photos: displacement (px) and validity on the grid, the grid, and the truth."""
    before, after = scene.render()
    rows, cols = scene.shape
    r = SETTINGS.subset_radius
    grid = Grid.over_roi((r, r, cols - r, rows - r), SETTINGS.step, 0)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    mask = scene.on_specimen(xx, yy)
    fill = SubsetMask(mask, grid.x.ravel(), grid.y.ravel(), r).fill().reshape(grid.shape)
    inside = mask[grid.y.astype(int), grid.x.astype(int)] & (fill >= MIN_FILL)
    grid = Grid(x=np.where(inside, grid.x, np.nan), y=np.where(inside, grid.y, np.nan))
    tracked = track_sequence([before, after], grid, SETTINGS, object_mask=mask if scene.hole else None)
    gx = np.arange(grid.shape[1]) * SETTINGS.step + r
    gy = np.arange(grid.shape[0]) * SETTINGS.step + r
    X, Y = np.meshgrid(gx.astype(float), gy.astype(float))
    u, v = tracked.field(1, "u"), tracked.field(1, "v")
    noise = noise_of(u, v, float(SETTINGS.step))
    truth = scene.truth(X, Y, faint_below=DETECTABLE_SNR * noise, subset_px=2 * r + 1)
    return {"u": u, "v": v, "valid": tracked.valid[1].reshape(grid.shape), "x": X, "y": Y, "truth": truth,
            "jump_px": scene.jump_size(X, Y), "step_px": float(SETTINGS.step), "noise": noise}
