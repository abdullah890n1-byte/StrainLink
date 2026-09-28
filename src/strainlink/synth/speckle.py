"""Synthetic speckle images with exactly known displacement fields.

A continuous texture is defined on a supersampled grid; a deformed image
is rendered by pulling each sub-pixel sample back to its reference
position (X = x - u(X), solved by fixed-point iteration) and averaging
the samples in each pixel, as a camera sensor integrates light. The
known u(X) is the ground truth used to validate the DIC engine and the
band detector.

All images produced here are SYNTHETIC and must be labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

DisplacementFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

SUPERSAMPLE = 4
FIXED_POINT_ITERATIONS = 25
# The pull-back iteration stops once no sample moves more than this (px).
FIXED_POINT_TOLERANCE_PX = 1e-3
# 8-bit camera range: grey levels 0..255.
GREY_LEVELS = 255.0


@dataclass
class SpeckleTexture:
    """Continuous reference texture f(X), X in image pixel coordinates."""

    coeffs: np.ndarray   # cubic-spline coefficients of the supersampled texture
    shape: tuple[int, int]

    @classmethod
    def granular(
        cls,
        shape: tuple[int, int],
        grain_radius_px: float = 2.0,
        dark_fraction: float = 0.5,
        seed: int = 0,
    ) -> "SpeckleTexture":
        """Soil-like texture: a random mix of dark and light grains.

        Mimics the paper's speckle layer of white soil mixed with soil
        sprayed black (Khan et al. 2026, Sec. 3.1).
        """
        rng = np.random.default_rng(seed)
        fine_shape = (shape[0] * SUPERSAMPLE, shape[1] * SUPERSAMPLE)
        noise = rng.standard_normal(fine_shape)
        # Smoothing white noise at grain scale gives grains of roughly that size;
        # thresholding at the dark-fraction quantile gives a two-phase mix.
        smooth = ndimage.gaussian_filter(noise, sigma=grain_radius_px * SUPERSAMPLE / 1.5, mode="wrap")
        threshold = np.quantile(smooth, dark_fraction)
        binary = (smooth > threshold).astype(np.float64)
        # Soften grain edges (optical blur) so the texture is band-limited.
        texture = ndimage.gaussian_filter(binary, sigma=0.6 * SUPERSAMPLE, mode="wrap")
        coeffs = ndimage.spline_filter(texture, order=3, mode="mirror")
        return cls(coeffs=coeffs, shape=shape)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        fine_x = (x + 0.5) * SUPERSAMPLE - 0.5
        fine_y = (y + 0.5) * SUPERSAMPLE - 0.5
        coords = np.stack([fine_y.ravel(), fine_x.ravel()])
        values = ndimage.map_coordinates(self.coeffs, coords, order=3, mode="mirror", prefilter=False)
        return values.reshape(x.shape)


def render(
    texture: SpeckleTexture,
    displacement: DisplacementFn | None = None,
    contrast: tuple[float, float] = (40.0, 210.0),
    noise_std: float = 0.0,
    quantise: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Render the texture deformed by displacement(X, Y) -> (u, v) in px.

    contrast: grey levels of fully dark and fully light grains.
    """
    height, width = texture.shape
    sub = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5
    ys = (np.arange(height)[:, None] + sub[None, :]).ravel()
    xs = (np.arange(width)[:, None] + sub[None, :]).ravel()
    x, y = np.meshgrid(xs, ys)

    ref_x, ref_y = x, y
    if displacement is not None:
        ref_x, ref_y = _pull_back(x, y, displacement)

    samples = texture.evaluate(ref_x, ref_y)
    pixels = samples.reshape(height, SUPERSAMPLE, width, SUPERSAMPLE).mean(axis=(1, 3))
    dark, light = contrast
    image = dark + (light - dark) * pixels
    if noise_std > 0:
        image = image + np.random.default_rng(seed).normal(0.0, noise_std, image.shape)
    if quantise:
        image = np.clip(np.round(image), 0, GREY_LEVELS)
    return image


def _pull_back(x: np.ndarray, y: np.ndarray, displacement: DisplacementFn) -> tuple[np.ndarray, np.ndarray]:
    """Solve X + u(X) = x for the reference position X of each current sample."""
    ref_x, ref_y = x.copy(), y.copy()
    for _ in range(FIXED_POINT_ITERATIONS):
        u, v = displacement(ref_x, ref_y)
        new_x, new_y = x - u, y - v
        change = max(np.max(np.abs(new_x - ref_x)), np.max(np.abs(new_y - ref_y)))
        ref_x, ref_y = new_x, new_y
        if change < FIXED_POINT_TOLERANCE_PX:
            break
    return ref_x, ref_y
