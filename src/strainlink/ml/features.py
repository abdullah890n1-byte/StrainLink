"""What the failure-zone model sees at each measurement point.

The features are computed from DIC displacement on its grid, the same way
for training scenes and for real analyses, and they are made independent
of the things that change from test to test but say nothing about failure:

- the scale (mm per pixel) and the grid spacing: everything is in pixels or
  in grid steps;
- the image quality: strain values are divided by the measurement noise of
  that very field, so "3" means three times the noise whatever the camera
  (the noise level itself is deliberately not a feature: real tests can be
  far noisier than any training scene, and a model must not key on it);
- direction: only quantities that do not change when the specimen is
  rotated are used (no x or y components, no position in the photo).

The features describe how strong the strain is compared with the noise and
with its surroundings (at several scales), whether the strain forms a line
(a ridge) rather than a blob, whether the movement has a step in it (a
jump that a smooth deformation cannot make), and whether DIC lost points
nearby (it does inside active zones).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import accuracy
from ..strain import compute_strain

# Smoothing scales (grid steps) and window sizes (grid points) the features look at.
SMOOTHING_SCALES = (1.0, 2.0, 4.0)
RIDGE_SCALES = (1.5, 3.0)
PLANE_WINDOWS = (5, 9)
CONTEXT_WINDOW = 15
LOST_WINDOW = 5
EDGE_CAP_STEPS = 10.0
STRAIN_WINDOW = 5
# Used when a field's noise cannot be measured (too few points): a typical good-image value.
NOISE_FLOOR = 1e-4

NAMES = (
    [f"gamma_over_noise_s{s:g}" for s in (0.0, *SMOOTHING_SCALES)]
    + ["gamma_over_local_median", "gamma_minus_local_median_over_noise", "volumetric_over_noise", "e1_over_noise", "e2_over_noise"]
    + [f"ridge_strength_s{s:g}" for s in RIDGE_SCALES] + [f"ridge_anisotropy_s{s:g}" for s in RIDGE_SCALES]
    + [f"step_in_movement_w{w}" for w in PLANE_WINDOWS]
    + ["lost_points_nearby", "distance_to_edge"]
)


def _nan_gaussian(field: np.ndarray, sigma: float) -> np.ndarray:
    known = np.isfinite(field)
    if sigma <= 0:
        return field
    num = ndimage.gaussian_filter(np.where(known, field, 0.0), sigma, mode="constant")
    den = ndimage.gaussian_filter(known.astype(float), sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(known, num / den, np.nan)


def _ridge(field: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """How line-like the field is: strength of a crest (−smallest Hessian eigenvalue) and its anisotropy."""
    smooth = np.nan_to_num(_nan_gaussian(field, sigma), nan=0.0)
    gyy = ndimage.gaussian_filter(smooth, sigma, order=(2, 0))
    gxx = ndimage.gaussian_filter(smooth, sigma, order=(0, 2))
    gxy = ndimage.gaussian_filter(smooth, sigma, order=(1, 1))
    mean, radius = 0.5 * (gxx + gyy), np.sqrt(0.25 * (gxx - gyy) ** 2 + gxy**2)
    low, high = mean - radius, mean + radius
    strength = np.maximum(-low, 0.0) * sigma**2          # scale-normalised
    anisotropy = (high - low) / (np.abs(low) + np.abs(high) + 1e-12)
    return strength, anisotropy


def _plane_residual(field: np.ndarray, window: int) -> np.ndarray:
    """RMS misfit of the best plane through each window: zero for any smooth stretch, large across a step."""
    known = np.isfinite(field).astype(float)
    f = np.where(known > 0, field, 0.0)
    half = window // 2
    oy, ox = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    terms = [np.ones_like(ox), ox, oy]

    def local(data, kernel):
        return ndimage.correlate(data, kernel, mode="constant", cval=0.0)

    n = local(known, terms[0])
    normal = np.stack([np.stack([local(known, a * b) for b in terms], -1) for a in terms], -2)
    rhs = np.stack([local(f, t) for t in terms], -1)
    ff = local(f * f, terms[0])
    usable = (n >= 0.6 * window * window) & (known > 0)
    out = np.full(field.shape, np.nan)
    if usable.any():
        coef = np.linalg.solve(normal[usable], rhs[usable][..., None])[..., 0]
        sse = ff[usable] - np.einsum("ij,ij->i", coef, rhs[usable])
        out[usable] = np.sqrt(np.maximum(sse, 0.0) / n[usable])
    return out


def strain_noise(strain) -> float:
    """Typical fine-scale scatter of a field's normal strains: the yardstick every feature is measured against."""
    noise = np.nanmean([accuracy.scatter(strain.exx).typical, accuracy.scatter(strain.eyy).typical])
    return float(noise) if np.isfinite(noise) and noise > 0 else NOISE_FLOOR


def noise_of(u_px: np.ndarray, v_px: np.ndarray, step_px: float) -> float:
    return strain_noise(compute_strain(u_px, v_px, step_px, step_px, STRAIN_WINDOW, order=2))


def compute(u_px: np.ndarray, v_px: np.ndarray, valid: np.ndarray, step_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Features (points, len(NAMES)) for every grid point with a measurement, and those points' mask.

    u_px, v_px: displacement in pixels on the DIC grid (NaN where not measured); valid: DIC tracked
    the point in this photo; step_px: grid spacing in pixels.
    """
    measured = np.isfinite(u_px) & np.isfinite(v_px)
    strain = compute_strain(u_px, v_px, step_px, step_px, STRAIN_WINDOW, order=2)
    gamma, e1, e2 = strain.gamma_max, strain.e1, strain.e2
    noise = strain_noise(strain)
    have = measured & np.isfinite(gamma)
    columns = [gamma / noise] + [_nan_gaussian(gamma, s) / noise for s in SMOOTHING_SCALES]
    local_median = ndimage.median_filter(np.where(np.isfinite(gamma), gamma, np.nanmedian(gamma)), size=CONTEXT_WINDOW,
                                         mode="nearest")
    columns += [gamma / (local_median + noise), (gamma - local_median) / noise, strain.volumetric / noise, e1 / noise, e2 / noise]
    ridges = [_ridge(gamma / noise, s) for s in RIDGE_SCALES]
    columns += [r[0] for r in ridges] + [r[1] for r in ridges]
    # A step in the movement, compared with how much a smooth field's movement scatters.
    disp_noise = noise * step_px * accuracy.window_factor(STRAIN_WINDOW)
    for window in PLANE_WINDOWS:
        columns.append(np.hypot(_plane_residual(u_px, window), _plane_residual(v_px, window)) / disp_noise)
    lost = 1.0 - ndimage.uniform_filter(np.where(measured, valid.astype(float), 0.0), LOST_WINDOW, mode="constant") / \
        np.maximum(ndimage.uniform_filter(measured.astype(float), LOST_WINDOW, mode="constant"), 1e-9)
    edge = np.minimum(ndimage.distance_transform_edt(np.pad(measured, 1))[1:-1, 1:-1], EDGE_CAP_STEPS)
    columns += [lost, edge]
    stack = np.stack([np.where(np.isfinite(c), c, 0.0) for c in columns], axis=-1)
    return stack[have].astype(np.float32), have
