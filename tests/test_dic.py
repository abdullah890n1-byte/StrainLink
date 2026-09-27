"""DIC engine accuracy against synthetic images with known displacement (SYNTHETIC)."""

import numpy as np
import pytest

from strainlink.dic import DICSettings, Grid, SplineImage, correlate_pair, track_sequence
from strainlink.synth.speckle import SpeckleTexture, render

SHAPE = (160, 160)
SETTINGS = DICSettings(subset_radius=12, step=10, search_radius=6, reference_mode="fixed")


@pytest.fixture(scope="module")
def texture():
    return SpeckleTexture.granular(SHAPE, grain_radius_px=1.8, seed=1)


def _grid():
    return Grid.over_roi((0, 0, SHAPE[1], SHAPE[0]), step=SETTINGS.step, margin=30)


def _correlate(texture, displacement, noise_std=0.0):
    ref = render(texture, None, noise_std=noise_std, seed=1)
    cur = render(texture, displacement, noise_std=noise_std, seed=2)
    grid = _grid()
    pair = correlate_pair(
        SplineImage.from_array(ref), SplineImage.from_array(cur, with_gradients=False),
        grid.x.ravel(), grid.y.ravel(), SETTINGS,
    )
    return grid, pair


@pytest.mark.parametrize("shift", [(0.3, -0.7), (2.45, 1.1), (-4.8, 3.25)])
def test_rigid_subpixel_shift(texture, shift):
    grid, pair = _correlate(texture, lambda X, Y: (np.full_like(X, shift[0]), np.full_like(Y, shift[1])))
    assert pair.valid.all()
    assert np.max(np.abs(pair.u - shift[0])) < 0.02
    assert np.max(np.abs(pair.v - shift[1])) < 0.02


def test_affine_strain(texture):
    exx, eyy = 0.005, -0.003
    cx, cy = SHAPE[1] / 2, SHAPE[0] / 2
    grid, pair = _correlate(texture, lambda X, Y: (exx * (X - cx), eyy * (Y - cy)))
    true_u = exx * (grid.x.ravel() - cx)
    true_v = eyy * (grid.y.ravel() - cy)
    assert pair.valid.all()
    assert np.max(np.abs(pair.u - true_u)) < 0.02
    assert np.max(np.abs(pair.v - true_v)) < 0.02
    assert np.median(pair.params[:, 1]) == pytest.approx(exx, abs=5e-4)
    assert np.median(pair.params[:, 5]) == pytest.approx(eyy, abs=5e-4)


def test_noise_is_tolerated(texture):
    grid, pair = _correlate(texture, lambda X, Y: (np.full_like(X, 0.5), np.full_like(Y, 0.5)), noise_std=4.0)
    assert pair.valid.mean() > 0.95
    assert np.std(pair.u[pair.valid] - 0.5) < 0.05


def test_incremental_tracking_large_motion(texture):
    """Accumulated 12 px motion in 1.5 px steps: beyond a single search window."""
    steps = np.arange(0, 9) * 1.5
    images = [render(texture, (lambda s: lambda X, Y: (np.full_like(X, s), 0.2 * np.full_like(Y, s)))(s))
              for s in steps]
    grid = Grid.over_roi((0, 0, SHAPE[1], SHAPE[0]), step=10, margin=40)
    settings = DICSettings(subset_radius=12, step=10, search_radius=4, reference_mode="incremental",
                           update_every=2)
    result = track_sequence(images, grid, settings)
    final_u = result.u[-1][result.valid[-1]]
    final_v = result.v[-1][result.valid[-1]]
    assert result.valid[-1].mean() > 0.9
    assert np.max(np.abs(final_u - steps[-1])) < 0.05
    assert np.max(np.abs(final_v - 0.2 * steps[-1])) < 0.05
    assert len(set(result.reference_frames)) > 1
