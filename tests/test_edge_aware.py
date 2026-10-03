"""Edge-aware subsets and strain next to edges (SYNTHETIC)."""

import numpy as np
import pytest

from strainlink.dic import DICSettings, Grid, SplineImage, correlate_pair, track_sequence
from strainlink.dic.mask import SubsetMask
from strainlink.strain import compute_strain
from strainlink.synth.speckle import SpeckleTexture, render

SHAPE = (160, 200)
EDGE_COL = 120            # the object is left of this column; plain still background to the right
SETTINGS = DICSettings(subset_radius=12, step=6, search_radius=6, reference_mode="fixed")
SHIFT = (2.3, -1.4)


@pytest.fixture(scope="module")
def scene():
    texture = SpeckleTexture.granular(SHAPE, grain_radius_px=1.8, seed=4)
    ref = render(texture, None, noise_std=0.5, seed=1)
    cur = render(texture, lambda X, Y: (np.full_like(X, SHIFT[0]), np.full_like(Y, SHIFT[1])), noise_std=0.5, seed=2)
    cols = np.arange(SHAPE[1])[None, :]
    on_object_ref = np.broadcast_to(cols < EDGE_COL, SHAPE)
    on_object_cur = np.broadcast_to(cols < EDGE_COL + SHIFT[0], SHAPE)
    ref = np.where(on_object_ref, ref, 30.0)
    cur = np.where(on_object_cur, cur, 30.0)
    return ref, cur, on_object_ref.copy()


def _pair(scene, xs, ys, masked):
    ref, cur, mask = scene
    sub = SubsetMask(mask, xs, ys, SETTINGS.subset_radius) if masked else None
    return correlate_pair(SplineImage.from_array(ref), SplineImage.from_array(cur, with_gradients=False),
                          xs, ys, SETTINGS, subset_mask=sub)


def test_edge_subsets_follow_the_object_not_the_background(scene):
    ys = np.arange(30.0, 130.0, 6.0)
    xs = np.full_like(ys, EDGE_COL - 4.0)          # the subset is two-thirds background
    plain = _pair(scene, xs, ys, masked=False)
    aware = _pair(scene, xs, ys, masked=True)
    assert aware.valid.all()
    assert np.max(np.abs(aware.u - SHIFT[0])) < 0.03 and np.max(np.abs(aware.v - SHIFT[1])) < 0.03
    # Without the mask the still background pulls the subset (or it is rejected).
    plain_err = np.where(plain.valid, np.abs(plain.u - SHIFT[0]), np.inf)
    assert np.median(plain_err) > 0.2


def test_interior_subsets_unchanged_by_the_mask(scene):
    gy, gx = np.mgrid[30:130:10, 30:90:10].astype(float)
    plain = _pair(scene, gx.ravel(), gy.ravel(), masked=False)
    aware = _pair(scene, gx.ravel(), gy.ravel(), masked=True)
    assert np.array_equal(plain.valid, aware.valid)
    assert np.max(np.abs(plain.u - aware.u)) < 1e-9 and np.max(np.abs(plain.v - aware.v)) < 1e-9


def test_mask_follows_the_material_through_incremental_references():
    texture = SpeckleTexture.granular(SHAPE, grain_radius_px=1.8, seed=5)
    stretch = 0.04
    images, cols = [], np.arange(SHAPE[1])[None, :]
    for k in range(4):
        e = stretch * k / 3
        img = render(texture, lambda X, Y: (e * X, np.zeros_like(Y)), noise_std=0.5, seed=k)
        images.append(np.where(np.broadcast_to(cols < EDGE_COL * (1 + e), SHAPE), img, 30.0))
    mask = np.broadcast_to(cols < EDGE_COL, SHAPE).copy()
    gy, gx = np.mgrid[30:130:6, 20:EDGE_COL:6].astype(float)
    settings = DICSettings(subset_radius=12, step=6, search_radius=8, reference_mode="incremental", update_every=1)
    tracked = track_sequence(images, Grid(x=gx, y=gy), settings, object_mask=mask)
    u = tracked.u[-1].reshape(gx.shape)
    edge = gx >= EDGE_COL - 6
    assert tracked.valid[-1].reshape(gx.shape)[edge].all()
    assert np.max(np.abs(u[edge] - stretch * gx[edge])) < 0.05


def test_edge_aware_strain_unbiased_at_an_edge():
    y, x = np.mgrid[0:30, 0:30].astype(float)
    u = 0.001 * x**2 + 0.0005 * x * y
    v = 0.0003 * y**2
    u[:, :10] = np.nan
    v[:, :10] = np.nan
    plane = compute_strain(u, v, 1.0, 1.0, 5, order=1)
    quad = compute_strain(u, v, 1.0, 1.0, 5, order=2)
    true_edge = 2 * 0.001 * 10 + 0.0005 * 15
    assert abs(plane.dudx[15, 10] - true_edge) > 1e-3          # the plane fit is biased inwards
    assert quad.dudx[15, 10] == pytest.approx(true_edge, abs=1e-9)
    assert quad.dudx[15, 20] == pytest.approx(plane.dudx[15, 20], abs=1e-12)   # full windows unchanged
