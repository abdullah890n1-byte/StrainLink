"""Markers: found on a speckled surface (not confused with the speckle), tracked to a small fraction of a pixel."""

import numpy as np

from strainlink.markers import detect, track
from strainlink.synth.speckle import SpeckleTexture, render

SHAPE = (160, 200)
CENTRES = np.array([[50.3, 60.7], [140.2, 45.1], [100.6, 115.4]])      # (col, row) px
RADIUS = 6.0


def _photo(texture, shift):
    """Speckle with dark marker dots (soft edge), all shifted by (du, dv) px."""
    du, dv = shift
    image = render(texture, lambda X, Y: (np.full_like(X, du), np.full_like(Y, dv)), quantise=False)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(float)
    for cx, cy in CENTRES:
        cover = np.clip(RADIUS + 0.5 - np.hypot(xx - cx - du, yy - cy - dv), 0, 1)
        image = cover * 8.0 + (1 - cover) * image
    return image


def test_markers_found_and_tracked_to_a_fraction_of_a_pixel():
    texture = SpeckleTexture.granular(SHAPE, grain_radius_px=2.0, seed=3)
    shifts = [(0.0, 0.0), (0.4, -0.3), (1.1, 0.6), (2.35, 1.2)]
    photos = [_photo(texture, s) for s in shifts]
    centres, diameters, levels = detect(photos[0], (8.0, 24.0), "dark")
    assert len(centres) == 3                                       # the speckle's own dark grains are not markers
    tracks = track(photos, centres, diameters, levels, "dark", (8.0, 24.0))
    moved = np.stack([tracks.x - tracks.x[0], tracks.y - tracks.y[0]], axis=-1)
    true = np.array(shifts)[:, None, :]
    assert np.all(np.isfinite(moved))
    assert np.sqrt(np.mean((moved - true) ** 2)) < 0.1
