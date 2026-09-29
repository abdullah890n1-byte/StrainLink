"""What kind of localisation a zone is: a crack opening or a shear band sliding.

Across a localisation zone the two sides of the object move relative to
each other. The jump vector, split into its component along the zone's
normal (opening) and along the zone (slip), tells the mechanism for any
material: cracks open, shear bands slide. The jump is measured 1.5 and 3
zone widths either side of the centreline and extrapolated to zero
distance, which removes the object's smooth background strain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..bands.detect import DetectedBand
from ..bands.ridges import fill_missing

# A zone is "opening" or "shear" when that component is at least this many
# times the other; otherwise "mixed".
DOMINANCE = 2.0
# Centreline points sampled per zone (evenly spaced).
SAMPLES = 25
# The jump is sampled this many zone widths (FWHM) either side of the centreline:
# far enough that the zone's own tails have settled (at one width, 7% of the
# jump was still missing), close enough not to reach other zones or edges.
NEAR_WIDTHS = 1.5
FAR_WIDTHS = 3.0


@dataclass
class ZoneKinematics:
    mode: str            # "opening" (crack), "shear" (shear band) or "mixed"
    opening_mm: float    # median jump normal to the zone (positive = opening)
    slip_mm: float       # median jump along the zone (absolute)
    angle_deg: float     # zone direction, degrees from horizontal (0-180)


def kinematics(band: DetectedBand, x: np.ndarray, z: np.ndarray, ux: np.ndarray, uz: np.ndarray) -> ZoneKinematics:
    """Opening, slip and mode of one detected zone from the displacement field (mm, z up)."""
    pts = band.points
    chord = pts[-1] - pts[0]
    angle = float(np.degrees(np.arctan2(chord[1], chord[0])) % 180.0)
    idx = np.linspace(0, len(pts) - 1, min(SAMPLES, len(pts))).astype(int)
    tangent = np.gradient(ndimage.gaussian_filter1d(pts, 2.0, axis=0, mode="nearest"), axis=0)[idx]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    ux_f, uz_f = fill_missing(ux), fill_missing(uz)
    x0, dx, z0, dz = x[0, 0], x[0, 1] - x[0, 0], z[0, 0], z[1, 0] - z[0, 0]

    def sample(points: np.ndarray) -> np.ndarray:
        coords = np.stack([(points[:, 1] - z0) / dz, (points[:, 0] - x0) / dx])
        return np.column_stack([ndimage.map_coordinates(ux_f, coords, order=1, mode="nearest"),
                                ndimage.map_coordinates(uz_f, coords, order=1, mode="nearest")])

    width = max(band.width_mm, 2.0 * abs(dx))
    centre = pts[idx]
    near, far = NEAR_WIDTHS * width, FAR_WIDTHS * width
    jump_near = sample(centre + near * normal) - sample(centre - near * normal)
    jump_far = sample(centre + far * normal) - sample(centre - far * normal)
    # Background strain adds in proportion to the sampling distance; extrapolate it out.
    jump = (far * jump_near - near * jump_far) / (far - near)
    opening = float(np.median(np.sum(jump * normal, axis=1)))
    slip = float(np.median(np.abs(np.sum(jump * tangent, axis=1))))
    if abs(opening) >= DOMINANCE * slip:
        mode = "opening"
    elif slip >= DOMINANCE * abs(opening):
        mode = "shear"
    else:
        mode = "mixed"
    return ZoneKinematics(mode, opening, slip, angle)
