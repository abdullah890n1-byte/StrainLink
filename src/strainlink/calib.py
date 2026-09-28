"""Image-to-wall coordinate calibration.

The single place where pixels become millimetres. Wall frame: origin at
the wall toe, x positive into the backfill, z positive upward (mm).
Image frame: column to the right, row downward (px).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WallFrame:
    mm_per_px: float
    toe_col: float        # image column of the wall toe (px)
    toe_row: float        # image row of the wall toe (px)
    wall_height_mm: float  # effective retained height H (soil surface at z = H)

    def to_wall(self, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(col) - self.toe_col) * self.mm_per_px
        z = (self.toe_row - np.asarray(row)) * self.mm_per_px
        return x, z

    def to_image(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = self.toe_col + np.asarray(x) / self.mm_per_px
        row = self.toe_row - np.asarray(z) / self.mm_per_px
        return col, row

    def displacement_to_wall(self, u_px: np.ndarray, v_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Image displacement (px, v downward) -> wall displacement (mm, z upward)."""
        return np.asarray(u_px) * self.mm_per_px, -np.asarray(v_px) * self.mm_per_px
