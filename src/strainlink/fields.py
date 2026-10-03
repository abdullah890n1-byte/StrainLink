"""Every measured quantity, by name: for the viewer overlay, the inspection tools and the export.

An analysis stores its displacement (and, depending on the mode, some
strain) per photo in results/fields.npz. Anything not stored is computed
from the displacement with the same strain window, so retaining-wall and
any-object analyses offer the same quantities.

Coordinates: mm, x to the right, z up (the wall frame or the photo frame).
Signs follow that frame (exz, rotation counter-clockwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bands.ridges import fill_missing
from .strain import compute_strain, measures

STRAIN_WINDOW = 5


@dataclass(frozen=True)
class Quantity:
    key: str
    label: str
    unit: str
    signed: bool          # diverging colours (± around zero) or a magnitude
    percent: bool = False # shown as % in the interface


QUANTITIES = {q.key: q for q in [
    Quantity("gamma", "Max shear strain γmax", "", False),
    Quantity("exx", "Strain εxx (Green–Lagrange)", "", True, True),
    Quantity("ezz", "Strain εzz (Green–Lagrange)", "", True, True),
    Quantity("exz", "Shear strain εxz (Green–Lagrange)", "", True, True),
    Quantity("e1", "Major principal strain ε1 (Hencky)", "", True, True),
    Quantity("e2", "Minor principal strain ε2 (Hencky)", "", True, True),
    Quantity("von_mises", "Von Mises equivalent strain", "", False, True),
    Quantity("exx_eng", "Engineering strain along x", "", True, True),
    Quantity("ezz_eng", "Engineering strain along z", "", True, True),
    Quantity("shear_eng", "Engineering shear angle", "rad", True),
    Quantity("almansi_xx", "Euler–Almansi εxx", "", True, True),
    Quantity("almansi_zz", "Euler–Almansi εzz", "", True, True),
    Quantity("almansi_xz", "Euler–Almansi εxz", "", True, True),
    Quantity("rotation_deg", "Rotation", "°", True),
    Quantity("ux", "Displacement u (x)", "mm", True),
    Quantity("uz", "Displacement w (z, up)", "mm", True),
    Quantity("umag", "Displacement magnitude", "mm", False),
]}
# Measures computed from the gradients (strain.measures uses y for the second axis; here it is z).
_RENAME = {"eyy_eng": "ezz_eng", "almansi_yy": "almansi_zz", "almansi_xy": "almansi_xz"}


class FieldStore:
    """The fields of one analysis, loaded once; derived quantities computed per photo on demand."""

    def __init__(self, results_dir: str | Path):
        with np.load(Path(results_dir) / "fields.npz") as npz:
            self.data = {name: npz[name] for name in npz.files}
        self.x, self.z = self.data["x_mm"], self.data["z_mm"]
        self.spacing = float(abs(self.x[0, 1] - self.x[0, 0]))
        self.frames = len(self.data["ux"])
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def get(self, key: str, k: int) -> np.ndarray:
        if key not in QUANTITIES:
            raise KeyError(f"unknown quantity {key}")
        if key in self.data and key not in ("x_mm", "z_mm"):
            return self.data[key][k]
        if (key, k) in self._cache:
            return self._cache[(key, k)]
        ux, uz = self.data["ux"][k], self.data["uz"][k]
        if key == "umag":
            out = np.hypot(ux, uz)
        else:
            measured = np.isfinite(ux)
            if not measured.any() or k == 0:
                out = np.where(measured, 0.0, np.nan)
            else:
                strain = compute_strain(fill_missing(ux), fill_missing(uz), self.spacing, -self.spacing, STRAIN_WINDOW)
                derived = {"gamma": strain.gamma_max, "exx": strain.exx, "ezz": strain.eyy, "exz": strain.exy,
                           "e1": strain.e1, "e2": strain.e2,
                           **{_RENAME.get(n, n): v for n, v in measures(strain).items()}}
                for name, values in derived.items():
                    self._cache[(name, k)] = np.where(measured, values, np.nan)
                out = self._cache[(key, k)]
        self._cache[(key, k)] = out
        return out
