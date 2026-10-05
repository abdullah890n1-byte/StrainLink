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

from . import accuracy
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

    def uncertainty(self, key: str, k: int, compressed: bool = False) -> np.ndarray:
        """95% uncertainty of a quantity at every point of photo k, in the quantity's own unit.

        Measured from the field's own fine-scale scatter (see strainlink.accuracy). The first
        photo is the reference (zero by definition), so it takes the next photo's value: that
        is the noise a zero reading would carry.
        """
        k = max(int(k), 1) if self.frames > 1 else 0
        name = ("u95", key, k, compressed)
        if name in self._cache:
            return self._cache[name]
        if key in ("ux", "uz"):
            along = self.get("exx" if key == "ux" else "ezz", k)
            out = accuracy.displacement_uncertainty(along, self.get(key, k), self.spacing, STRAIN_WINDOW, compressed)
        elif key == "umag":
            out = np.maximum(self.uncertainty("ux", k, compressed), self.uncertainty("uz", k, compressed))
        else:
            out = accuracy.point_uncertainty(self.get(key, k), compressed)
        self._cache[name] = out
        return out


# Whole-object strains stated with their accuracy (the object's typical strain along each axis).
STATEMENT_STRAINS = ("exx", "ezz", "exz")


def accuracy_statement(store: FieldStore, compressed: bool = False, mm_per_px: float | None = None,
                       lens_corrected: bool = False) -> dict:
    """How accurate this analysis is: per point and for the whole object, per photo (95%).

    A plain dictionary for summary.json, the app and the report.
    """
    last = store.frames - 1

    def typical(key: str, k: int) -> float:
        values = store.uncertainty(key, k, compressed)
        return float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan")

    points = {key: typical(key, last) for key in (*STATEMENT_STRAINS, "gamma", "ux", "uz")}
    displacement_mm = max(points["ux"], points["uz"])
    whole, per_photo = {}, {key: [] for key in STATEMENT_STRAINS}
    for key in STATEMENT_STRAINS:
        for k in range(store.frames):
            field = store.get(key, k)
            known = np.isfinite(field)
            if k == 0 or known.sum() < 9:
                per_photo[key].append(None)
                continue
            value = float(np.median(field[known]))
            stated = accuracy.average_uncertainty(value, accuracy.scatter(field).typical, int(known.sum()), compressed, STRAIN_WINDOW)
            per_photo[key].append(stated if np.isfinite(stated) else None)
            if k == last:
                share = accuracy.relative(value, stated)
                whole[key] = {"value": value, "u95": stated, "relative": share, "text": accuracy.describe(value, stated)}
    peak = float(np.nanpercentile(np.abs(store.get("e1", last)), 99)) if last > 0 and np.isfinite(store.get("e1", last)).any() else float("nan")
    return {
        "confidence": "95%",
        "point": {"strain": max(points[key] for key in STATEMENT_STRAINS), **points,
                  "displacement_mm": displacement_mm,
                  "displacement_px": displacement_mm / mm_per_px if mm_per_px else None},
        "whole_object": whole, "per_photo": per_photo,
        "peak_strain": peak,
        "peak_relative": accuracy.relative(peak, max(points[key] for key in STATEMENT_STRAINS)),
        "compressed_images": compressed, "lens_corrected": lens_corrected,
        "method": ("Random part: measured from this analysis's own fine-scale scatter (2.5 x local scatter per point). "
                   "Systematic part: allowances from tests with exact answers (25 microstrain, and 0.3% of the value, "
                   "or 1.5% for JPEG or video frames). Checked on 20 known-answer cases: at least 95% of true errors "
                   "were inside the stated accuracy in every one."),
        "not_included": [n for n in accuracy.NOT_INCLUDED if not (lens_corrected and n.startswith("lens"))],
    }
