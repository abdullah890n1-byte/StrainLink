"""Export every field of an analysis for other software (as Vic-2D exports its data).

CSV: one file per photo, one row per measurement point, a column per
quantity. MAT (MATLAB / Octave / Python): one file holding every quantity
as a (photos, rows, columns) array, plus the grid and the test axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fields import QUANTITIES, FieldStore


def export(results_dir: str | Path, fmt: str = "csv") -> Path:
    """Write results/export/ (CSV files) or results/export/fields.mat; return the path."""
    results_dir = Path(results_dir)
    store = FieldStore(results_dir)
    out = results_dir / "export"
    out.mkdir(exist_ok=True)
    keys = list(QUANTITIES)
    axis = store.data.get("push_mm", np.arange(store.frames, dtype=float))
    if fmt == "mat":
        from scipy.io import savemat
        arrays = {k: np.stack([store.get(k, j) for j in range(store.frames)]) for k in keys}
        savemat(out / "fields.mat", {"x_mm": store.x, "z_mm": store.z, "axis": axis, **arrays,
                                     "quantities": np.array([f"{k}: {QUANTITIES[k].label}" for k in keys], dtype=object)},
                do_compression=True)
        return out / "fields.mat"
    for j in range(store.frames):
        frame = {"x_mm": store.x.ravel(), "z_mm": store.z.ravel()}
        for k in keys:
            frame[k] = store.get(k, j).ravel()
        table = pd.DataFrame(frame)
        table = table[np.isfinite(table["ux"])]
        table.to_csv(out / f"photo_{j + 1:04d}.csv", index=False, float_format="%.6g")
    (out / "README.txt").write_text(
        "StrainLink field export: one CSV per photo; x_mm, z_mm are positions in the first photo (z up).\n"
        "Axis value of each photo (wall push, displacement or time): " + json.dumps(np.round(axis, 6).tolist()) + "\n\n"
        + "\n".join(f"{k}: {q.label}" + (f" ({q.unit})" if q.unit else "") for k, q in QUANTITIES.items()) + "\n")
    return out
