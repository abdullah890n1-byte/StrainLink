"""Run the failure-zone model on a finished analysis (its results/fields.npz)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import ZoneModel, load, zones


def analysis_fields(results_dir: Path) -> tuple[dict, float]:
    """The saved fields of an analysis, and its scale (mm per pixel)."""
    results_dir = Path(results_dir)
    with np.load(results_dir / "fields.npz") as npz:
        data = {name: npz[name] for name in npz.files}
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    mm_per_px = summary.get("frame", {}).get("mm_per_px")
    if not mm_per_px:
        project = json.loads((results_dir.parent / "project.json").read_text(encoding="utf-8"))
        mm_per_px = project["calibration"]["mm_per_px"]
    return data, float(mm_per_px)


def zone_probability(results_dir: Path, model: ZoneModel | None = None, photos: list[int] | None = None) -> dict[int, np.ndarray]:
    """Zone probability grid for each photo of an analysis (all photos after the first by default)."""
    model = model or load()
    data, mm_per_px = analysis_fields(results_dir)
    step_px = float(abs(data["x_mm"][0, 1] - data["x_mm"][0, 0])) / mm_per_px
    count = len(data["ux"])
    out = {}
    for k in photos if photos is not None else range(1, count):
        u_px, v_px = data["ux"][k] / mm_per_px, -data["uz"][k] / mm_per_px       # image axes: y down
        valid = data["valid"][k] if "valid" in data else np.isfinite(u_px)
        out[k] = model.probability(u_px, v_px, valid, step_px)
    return out


def first_detection(probabilities: dict[int, np.ndarray], threshold: float) -> tuple[int | None, dict[int, list[dict]]]:
    """The first photo in which the model reports a zone, and the zones found in every photo."""
    found = {k: zones(p, threshold)[1] for k, p in probabilities.items()}
    first = next((k for k in sorted(found) if found[k]), None)
    return first, found


RESULT_FIELDS = "ml_zones.npz"
RESULT_SUMMARY = "ml_zones.json"
NOTE = ("Experimental AI model (machine learning), an add-on to StrainLink's own zone detection. It was trained on "
        "SYNTHETIC scenes measured with StrainLink's DIC; on real tests its zones are a second opinion, not a measurement.")


def run_on_analysis(project_path: str | Path, model: ZoneModel | None = None) -> dict:
    """Run the model on every photo of a finished analysis; write results/ml_zones.npz and .json.

    The analysis itself is not changed: these two files sit beside its results.
    """
    results_dir = Path(project_path).parent / "results"
    model = model or load()
    probs = zone_probability(results_dir, model)
    data, _ = analysis_fields(results_dir)
    stack = np.full(data["ux"].shape, np.nan, dtype=np.float32)
    for k, p in probs.items():
        stack[k] = p
    first, found = first_detection(probs, model.threshold)
    summary_path = results_dir / "summary.json"
    axis = json.loads(summary_path.read_text(encoding="utf-8")).get("frame_push_mm") if summary_path.exists() else None
    last = max(probs) if probs else None
    result = {
        "model": model.info.get("model"), "trained_on": model.info.get("data"), "threshold": model.threshold,
        "first_photo_with_zone": None if first is None else first + 1,
        "first_zone_axis": None if first is None or not axis else axis[first],
        "zones_last_photo": found.get(last, []) if last is not None else [],
        "photos_with_zone": [k + 1 for k in sorted(found) if found[k]],
        "note": NOTE,
    }
    np.savez_compressed(results_dir / RESULT_FIELDS, zone_probability=stack)
    (results_dir / RESULT_SUMMARY).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
