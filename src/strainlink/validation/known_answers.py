"""Compare an analysis of a synthetic test with its known answers (truth.json).

Synthetic demo folders carry the exact answers the test was generated
with. This module lines up every automatic result with its true value and
judges it against a stated tolerance, so accuracy can be checked at a
glance (the app's Accuracy tab) instead of by reading files. Tolerances
are the ones used in Validation.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TOL_FORCE_FRACTION = 0.02
TOL_KP_FRACTION = 0.02
TOL_UPLIFT_MM = 0.5
TOL_PUSH_MATCH_MM = 0.05
TOL_LIFT_OFF_DEG = 3.0
TOL_LENGTH_MM = 20.0
TOL_AREA_FRACTION = 0.05
TOL_TIP_REACH_MM = 2.0
# Truth: band-1 counts as "appeared" once its developed length reaches this (as the detector's minimum length).
ONSET_LENGTH_MM = 50.0
# Band measurements are compared every this many mm of push.
COMPARE_EVERY_MM = 5.0
# How many folders above the project to look for the known answers.
TRUTH_SEARCH_LEVELS = 2


@dataclass
class Check:
    group: str
    item: str
    found: str
    true: str
    error: str
    tolerance: str
    ok: bool
    note: str = ""


def _finite(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _fmt(v, digits: int = 2, unit: str = "") -> str:
    return "–" if not _finite(v) else f"{v:.{digits}f}{(' ' + unit) if unit else ''}"


def _polyline_length(points) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def find_truth(project_dir: Path) -> Path:
    """truth.json next to the project, or in the test folder above it.

    The app saves a new analysis in "<test folder>/StrainLink analysis/", so
    the known answers of a synthetic test folder sit one level up.
    """
    for folder in (project_dir, *list(project_dir.parents)[:TRUTH_SEARCH_LEVELS]):
        candidate = folder / "truth.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no truth.json: only synthetic tests have known answers")


def compare(project_dir: str | Path) -> dict:
    """All checks for an analysed synthetic project. Raises FileNotFoundError without truth.json."""
    project_dir = Path(project_dir)
    truth = json.loads(find_truth(project_dir).read_text())
    summary = json.loads((project_dir / "results" / "summary.json").read_text())
    checks: list[Check] = []
    m, p = summary["mechanics"], summary["bands"]["primary"]

    # --- engineering results from the sensors ---
    pp, pp_true = m["passive"]["peak"], truth["peak_force_kN"]
    checks.append(Check("Sensor results", "Peak passive force Pp", _fmt(pp, 2, "kN"), _fmt(pp_true, 2, "kN"),
                        f"{100 * (pp - pp_true) / pp_true:+.1f}%", f"±{100 * TOL_FORCE_FRACTION:.0f}%",
                        abs(pp - pp_true) <= TOL_FORCE_FRACTION * pp_true))
    kp, kp_true = m["kp"], truth["kp"]
    checks.append(Check("Sensor results", "Passive earth-pressure coefficient Kp", _fmt(kp, 2), _fmt(kp_true, 2),
                        f"{100 * (kp - kp_true) / kp_true:+.1f}%", f"±{100 * TOL_KP_FRACTION:.0f}%",
                        abs(kp - kp_true) <= TOL_KP_FRACTION * kp_true))
    lo, hi = m["passive"]["push_range_mm"]
    dp_true = truth["delta_p_mm"]
    checks.append(Check("Sensor results", "Push at peak δp", f"{_fmt(m['passive']['push_at_peak_mm'], 1, 'mm')} (range {lo:.1f}–{hi:.1f})",
                        _fmt(dp_true, 1, "mm"), "inside range" if lo <= dp_true <= hi else "outside range",
                        "true value inside the reported range", lo - 1e-6 <= dp_true <= hi + 1e-6,
                        "plateau: the force is within noise of its peak over the whole range" if m["passive"].get("plateau") else ""))
    up, up_true = m["uplift_onset_mm"], truth["uplift_start_mm"]
    if _finite(up) and _finite(up_true):
        checks.append(Check("Sensor results", "Wall uplift starts at", _fmt(up, 2, "mm"), _fmt(up_true, 2, "mm"),
                            f"{up - up_true:+.2f} mm", f"±{TOL_UPLIFT_MM} mm", abs(up - up_true) <= TOL_UPLIFT_MM))

    # --- photo timing ---
    found_push = np.asarray(summary["frame_push_mm"], dtype=float)
    true_push = np.asarray(truth["frame_push_mm"], dtype=float)
    worst = float(np.max(np.abs(found_push - true_push)))
    checks.append(Check("Photo timing", "Photos matched to the logger (worst photo)", f"±{worst:.3f} mm of push",
                        "exact", f"{worst:.3f} mm", f"{TOL_PUSH_MATCH_MM} mm", worst <= TOL_PUSH_MATCH_MM,
                        f"stated uncertainty ±{summary['sync'].get('push_sigma_mm', float('nan')):.3f} mm"))

    # --- slip surface ---
    step = float(np.median(np.diff(true_push)))
    primary_len = [max([_polyline_length(b["points"]) for b in frame if b["label"] == "primary"] or [0.0])
                   for frame in truth["bands"]]
    onset_true = next((d for d, L in zip(true_push, primary_len) if L >= ONSET_LENGTH_MM), None)
    if _finite(p.get("onset_mm")) and onset_true is not None:
        err = p["onset_mm"] - onset_true
        checks.append(Check("Slip surface (band-1)", f"Appears (≥ {ONSET_LENGTH_MM:.0f} mm long) at", _fmt(p["onset_mm"], 1, "mm"),
                            _fmt(onset_true, 1, "mm"), f"{err:+.1f} mm", f"±1 photo ({step:.1f} mm)", abs(err) <= step + 0.01))
    surface_true = next((d for d, L in zip(true_push, truth["length_mm"]) if _finite(L)), None)
    if _finite(p.get("surface_reached_mm")) and surface_true is not None:
        err = p["surface_reached_mm"] - surface_true
        checks.append(Check("Slip surface (band-1)", "Reaches the soil surface at", _fmt(p["surface_reached_mm"], 1, "mm"),
                            _fmt(surface_true, 1, "mm"), f"{err:+.1f} mm", f"±1 photo ({step:.1f} mm)", abs(err) <= step + 0.01))

    found_by_push = {round(d, 3): i for i, d in enumerate(p["delta_mm"])}
    for k, d in enumerate(true_push):
        if d < COMPARE_EVERY_MM - 1e-6 or abs(d / COMPARE_EVERY_MM - round(d / COMPARE_EVERY_MM)) > 1e-6:
            continue
        i = k if k < len(p["delta_mm"]) else found_by_push.get(round(d, 3))
        if i is None or not p["detected"][i]:
            continue
        label = f"at {d:.0f} mm push"
        lift, lift_true = p["lift_off_deg"][i], truth["lift_off_span_deg"][k]
        if _finite(lift) and _finite(lift_true):
            checks.append(Check("Slip surface (band-1)", f"Lift-off angle {label}", f"{lift:.1f}° ± {p['lift_off_sigma_deg'][i]:.1f}°",
                                f"{lift_true:.1f}°", f"{lift - lift_true:+.1f}°", f"±{TOL_LIFT_OFF_DEG:.0f}°",
                                abs(lift - lift_true) <= TOL_LIFT_OFF_DEG))
        length, length_true = p["length_mm"][i], truth["length_mm"][k]
        if _finite(length) and _finite(length_true):
            checks.append(Check("Slip surface (band-1)", f"Impacted length {label}", _fmt(length, 0, "mm"), _fmt(length_true, 0, "mm"),
                                f"{length - length_true:+.0f} mm", f"±{TOL_LENGTH_MM:.0f} mm", abs(length - length_true) <= TOL_LENGTH_MM))
        area, area_true = p["area_mm2"][i], truth["area_mm2"][k]
        if _finite(area) and _finite(area_true):
            checks.append(Check("Slip surface (band-1)", f"Wedge area {label}", f"{area / 1e4:.2f} ×10⁴ mm²", f"{area_true / 1e4:.2f} ×10⁴ mm²",
                                f"{100 * (area - area_true) / area_true:+.1f}%", f"±{100 * TOL_AREA_FRACTION:.0f}%",
                                abs(area - area_true) <= TOL_AREA_FRACTION * area_true))
        shape, shape_true = p["shape"][i], truth["shape"][k]
        if shape:
            ambiguous = shape.endswith("?")
            ok = shape.rstrip("?") == shape_true
            checks.append(Check("Slip surface (band-1)", f"Shape {label}", shape.rstrip("?") + (" (flagged ambiguous)" if ambiguous else ""),
                                shape_true, "same" if ok else "different", "same label", ok or ambiguous,
                                "different, but flagged for your review" if (ambiguous and not ok) else ""))

    # --- correlation engine ---
    events = {e["key"]: e for e in summary["correlation"]["events"]}
    for cell, true_reach in truth.get("pt_band_tip_pass_mm", {}).items():
        event = events.get(f"band-1.reaches.{cell}")
        if event is None or not _finite(true_reach):
            continue
        err = event["push_mm"] - true_reach
        checks.append(Check("Correlation engine", f"Band-1 tip reaches the level of {cell} at", _fmt(event["push_mm"], 1, "mm"),
                            _fmt(true_reach, 1, "mm"), f"{err:+.1f} mm", f"±{TOL_TIP_REACH_MM:.0f} mm", abs(err) <= TOL_TIP_REACH_MM))
    finding = next((f for f in summary["correlation"]["findings"] if f["key"] == "cells-vs-tip"), None)
    if finding is not None:
        checks.append(Check("Correlation engine", "Finds the built-in link: cells soften when the band tip reaches them",
                            finding["strength"], "present (built into the test)", "", "graded strong or moderate",
                            finding["strength"] in ("strong", "moderate"), finding["statement"]))

    # Plain Python types only: the result is sent to the app as JSON (numpy bools are not serialisable).
    rows = [{**asdict(c), "ok": bool(c.ok)} for c in checks]
    return {"preset": truth.get("preset"), "checks": rows, "passed": int(sum(r["ok"] for r in rows)), "total": len(rows)}
