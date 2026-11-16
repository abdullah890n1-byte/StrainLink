"""How well does the failure-zone model work, compared with StrainLink's rule-based detector?

    .venv/bin/python scripts/validate_ml.py

A. Unseen test scenes (SYNTHETIC, same kind as training, never seen by the model):
   per point (precision, recall, overlap with the truth) and per zone (found or
   not, by the size of its jump), for the model and for the rule-based detector
   on exactly the same measured fields; false alarms in scenes with no zone.
B. Independent synthetic tests (made by other generators, full StrainLink runs,
   bigger photos, many photos): the cracking and shearing plates (when is the
   zone first found, any false alarm before it forms?) and every no-failure
   case of the Vic-2D task suite (any false alarm at all?).
C. Real images (no exact answer): the split soil disc (the published crack
   appears at 67 s) and the sand photos.

Output: reports/ml_validation.md, reports/ml_validation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from strainlink.bands import DetectionSettings, detect_bands
from strainlink.general.pipeline import MAX_ZONE_WIDTH_FRACTION, MIN_ZONE_LENGTH_FRACTION, _zone_mask
from strainlink.ml.apply import analysis_fields, first_detection, zone_probability
from strainlink.ml.model import load
from strainlink.strain import compute_strain

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "ml" / "test"
SYNTH = ROOT / "data" / "synthetic"
REAL = ROOT / "data" / "external"
JUMP_BINS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
# A zone counts as found when at least this share of its corridor points is flagged.
FOUND_SHARE = 0.5


def rule_based_mask(u, v, step_px):
    """StrainLink's rule-based zone detector on one field, as the any-object pipeline runs it."""
    strain = compute_strain(u, v, step_px, step_px, 5, order=2)
    rows, cols = u.shape
    X, Y = np.meshgrid(np.arange(cols) * step_px, np.arange(rows) * step_px)
    x, z = X, (rows - 1) * step_px - Y                         # z up, as in the pipeline
    measured = np.isfinite(u)
    size = min(np.ptp(x[measured]), np.ptp(z[measured]))
    det = DetectionSettings(min_width_mm=2.0 * step_px, max_width_mm=MAX_ZONE_WIDTH_FRACTION * size,
                            min_length_mm=MIN_ZONE_LENGTH_FRACTION * size, attach_tolerance_mm=0.0)
    found = detect_bands(strain.gamma_max, x, z, None, det, uz=-v, reconstructed=np.zeros(u.shape, bool))
    return _zone_mask(found, x, z) & measured, len(found.bands)


def part_a(model) -> dict:
    labels, p_model, f_rule = [], [], []
    per_zone = {"model": [], "rules": []}          # (jump, found)
    clean = {"scenes": 0, "model_alarm": 0, "rules_alarm": 0, "model_fp_points": 0, "rules_fp_points": 0, "points": 0}
    paths = sorted(TEST.glob("scene_*.npz"))
    unmeasurable = 0
    for path in paths:
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        u, v, valid, step = d["u"], d["v"], d["valid"], float(d["step_px"])
        if not np.isfinite(u).any():          # DIC could not measure this scene at all: nothing to score
            unmeasurable += 1
            continue
        prob = model.probability(u, v, valid, step)
        mask_model, _ = model.zones(prob)
        mask_rule, _ = rule_based_mask(u, v, step)
        truth, jump = d["truth_grid"], d["jump_grid"]
        have = np.isfinite(prob)
        scored = have & (truth >= 0)
        labels.append(truth[scored]); p_model.append(prob[scored]); f_rule.append(mask_rule[scored])
        if not meta["zones"]:
            clean["scenes"] += 1
            clean["model_alarm"] += int(mask_model.any()); clean["rules_alarm"] += int(mask_rule.any())
            clean["model_fp_points"] += int(mask_model[have].sum()); clean["rules_fp_points"] += int(mask_rule[have].sum())
            clean["points"] += int(have.sum())
        # Per zone: the corridor points of the strongest zone (scenes with one zone, for a clean attribution).
        if len(meta["zones"]) == 1:
            corridor = have & (truth == 1)
            if corridor.sum() >= 10:
                size = meta["zones"][0]["jump_px"]
                per_zone["model"].append((size, float(mask_model[corridor].mean()) >= FOUND_SHARE))
                per_zone["rules"].append((size, float(mask_rule[corridor].mean()) >= FOUND_SHARE))
    y = np.concatenate(labels); p = np.concatenate(p_model); r = np.concatenate(f_rule)

    def scores(flag):
        tp = np.sum(flag & (y == 1)); fp = np.sum(flag & (y == 0)); fn = np.sum(~flag & (y == 1))
        return {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
                "overlap_iou": float(tp / max(tp + fp + fn, 1))}

    curve = {}
    for name, rows in per_zone.items():
        rows = np.array(rows) if rows else np.zeros((0, 2))
        curve[name] = []
        for lo, hi in zip(JUMP_BINS[:-1], JUMP_BINS[1:]):
            sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi) if len(rows) else np.zeros(0, bool)
            curve[name].append({"jump_px": [lo, hi], "zones": int(sel.sum()),
                                "found_share": float(rows[sel, 1].mean()) if sel.any() else None})
    return {"scenes": len(paths), "unmeasurable_scenes": unmeasurable, "points_scored": int(len(y)), "zone_share": float(y.mean()),
            "model": {"average_precision": float(average_precision_score(y, p)), **scores(p >= model.threshold)},
            "rules": scores(r), "per_zone_by_jump": curve, "no_zone_scenes": clean}


def part_b(model) -> dict:
    out = {"plates": [], "no_failure_cases": []}
    for name in ("SYNTH_general_opening", "SYNTH_general_shear"):
        results = SYNTH / name / "results"
        if not (results / "fields.npz").exists():
            continue
        truth = json.loads((SYNTH / name / "truth.json").read_text(encoding="utf-8"))
        onset = int(truth.get("onset_photo", 20)) if "onset_photo" in truth else None
        probs = zone_probability(results, model)
        first, found = first_detection(probs, model.threshold)
        summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
        rule_onset = next((z["onset"] for z in summary["zones"] if z.get("onset") is not None), None)
        out["plates"].append({"test": name, "truth_onset_photo_index": onset, "model_first_photo_index": first,
                              "model_alarms_before_onset": int(sum(1 for k, f in found.items() if f and onset is not None and k < onset)),
                              "rule_based_onset_axis": rule_onset, "photos": len(probs) + 1})
    tasks = SYNTH / "VIC_TASKS"
    for folder in sorted(tasks.iterdir()) if tasks.exists() else []:
        if not (folder / "results" / "fields.npz").exists():
            continue
        probs = zone_probability(folder / "results", model)
        _, found = first_detection(probs, model.threshold)
        alarms = [k for k, f in found.items() if f]
        last = max(probs)
        out["no_failure_cases"].append({"case": folder.name, "photos_with_alarm": len(alarms), "photos": len(probs),
                                        "largest_probability_last_photo": float(np.nanmax(probs[last])),
                                        "flagged_points_last_photo": int(model.zones(probs[last])[0].sum()),
                                        "points": int(np.isfinite(probs[last]).sum())})
    return out


def part_c(model) -> dict:
    out = {}
    springer = REAL / "springer_bts" / "StrainLink analysis" / "results"
    if (springer / "fields.npz").exists():
        probs = zone_probability(springer, model)
        first, found = first_detection(probs, model.threshold)
        summary = json.loads((springer / "summary.json").read_text(encoding="utf-8"))
        axis = summary["frame_push_mm"]
        out["springer_soil_disc"] = {"model_first_zone_time_s": None if first is None else float(axis[first]),
                                     "rule_based_crack_time_s": next((z["onset"] for z in summary["zones"] if z.get("onset") is not None), None),
                                     "published_crack_time_s": 67.0,
                                     "photos_with_zone": [float(axis[k]) for k, f in sorted(found.items()) if f]}
    pypiv = REAL / "pypiv_sandbox" / "StrainLink analysis" / "results"
    if (pypiv / "fields.npz").exists():
        probs = zone_probability(pypiv, model)
        last = max(probs)
        mask, found = model.zones(probs[last])
        summary = json.loads((pypiv / "summary.json").read_text(encoding="utf-8"))
        out["pypiv_sand"] = {"model_zones_last_photo": found, "rule_based_zones": [z["mode"] for z in summary["zones"]]}
    return out


def report(result: dict) -> str:
    a = result["A"]
    lines = ["# Failure-zone model: validation", "",
             f"Model: {result['model']['model']}; trained on {result['model']['training_scenes']} SYNTHETIC scenes.", "",
             "## A. Unseen test scenes (SYNTHETIC, never seen in training)", "",
             f"{a['scenes']} scenes ({a['unmeasurable_scenes']} too poor for DIC to measure at all), {a['points_scored']} "
             f"scored points ({a['zone_share']:.1%} in zones).", "",
             f"Decision threshold {result['model'].get('threshold')} (chosen on validation scenes, not these).", "",
             "| | Model | Rule-based detector |", "|---|---|---|"]
    for key in ("precision", "recall", "overlap_iou"):
        lines.append(f"| {key.replace('_', ' ')} | {a['model'][key]:.3f} | {a['rules'][key]:.3f} |")
    lines.append(f"| average precision | {a['model']['average_precision']:.3f} | — |")
    c = a["no_zone_scenes"]
    lines += ["", f"Scenes with no zone: {c['scenes']}. Model raised a false alarm in {c['model_alarm']}, the rules in {c['rules_alarm']} "
              f"(false points: model {c['model_fp_points']}, rules {c['rules_fp_points']} of {c['points']}).", "",
              "Zones found, by the size of the jump across them:", "", "| Jump (px) | Zones | Model found | Rules found |", "|---|---|---|---|"]
    for m, r in zip(a["per_zone_by_jump"]["model"], a["per_zone_by_jump"]["rules"]):
        f = lambda v: "—" if v is None else f"{100 * v:.0f}%"  # noqa: E731
        lines.append(f"| {m['jump_px'][0]:g}–{m['jump_px'][1]:g} | {m['zones']} | {f(m['found_share'])} | {f(r['found_share'])} |")
    lines += ["", "## B. Independent synthetic tests (full StrainLink runs)", ""]
    for p in result["B"]["plates"]:
        lines.append(f"- {p['test']}: zone forms at photo index {p['truth_onset_photo_index']}; model first finds one at "
                     f"{p['model_first_photo_index']}; false alarms before it formed: {p['model_alarms_before_onset']}.")
    lines += ["", "| No-failure case | Photos with a false alarm | Flagged points (last photo) |", "|---|---|---|"]
    for n in result["B"]["no_failure_cases"]:
        lines.append(f"| {n['case']} | {n['photos_with_alarm']} of {n['photos']} | {n['flagged_points_last_photo']} of {n['points']} |")
    lines += ["", "## C. Real images", "", "```", json.dumps(result["C"], indent=1), "```"]
    return "\n".join(lines) + "\n"


def main() -> None:
    model = load()
    result = {"model": model.info, "A": part_a(model), "B": part_b(model), "C": part_c(model)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "ml_validation.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    text = report(result)
    (ROOT / "reports" / "ml_validation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
