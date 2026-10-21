"""Are the stated accuracies honest? (SYNTHETIC DATA, exact answers)

StrainLink states a 95% uncertainty with every strain and displacement
(strainlink.accuracy). This script checks those statements on the Vic-2D
task suite, where the true error of every point is known:

- coverage: the share of points whose true error is inside the stated
  uncertainty (target: at least 95%);
- whole-object strain: is its true error inside its stated uncertainty?

Run after scripts/vic2d_tasks.py:  .venv/bin/python scripts/validate_accuracy.py
Output: reports/accuracy_validation.md, reports/accuracy_validation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import vic2d_tasks
from strainlink import accuracy
from strainlink.synth import cases as C

ROOT = Path(__file__).resolve().parents[1]
TARGET = 95.0
FAR_FROM_ZONE_MM = 4.0


def check(case: C.Case) -> dict:
    folder = vic2d_tasks.OUT / case.name / "results"
    f = np.load(folder / "fields.npz")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    mm = case.mm_per_px
    X, Y = f["x_mm"] / mm, summary["frame"]["height_px"] - f["z_mm"] / mm
    k = len(f["ux"]) - 1
    step = float(np.median(np.diff(X[0])))
    compressed = bool(case.jpeg_quality) or case.file_format == "jpg"
    truth = C.true_strain(case.field, X, Y, 1.0)
    coverage, needed, typical = [], [], []
    for key, true_key, sign in (("exx", "exx", 1.0), ("ezz", "eyy", 1.0), ("exz", "exy", -1.0)):
        field = f[key][k]
        known = np.isfinite(field)
        error = np.abs(field - sign * truth[true_key])[known]
        stated = accuracy.point_uncertainty(field, compressed)[known]
        coverage.append(100.0 * np.mean(error <= stated))
        needed.append(float(np.percentile(error / (stated / accuracy.COVERAGE_FACTOR), 95)))
        typical.append(float(np.median(stated)))
    tu, tv = case.field(X, Y, 1.0)
    disp_cov = []
    for found, true, strain in ((f["ux"][k] / mm, tu, f["exx"][k]), (-f["uz"][k] / mm, tv, f["ezz"][k])):
        known = np.isfinite(found)
        stated = accuracy.displacement_uncertainty(strain, found, step, compressed=compressed)[known]
        disp_cov.append(100.0 * np.mean(np.abs(found - true)[known] <= stated))
        disp_typical = float(np.median(stated))
    # Whole-object strain (median over the object), as the pipeline reports it.
    whole = []
    for key, true_key in (("exx", "exx"), ("ezz", "eyy")):
        field = f[key][k]
        known = np.isfinite(field)
        value, true = float(np.median(field[known])), float(np.median(truth[true_key][known]))
        # The error of the object's typical strain: the median of the point-by-point differences.
        # (For a uniform field that equals the difference of the medians; for a field that runs
        # from negative to positive, as in bending, the median itself is not a stable number.)
        error = abs(float(np.median((field - truth[true_key])[known])))
        stated = accuracy.average_uncertainty(value, accuracy.scatter(field).typical, int(known.sum()), compressed)
        whole.append({"value": value, "true": true, "error": error, "stated": stated, "inside": bool(error <= stated)})
    return {"case": case.name, "title": case.title, "compressed": compressed,
            "strain_coverage": float(min(coverage)), "strain_factor_needed": float(max(needed)),
            "strain_u95_typical": float(np.median(typical)),
            "displacement_coverage": float(min(disp_cov)), "displacement_u95_px": disp_typical,
            "whole": whole, "whole_inside": all(w["inside"] for w in whole),
            "limit": case.meta.get("task") == "out_of_plane"}


def independent() -> list[dict]:
    """The same statements on data that played no part in setting the factors."""
    from scipy.spatial import cKDTree
    out = []
    synthetic = ROOT / "data" / "synthetic"
    # Plate tests (a crack / a shear band): the elastic strain away from the zone is known.
    for name in ("SYNTH_general_opening", "SYNTH_general_shear"):
        results = synthetic / name / "results"
        if not (results / "fields.npz").exists():
            continue
        f = np.load(results / "fields.npz")
        truth = json.loads((synthetic / name / "truth.json").read_text(encoding="utf-8"))
        points = np.column_stack([f["x_mm"].ravel(), f["z_mm"].ravel()])
        far = cKDTree(np.asarray(truth["zone_points_mm"], float)).query(points)[0].reshape(f["x_mm"].shape) > FAR_FROM_ZONE_MM
        worst = 100.0
        for k in range(10, len(f["exx"]), 10):
            for key, true_key in (("exx", "mean_exx"), ("ezz", "mean_ezz")):
                field = f[key][k]
                known = np.isfinite(field) & far
                stated = accuracy.point_uncertainty(field)
                worst = min(worst, 100.0 * np.mean(np.abs(field - truth[true_key][k])[known] <= stated[known]))
        out.append({"data": f"{name} (away from the zone, 4 photos, εxx and εzz)", "strain_coverage": float(worst)})
    # Lens-corrected photos (resampled before DIC).
    for name, true in (("translation", (0.0, 0.0)), ("tension", (0.5 * (1.01**2 - 1), 0.5 * ((1 - 0.003) ** 2 - 1)))):
        results = synthetic / "LENS" / name / "corrected" / "results"
        if not (results / "fields.npz").exists():
            continue
        f = np.load(results / "fields.npz")
        k = len(f["exx"]) - 1
        worst = 100.0
        for key, value in (("exx", true[0]), ("ezz", true[1])):
            field = f[key][k]
            known = np.isfinite(field)
            worst = min(worst, 100.0 * np.mean(np.abs(field - value)[known] <= accuracy.point_uncertainty(field)[known]))
        out.append({"data": f"Lens-corrected {name} (εxx and εzz)", "strain_coverage": float(worst)})
    # Smaller specimens (a larger share of edge points), rendered here.
    import tempfile
    from strainlink.general import run
    for shape, photos in (((260, 380), 4), ((200, 260), 5), ((300, 300), 6)):
        folder = Path(tempfile.mkdtemp())
        centre = ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
        case = C.Case("small", "small", C.uniaxial(0.01, 0.3, centre), shape=shape, photos=photos, seed=7)
        run(C.render_case(case, folder))
        f = np.load(folder / "results" / "fields.npz")
        k = len(f["exx"]) - 1
        worst = 100.0
        for key, value in (("exx", 0.5 * (1.01**2 - 1)), ("ezz", 0.5 * ((1 - 0.003) ** 2 - 1)), ("exz", 0.0)):
            field = f[key][k]
            known = np.isfinite(field)
            worst = min(worst, 100.0 * np.mean(np.abs(field - value)[known] <= accuracy.point_uncertainty(field)[known]))
        out.append({"data": f"Small specimen {shape[1]}×{shape[0]} px, tension 1%, {photos} photos (εxx, εzz, εxz)",
                    "strain_coverage": float(worst)})
    return out


def main() -> None:
    rows = [check(case) for case in vic2d_tasks.cases()]
    (ROOT / "reports").mkdir(exist_ok=True)
    lines = ["| Case | Stated strain accuracy (95%, per point) | Points inside it | Stated displacement accuracy | Points inside it "
             "| Whole-object strain: error / stated | Inside |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        w = max(r["whole"], key=lambda x: x["error"] / max(x["stated"], 1e-12))
        lines.append(f"| {r['title']} | ±{r['strain_u95_typical'] * 1e6:.0f} µε | {r['strain_coverage']:.1f}% "
                     f"| ±{r['displacement_u95_px']:.4f} px | {r['displacement_coverage']:.1f}% "
                     f"| {w['error'] * 1e6:.0f} / {w['stated'] * 1e6:.0f} µε | {'yes' if r['whole_inside'] else 'NO'} |")
        print(f"{r['case']:13s} strain ±{r['strain_u95_typical'] * 1e6:5.0f} µε cover {r['strain_coverage']:5.1f}% (needed x{r['strain_factor_needed']:.2f})  "
              f"disp ±{r['displacement_u95_px']:.4f} px cover {r['displacement_coverage']:5.1f}%  "
              f"whole {w['error'] * 1e6:5.0f}/{w['stated'] * 1e6:5.0f} µε {'ok' if r['whole_inside'] else 'OUT'}")
    extra = independent()
    lines += ["", "Independent checks (not used to set the factors):", "",
              "| Data | Points inside the stated strain accuracy (worst component) |", "|---|---|"]
    print("\nIndependent checks:")
    for e in extra:
        lines.append(f"| {e['data']} | {e['strain_coverage']:.1f}% |")
        print(f"  {e['data']}: {e['strain_coverage']:.1f}%")
    (ROOT / "reports" / "accuracy_validation.json").write_text(json.dumps({"suite": rows, "independent": extra}, indent=1), encoding="utf-8")
    (ROOT / "reports" / "accuracy_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scored = [r for r in rows if not r["limit"]]
    print(f"\nPer-point strain coverage ≥ {TARGET:.0f}%: {sum(r['strain_coverage'] >= TARGET for r in scored)} of {len(scored)}; "
          f"displacement: {sum(r['displacement_coverage'] >= TARGET for r in scored)} of {len(scored)}; "
          f"whole-object strain inside its stated accuracy: {sum(r['whole_inside'] for r in scored)} of {len(scored)}; "
          f"independent checks ≥ {TARGET:.0f}%: {sum(e['strain_coverage'] >= TARGET for e in extra)} of {len(extra)}")


if __name__ == "__main__":
    main()
