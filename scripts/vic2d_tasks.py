"""The Vic-2D task suite: every standard DIC job, with exact answers (SYNTHETIC DATA).

Each case is rendered (synth/cases.py), analysed end-to-end by StrainLink's
any-object mode exactly as a user's test would be, and scored point by
point against the analytical solution:

- displacement error (px RMS) over the measured area;
- strain error (Green-Lagrange, µε RMS) away from the area's edges;
- the task's own engineering result (Poisson's ratio, rotation, strain
  concentration, curvature and neutral axis, ...).

Run:  .venv/bin/python scripts/vic2d_tasks.py [--only name ...] [--analyse-only]
Output: data/synthetic/VIC_TASKS/<case>/, reports/vic2d_tasks.md and .json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from strainlink.general import run
from strainlink.synth import cases as C

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic" / "VIC_TASKS"
REPORT = ROOT / "reports"
CENTRE = (300.0, 200.0)
HOLE_R = 50.0
# Points within this many grid steps of the measured area's edge are left out of strain scoring
# (the strain window and the subset reach past the edge there).
EDGE_STEPS = 4
# Rings around the hole (in radii) for the edge check, in a sector around the line through the
# hole's centre across the load, where the concentration is highest (Kirsch: 3 at the edge).
HOLE_RINGS = ((1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.5), (1.5, 2.0))
HOLE_SECTOR_DEG = 15.0
# Tolerances.
DISP_TOL_PX = {"clean": 0.05, "degraded": 0.15}
STRAIN_TOL = {"clean": 2e-4, "degraded": 5e-4}       # RMS, absolute (200 / 500 µε)


def cases() -> list[C.Case]:
    tension = dict(field=C.uniaxial(0.01, 0.3, CENTRE), meta={"strain": 0.01, "poisson": 0.3})
    return [
        C.Case("translation", "Rigid translation, sub-pixel (3.37, −1.82 px)", C.translation(3.37, -1.82),
               meta={"task": "translation", "du": 3.37, "dv": -1.82}),
        C.Case("rotation", "Rigid rotation 2°", C.rotation(2.0, CENTRE), meta={"task": "rotation", "degrees": 2.0}),
        C.Case("tension", "Tension test 1% (ν = 0.3)", **tension | {"meta": {"task": "uniaxial", **tension["meta"]}}),
        C.Case("compression", "Compression test −1% (ν = 0.3)", C.uniaxial(-0.01, 0.3, CENTRE),
               meta={"task": "uniaxial", "strain": -0.01, "poisson": 0.3}),
        C.Case("shear", "Simple shear γ = 1%", C.simple_shear(0.01, CENTRE), meta={"task": "shear", "gamma": 0.01}),
        C.Case("large_strain", "Large strain: 30% stretch (rubber-like, ν = 0.5)", C.large_tension(0.30, 0.5, CENTRE),
               photos=12, meta={"task": "large", "strain": 0.30, "poisson": 0.5}),
        C.Case("hole", "Plate with a hole (Kirsch), 0.3% remote strain", C.plate_with_hole(0.003, 0.3, HOLE_R, CENTRE),
               region=lambda X, Y: np.hypot(X - CENTRE[0], Y - CENTRE[1]) > HOLE_R,
               meta={"task": "hole", "strain": 0.003, "poisson": 0.3, "radius_px": HOLE_R}),
        C.Case("bending", "Beam bending (0.8% at the outer fibres)", C.bending(4e-5, 200.0, 300.0),
               meta={"task": "bending", "curvature_per_px": 4e-5, "neutral_row": 200.0}),
        C.Case("thermal", "Thermal expansion 0.1% (small uniform strain)", C.isotropic(0.001, CENTRE),
               meta={"task": "isotropic", "strain": 0.001}),
        C.Case("static", "Static images (noise floor)", C.translation(0.0, 0.0), meta={"task": "static"}),
        C.Case("noise_high", "Tension, heavy image noise (6 grey levels)", noise_grey=6.0, **tension | {"meta": {"task": "uniaxial", "quality": "degraded", **tension["meta"]}}),
        C.Case("low_contrast", "Tension, low contrast (100–150 grey)", contrast=(100.0, 150.0), **tension | {"meta": {"task": "uniaxial", "quality": "degraded", **tension["meta"]}}),
        C.Case("blur", "Tension, slightly out of focus (blur 1.2 px)", blur_px=1.2, **tension | {"meta": {"task": "uniaxial", "quality": "degraded", **tension["meta"]}}),
        C.Case("lighting", "Tension, lighting brightens 25% during the test", lighting_drift=0.25, **tension | {"meta": {"task": "uniaxial", **tension["meta"]}}),
        C.Case("jpeg60", "Tension, strong JPEG compression (quality 60)", jpeg_quality=60, **tension | {"meta": {"task": "uniaxial", "quality": "degraded", **tension["meta"]}}),
        C.Case("tif16", "Tension, 16-bit TIFF", file_format="tif16", photos=5, **tension | {"meta": {"task": "uniaxial", **tension["meta"]}}),
        C.Case("png", "Tension, PNG files", file_format="png", photos=5, **tension | {"meta": {"task": "uniaxial", **tension["meta"]}}),
        C.Case("jpg", "Tension, JPG files (quality 90)", file_format="jpg", photos=5, **tension | {"meta": {"task": "uniaxial", **tension["meta"]}}),
        C.Case("bmp", "Tension, BMP files", file_format="bmp", photos=5, **tension | {"meta": {"task": "uniaxial", **tension["meta"]}}),
        C.Case("out_of_plane", "LIMIT: specimen moves 0.5% closer to the camera (no real strain)", C.isotropic(0.005, CENTRE),
               meta={"task": "out_of_plane", "apparent": 0.005}),
    ]


def score(case: C.Case, folder: Path) -> dict:
    res = folder / "results"
    f = np.load(res / "fields.npz")
    summary = json.loads((res / "summary.json").read_text())
    mm = case.mm_per_px
    height = summary["frame"]["height_px"]
    X, Y = f["x_mm"] / mm, height - f["z_mm"] / mm
    k = len(f["ux"]) - 1
    load = 1.0
    u, v = f["ux"][k] / mm, -f["uz"][k] / mm
    tu, tv = case.field(X, Y, load)
    measured = np.isfinite(u)
    disp_err = np.hypot(u - tu, v - tv)[measured]
    # Strain away from the edges of the measured area.
    from scipy import ndimage
    interior = ndimage.binary_erosion(measured, iterations=EDGE_STEPS)
    truth = C.true_strain(case.field, X, Y, load)
    found = {"exx": f["exx"][k], "eyy": f["ezz"][k], "exy": -f["exz"][k]}      # z up -> image y down flips exy
    ok = interior & np.isfinite(found["exx"])
    strain_err = np.sqrt(np.mean([(found[c][ok] - truth[c][ok]) ** 2 for c in ("exx", "eyy", "exy")], axis=0))
    quality = case.meta.get("quality", "clean")
    out = {
        "case": case.name, "title": case.title, "points": int(measured.sum()),
        "measured_share": float(measured.mean()),
        "disp_rms_px": float(np.sqrt(np.mean(disp_err**2))), "disp_max_px": float(disp_err.max()),
        "strain_rms": float(np.sqrt(np.mean(strain_err**2))) if ok.any() else float("nan"),
        "zones": [z["name"] + ":" + z["mode"] for z in summary.get("zones", [])],
        "seconds": summary["seconds"], "quality": quality,
    }
    out.update(task_result(case, found, truth, X, Y, ok, f, k, measured))
    disp_ok = out["disp_rms_px"] <= DISP_TOL_PX[quality]
    strain_ok = not np.isfinite(out["strain_rms"]) or out["strain_rms"] <= STRAIN_TOL[quality] * max(1.0, out.get("strain_scale", 1.0))
    out["pass"] = bool(disp_ok and strain_ok and out.get("task_ok", True))
    if case.meta.get("task") == "out_of_plane":
        out["pass"] = None          # a known 2D limit, demonstrated, not scored
    return out


def task_result(case, found, truth, X, Y, ok, f, k, measured) -> dict:
    m, task = case.meta, case.meta.get("task")
    med = lambda a: float(np.median(a[ok]))  # noqa: E731
    if task == "translation":
        u = f["ux"][k][measured] / case.mm_per_px
        v = -f["uz"][k][measured] / case.mm_per_px
        bias = (float(np.mean(u) - m["du"]), float(np.mean(v) - m["dv"]))
        return {"result": f"mean move ({np.mean(u):.4f}, {np.mean(v):.4f}) px vs ({m['du']}, {m['dv']})",
                "task_ok": bool(max(abs(b) for b in bias) <= 0.01)}
    if task == "rotation":
        strain = max(abs(med(found["exx"])), abs(med(found["eyy"])), abs(med(found["exy"])))
        u, v = f["ux"][k] / case.mm_per_px, -f["uz"][k] / case.mm_per_px
        gy_u, gx_u = np.gradient(u)
        gy_v, gx_v = np.gradient(v)
        step = float(np.nanmedian(np.diff(X[0])))
        # Rigid rotation: sin(angle) = (dv/dx - du/dy) / 2, gradients per pixel (np.gradient is per grid step).
        angle = float(np.degrees(np.arcsin(np.nanmedian((gx_v - gy_u)[ok]) / step / 2)))
        return {"result": f"rotation {angle:.3f}° (true {m['degrees']}°); false strain {strain * 1e6:.0f} µε",
                "task_ok": bool(abs(angle - m["degrees"]) <= 0.02 and strain <= 1e-4)}
    if task in ("uniaxial", "large"):
        exx, eyy = med(found["exx"]), med(found["eyy"])
        lam = 1 + m["strain"]
        if task == "large":
            true_exx = 0.5 * (lam**2 - 1)
            nu = float(-np.log(np.sqrt(2 * eyy + 1)) / np.log(np.sqrt(2 * exx + 1)))      # from Hencky strains
            return {"result": f"εxx {exx * 100:.3f}% (true {true_exx * 100:.3f}%), ν {nu:.3f} (true {m['poisson']})",
                    "task_ok": bool(abs(exx - true_exx) <= 0.01 * true_exx and abs(nu - m["poisson"]) <= 0.02),
                    "strain_scale": 10.0}
        e_x, e_y = np.sqrt(2 * exx + 1) - 1, np.sqrt(2 * eyy + 1) - 1          # engineering strains from Green-Lagrange
        nu = -e_y / e_x
        return {"result": f"εxx {e_x * 100:.4f}% (true {m['strain'] * 100:.2f}%), ν {nu:.4f} (true {m['poisson']})",
                "task_ok": bool(abs(e_x - m["strain"]) <= 0.01 * abs(m["strain"]) and abs(nu - m["poisson"]) <= 0.02)}
    if task == "shear":
        exy = med(found["exy"])
        return {"result": f"γ {2 * exy * 100:.4f}% (true {m['gamma'] * 100:.2f}%)",
                "task_ok": bool(abs(2 * exy - m["gamma"]) <= 0.01 * m["gamma"] + 2e-5)}
    if task == "isotropic":
        exx, eyy = med(found["exx"]), med(found["eyy"])
        return {"result": f"εxx {exx * 1e6:.0f} µε, εyy {eyy * 1e6:.0f} µε (true {m['strain'] * 1e6:.0f})",
                "task_ok": bool(max(abs(exx - m["strain"]), abs(eyy - m["strain"])) <= 3e-5)}
    if task == "static":
        noise = float(np.sqrt(np.mean([found[c][ok] ** 2 for c in ("exx", "eyy", "exy")])))
        return {"result": f"strain noise floor {noise * 1e6:.0f} µε RMS", "task_ok": bool(noise <= 1e-4)}
    if task == "hole":
        # Strain concentration: εxx along the vertical line above/below the hole, vs the Kirsch solution.
        r = np.hypot(X - CENTRE[0], Y - CENTRE[1])
        line = ok & (np.abs(X - CENTRE[0]) <= 3) & (r >= 1.3 * m["radius_px"]) & (r <= 2.5 * m["radius_px"])
        ratio = found["exx"][line] / truth["exx"][line]
        nearest = ok & (np.abs(X - CENTRE[0]) <= 3)
        peak_found = float(np.max(found["exx"][nearest])) / m["strain"]
        peak_true = float(np.max(truth["exx"][nearest])) / m["strain"]
        hole_ok = not measured[np.hypot(X - CENTRE[0], Y - CENTRE[1]) < 0.8 * m["radius_px"]].any()
        # Right at the edge: every measured strain point (the general score leaves out the area's edges).
        has_strain = np.isfinite(found["exx"])
        # Angle from the line across the load (vertical in the photo), both above and below the hole.
        across = np.degrees(np.abs(np.arctan2(X - CENTRE[0], np.abs(Y - CENTRE[1]))))
        sector = across <= HOLE_SECTOR_DEG
        rings = {}
        for lo, hi in HOLE_RINGS:
            sel = has_strain & sector & (r >= lo * m["radius_px"]) & (r < hi * m["radius_px"])
            if sel.any():
                rings[f"{lo}-{hi}"] = round(float(np.median(found["exx"][sel] / truth["exx"][sel])), 3)
        nearest_r = float(r[has_strain].min() / m["radius_px"])
        ring_text = ", ".join(f"{k}: {v:.2f}" for k, v in rings.items())
        return {"result": f"εxx near the hole / remote: {peak_found:.2f} (Kirsch at the same points {peak_true:.2f}; 3.00 at the edge); "
                          f"found/true on the line 1.3–2.5 radii: {np.median(ratio):.3f}; hole left out: {'yes' if hole_ok else 'NO'}; "
                          f"nearest strain point {nearest_r:.2f} radii; found/Kirsch by ring (radii) {ring_text}",
                "task_ok": bool(abs(np.median(ratio) - 1) <= 0.05 and hole_ok), "strain_scale": 3.0,
                "hole_rings": rings, "nearest_strain_radii": nearest_r}
    if task == "bending":
        rows = ok & (np.abs(X - 300) <= 100)
        slope, intercept = np.polyfit(Y[rows], found["exx"][rows], 1)
        neutral = -intercept / slope
        return {"result": f"curvature {slope:.3e}/px (true {m['curvature_per_px']:.1e}), neutral axis row {neutral:.2f} (true {m['neutral_row']})",
                "task_ok": bool(abs(slope / m["curvature_per_px"] - 1) <= 0.03 and abs(neutral - m["neutral_row"]) <= 1.0)}
    if task == "out_of_plane":
        exx = med(found["exx"])
        return {"result": f"apparent strain {exx * 100:.3f}% where there is none (2D cannot see depth; Vic-3D/stereo needed)",
                "task_ok": None}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True)
    results = []
    for case in cases():
        if args.only and case.name not in args.only:
            continue
        folder = OUT / case.name
        t0 = time.perf_counter()
        project = folder / "project.json"
        if not (args.analyse_only and project.exists()):
            project = C.render_case(case, folder)
        run(project)
        r = score(case, folder)
        r["total_seconds"] = time.perf_counter() - t0
        results.append(r)
        mark = {True: "PASS", False: "FAIL", None: "LIMIT"}[r["pass"]]
        print(f"{mark:5s} {case.name:13s} disp {r['disp_rms_px']:.4f} px  strain {r['strain_rms'] * 1e6:7.0f} µε  | {r.get('result', '')}", flush=True)
    if args.only:
        # A partial run updates its own cases and keeps the others' last results.
        saved = REPORT / "vic2d_tasks.json"
        previous = {r["case"]: r for r in json.loads(saved.read_text())} if saved.exists() else {}
        previous.update({r["case"]: r for r in results})
        results = [previous[c.name] for c in cases() if c.name in previous]
    (REPORT / "vic2d_tasks.json").write_text(json.dumps(results, indent=1))
    lines = ["| Case | Result | Displacement RMS (px) | Strain RMS (µε) | Task result |", "|---|---|---|---|---|"]
    for r in results:
        mark = {True: "✅ pass", False: "❌ fail", None: "⚠ known 2D limit"}[r["pass"]]
        lines.append(f"| {r['title']} | {mark} | {r['disp_rms_px']:.4f} | {r['strain_rms'] * 1e6:.0f} | {r.get('result', '')} |")
    (REPORT / "vic2d_tasks.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
