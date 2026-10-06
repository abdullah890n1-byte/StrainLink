"""Score one program's results against the exact answers, and rebuild the comparison table.

    .venv/bin/python scripts/head_to_head/score.py "Program name" folder/with/<case>.csv [--note "settings used"]

Each CSV: the last photo's data, columns x, y, u, v (px; y down) and optionally
exx, eyy, exy (see data/head_to_head/package/SETTINGS.md). Output:
data/head_to_head/results/<program>/scores.json and reports/head_to_head.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPORT, RESULTS, STRAIN_MARGIN_PX, aoi, cases, truth

# Share of the area a program measured is counted on cells of this size (px).
COVERAGE_CELL_PX = 10
COLUMNS = {"x": ("x",), "y": ("y",), "u": ("u",), "v": ("v",), "exx": ("exx",), "eyy": ("eyy",), "exy": ("exy",),
           "sigma": ("sigma",)}


def read(path: Path) -> pd.DataFrame:
    """A results CSV with its columns renamed to x, y, u, v, exx, eyy, exy (and sigma if present)."""
    table = pd.read_csv(path, skipinitialspace=True)
    clean = {c: re.sub(r"[^a-z]", "", c.lower().split("[")[0].split("(")[0]) for c in table.columns}
    wanted = {}
    for name, options in COLUMNS.items():
        match = [c for c, short in clean.items() if short in options]
        if match:
            wanted[match[0]] = name
    table = table[list(wanted)].rename(columns=wanted).apply(pd.to_numeric, errors="coerce")
    missing = {"x", "y", "u", "v"} - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: no column for {', '.join(sorted(missing))}")
    if "sigma" in table:
        table = table[table["sigma"] >= 0]          # Vic-2D marks uncorrelated points with sigma = -1
    return table[np.isfinite(table["u"]) & np.isfinite(table["v"])]


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def score_case(case, table: pd.DataFrame) -> dict:
    x0, y0, x1, y1 = aoi(case.shape)
    inside = (table["x"] >= x0) & (table["x"] < x1) & (table["y"] >= y0) & (table["y"] < y1)
    t = table[inside]
    if len(t) < 10:
        return {"case": case.name, "points": int(len(t)), "error": "fewer than 10 points in the area of interest"}
    x, y = t["x"].to_numpy(float), t["y"].to_numpy(float)
    true = truth(case, x, y)
    u_err = t["u"].to_numpy(float) - true["u"]
    # Programs differ on whether v (and exy) is positive up or down: the better of the two is taken, and reported.
    v = t["v"].to_numpy(float)
    v_sign = 1.0 if rms(v - true["v"]) <= rms(-v - true["v"]) else -1.0
    v_err = v_sign * v - true["v"]
    cells = set(zip(((x - x0) // COVERAGE_CELL_PX).astype(int), ((y - y0) // COVERAGE_CELL_PX).astype(int)))
    total_cells = int(np.ceil((x1 - x0) / COVERAGE_CELL_PX) * np.ceil((y1 - y0) / COVERAGE_CELL_PX))
    out = {"case": case.name, "title": case.title, "points": int(len(t)), "coverage": len(cells) / total_cells,
           "disp_rms_px": float(np.sqrt(np.mean(u_err**2 + v_err**2))), "disp_max_px": float(np.max(np.hypot(u_err, v_err))),
           "u_bias_px": float(np.mean(u_err)), "v_bias_px": float(np.mean(v_err)), "v_positive": "down" if v_sign > 0 else "up"}
    if {"exx", "eyy", "exy"} <= set(t.columns):
        deep = ((x >= x0 + STRAIN_MARGIN_PX) & (x < x1 - STRAIN_MARGIN_PX)
                & (y >= y0 + STRAIN_MARGIN_PX) & (y < y1 - STRAIN_MARGIN_PX))
        e = {k: t[k].to_numpy(float)[deep] for k in ("exx", "eyy", "exy")}
        tr = {k: true[k][deep] for k in ("exx", "eyy", "exy")}
        ok = np.isfinite(e["exx"]) & np.isfinite(e["eyy"]) & np.isfinite(e["exy"])
        if ok.sum() >= 10:
            shear_sign = 1.0 if rms((e["exy"] - tr["exy"])[ok]) <= rms((-e["exy"] - tr["exy"])[ok]) else -1.0
            parts = [np.mean((e["exx"] - tr["exx"])[ok] ** 2), np.mean((e["eyy"] - tr["eyy"])[ok] ** 2),
                     np.mean((shear_sign * e["exy"] - tr["exy"])[ok] ** 2)]
            out.update({"strain_rms": float(np.sqrt(np.mean(parts))), "strain_points": int(ok.sum()),
                        "exx_median": float(np.median(e["exx"][ok])), "exx_true_median": float(np.median(tr["exx"][ok])),
                        "eyy_median": float(np.median(e["eyy"][ok])), "eyy_true_median": float(np.median(tr["eyy"][ok]))})
    return out


def write_report() -> None:
    programs = []
    for path in sorted(RESULTS.glob("*/scores.json")):
        programs.append(json.loads(path.read_text()))
    if not programs:
        return
    names = [p["program"] for p in programs]
    lines = ["# Head-to-head on identical images (SYNTHETIC DATA, exact answers)", "",
             "Same photos, same area of interest, same scorer (`scripts/head_to_head/`). "
             "Displacement error: RMS over the whole area (px). Strain error: RMS of εxx, εyy, εxy away from the "
             "area's edge (µε). Lower is better.", ""]
    for p in programs:
        lines.append(f"- **{p['program']}**: {p.get('note', '')}")
    lines += ["", "## Displacement error (px RMS)", "", "| Case | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    order = [c.name for c in cases()]
    titles = {c.name: c.title for c in cases()}
    def cell(p, name, key, scale=1.0, digits=4):
        row = next((r for r in p["cases"] if r["case"] == name), None)
        if row is None or key not in row:
            return "—"
        return f"{row[key] * scale:.{digits}f}"
    for name in order:
        lines.append(f"| {titles[name]} | " + " | ".join(cell(p, name, "disp_rms_px") for p in programs) + " |")
    lines += ["", "## Strain error (µε RMS)", "", "| Case | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for name in order:
        lines.append(f"| {titles[name]} | " + " | ".join(cell(p, name, "strain_rms", 1e6, 0) for p in programs) + " |")
    lines += ["", "## Share of the area measured (%)", "", "| Case | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for name in order:
        lines.append(f"| {titles[name]} | " + " | ".join(cell(p, name, "coverage", 100, 0) for p in programs) + " |")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("program")
    parser.add_argument("folder")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    rows = []
    for case in cases():
        path = Path(args.folder) / f"{case.name}.csv"
        if not path.exists():
            continue
        row = score_case(case, read(path))
        rows.append(row)
        if "error" in row:
            print(f"{case.name:13s} {row['error']}")
            continue
        strain = f"strain {row['strain_rms'] * 1e6:6.0f} µε" if "strain_rms" in row else "no strain columns"
        print(f"{case.name:13s} disp {row['disp_rms_px']:.4f} px (max {row['disp_max_px']:.3f})  {strain}  "
              f"points {row['points']}  area measured {100 * row['coverage']:.0f}%")
    slug = re.sub(r"[^a-z0-9]+", "_", args.program.lower()).strip("_")
    (RESULTS / slug).mkdir(parents=True, exist_ok=True)
    (RESULTS / slug / "scores.json").write_text(json.dumps({"program": args.program, "note": args.note, "cases": rows}, indent=1))
    write_report()
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
