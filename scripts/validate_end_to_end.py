"""End-to-end known-answer validation (SYNTHETIC movement and sensors).

A demo test (exact wall movement, bands and sensor logs) is rendered as
photos, analysed end-to-end exactly as a user's test would be, and every
automatic result is scored against its known answer (the Accuracy tab's
checks and tolerances).

Textures:
  generated  the synthetic dark/light grains (high contrast)
  real       a real photo of sand: real grain shapes, a third of the
             contrast, real camera/JPEG character. Texture: PyPIV
             subduction-4_0034.jpg (GPL-3.0), an in-focus patch; see
             Datasets.md. Not bundled with StrainLink; read from
             data/external/pypiv_sandbox/.

Run:  .venv/bin/python scripts/validate_end_to_end.py --preset loose|medium|dense --texture generated|real
Output: data/synthetic/SYNTH_<texture>_<preset>/ (with results, report and accuracy.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from strainlink import pipeline, report
from strainlink.synth.demo import DemoSpec, make_demo
from strainlink.validation.known_answers import compare

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "data" / "external" / "pypiv_sandbox" / "subduction-4_0034.jpg"
# Rows and columns (px) of a patch that is entirely in-focus sand (checked
# with the sharpness mask of scripts/run_real_sand.py: 99.6% in focus).
PATCH_ROWS = (420, 680)
PATCH_COLS = (0, 1560)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="medium", choices=["loose", "medium", "dense"])
    parser.add_argument("--texture", default="real", choices=["generated", "real"])
    parser.add_argument("--analyse-only", action="store_true", help="re-analyse the photos rendered earlier")
    parser.add_argument("--push-step", type=float, default=1.0, help="wall push per photo, mm (default 1)")
    parser.add_argument("--max-push", type=float, default=None, help="total wall push, mm (default: the preset's 60)")
    args = parser.parse_args()

    fine = "" if args.push_step == 1.0 and args.max_push is None else f"_step{args.push_step:g}mm"
    folder = ROOT / "data" / "synthetic" / f"SYNTH_{args.texture}_{args.preset}{fine}"
    protocol = {"push_step_mm": args.push_step, "max_push_mm": args.max_push}
    if args.texture == "real":
        grey = np.asarray(Image.open(PHOTO).convert("L"), dtype=np.float64)
        patch = grey[PATCH_ROWS[0]:PATCH_ROWS[1], PATCH_COLS[0]:PATCH_COLS[1]]
        spec = DemoSpec(preset=args.preset, texture_pixels=patch, **protocol,
                        texture_source=f"real photo patch: PyPIV {PHOTO.name} rows {PATCH_ROWS} cols {PATCH_COLS} (GPL-3.0)")
    else:
        spec = DemoSpec(preset=args.preset, **protocol)

    started = time.perf_counter()
    if args.analyse_only and (folder / "project.json").exists():
        project = folder / "project.json"
    else:
        project = make_demo(folder, spec)
        print(f"Rendered in {time.perf_counter() - started:.0f} s")

    started = time.perf_counter()
    summary = pipeline.run(project)
    print(f"Analysed in {time.perf_counter() - started:.0f} s")
    report.build(folder / "results")

    result = compare(folder)
    (folder / "results" / "accuracy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Known-answer check: {result['passed']} of {result['total']} within tolerance")
    for check in result["checks"]:
        mark = "ok " if check["ok"] else "OUT"
        print(f"  {mark} {check['group']:<22} {check['item']:<55} found {check['found']:<28} true {check['true']:<18} "
              f"err {check['error']:<14} tol {check['tolerance']}")
    for flag in summary.flags:
        print("  review:", flag)


if __name__ == "__main__":
    main()
