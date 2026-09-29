"""Known-answer test on real sand texture (SYNTHETIC movement, REAL grain appearance).

The medium-sand demo test (exact wall movement, bands and sensor logs) is
rendered with a real photo of sand as the soil texture instead of the
generated grains, then analysed end-to-end and scored against its known
answers, exactly like the Accuracy tab. This checks StrainLink on real
grain shapes, real (low) contrast and real camera/JPEG character, which
the generated texture does not have.

Texture: PyPIV subduction-4_0034.jpg (GPL-3.0), an in-focus patch of the
sand; see Datasets.md. The photos are not bundled with StrainLink; they
are read from data/external/pypiv_sandbox/.

Run:  .venv/bin/python scripts/make_real_texture_demo.py [--preset medium]
Output: data/synthetic/SYNTH_realtexture_<preset>/ (with results and report).
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
    args = parser.parse_args()

    grey = np.asarray(Image.open(PHOTO).convert("L"), dtype=np.float64)
    patch = grey[PATCH_ROWS[0]:PATCH_ROWS[1], PATCH_COLS[0]:PATCH_COLS[1]]
    folder = ROOT / "data" / "synthetic" / f"SYNTH_realtexture_{args.preset}"
    spec = DemoSpec(preset=args.preset, texture_pixels=patch,
                    texture_source=f"real photo patch: PyPIV {PHOTO.name} rows {PATCH_ROWS} cols {PATCH_COLS} (GPL-3.0)")

    started = time.perf_counter()
    project = make_demo(folder, spec, progress=lambda f, m: print(f"\r  {m}".ljust(60), end="", flush=True))
    print(f"\nRendered in {time.perf_counter() - started:.0f} s")

    started = time.perf_counter()
    summary = pipeline.run(project)
    print(f"Analysed in {time.perf_counter() - started:.0f} s")
    report.build(folder / "results")

    result = compare(folder)
    (folder / "results" / "accuracy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nKnown-answer check: {result['passed']} of {result['total']} within tolerance")
    for check in result["checks"]:
        mark = "ok " if check["ok"] else "OUT"
        print(f"  {mark} {check['group']:<22} {check['item']:<55} found {check['found']:<28} true {check['true']:<18} "
              f"err {check['error']:<14} tol {check['tolerance']}")
    for flag in summary.flags:
        print("  review:", flag)


if __name__ == "__main__":
    main()
