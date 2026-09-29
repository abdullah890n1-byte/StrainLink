"""Known-answer validation of the any-object mode (SYNTHETIC DATA).

Two synthetic plate tests with exact answers: a crack that opens (70°) and
a shear band that slides (50°). Each is rendered, analysed end-to-end by
strainlink.general (object found automatically, clocks matched
automatically) and scored against its known answers.

Run:  .venv/bin/python scripts/validate_general.py [--analyse-only]
Output: data/synthetic/SYNTH_general_<mode>/ (results + accuracy.json)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from strainlink.general import run
from strainlink.synth.general import GeneralSpec, make_general_test
from strainlink.validation.known_answers import compare

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"opening": GeneralSpec(mode="opening", zone_angle_deg=70.0),
            "shear": GeneralSpec(mode="shear", zone_angle_deg=50.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--mode", choices=list(VARIANTS), nargs="*", default=list(VARIANTS))
    args = parser.parse_args()
    for mode in args.mode:
        folder = ROOT / "data" / "synthetic" / f"SYNTH_general_{mode}"
        started = time.perf_counter()
        project = folder / "project.json"
        if not (args.analyse_only and project.exists()):
            project = make_general_test(folder, VARIANTS[mode])
            print(f"[{mode}] rendered in {time.perf_counter() - started:.0f} s")
        started = time.perf_counter()
        run(project)
        print(f"[{mode}] analysed in {time.perf_counter() - started:.0f} s")
        result = compare(folder)
        (folder / "results" / "accuracy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[{mode}] Known-answer check: {result['passed']} of {result['total']} within tolerance "
              f"(flagged {result['flagged']}, silent {result['silent']})")
        for c in result["checks"]:
            print(f"  {'ok ' if c['ok'] else 'OUT'} {c['item']:<62} found {c['found']:<32} true {c['true']:<26} "
                  f"err {c['error']:<12} tol {c['tolerance']}")


if __name__ == "__main__":
    main()
