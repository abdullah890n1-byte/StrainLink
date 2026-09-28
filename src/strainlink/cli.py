"""Command line: `strainlink app | run | demo | report`.

The desktop app is the main way to use StrainLink; the command line is for
scripting and batch processing of many tests.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _progress(stage: str, fraction: float, message: str, _last=[0.0]) -> None:
    now = time.time()
    if now - _last[0] > 2 or fraction >= 1:
        print(f"  {stage:<32} {100 * fraction:5.1f}%  {message}", flush=True)
        _last[0] = now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="strainlink", description="Geotechnical DIC with sensor fusion and correlation.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("app", help="open the desktop app")
    run = sub.add_parser("run", help="analyse a project (project.json)")
    run.add_argument("project")
    run.add_argument("--no-report", action="store_true")
    demo = sub.add_parser("demo", help="create a synthetic demo test (known answers)")
    demo.add_argument("folder")
    demo.add_argument("--preset", choices=["loose", "medium", "dense"], default="medium")
    rep = sub.add_parser("report", help="rebuild the HTML report of an analysed project")
    rep.add_argument("project")
    args = parser.parse_args(argv)

    if args.command == "app":
        from .app.main import main as app_main
        app_main()
        return 0
    if args.command == "run":
        from . import pipeline, report
        try:
            summary = pipeline.run(args.project, progress=_progress)
        except pipeline.ProjectError as error:
            print(f"Problem: {error}", file=sys.stderr)
            return 2
        if not args.no_report:
            out = report.build(Path(args.project).parent / "results")
            print(f"Report: {out}")
        for flag in summary.flags:
            print(f"  review: {flag}")
        return 0
    if args.command == "demo":
        from .synth.demo import DemoSpec, make_demo
        path = make_demo(args.folder, DemoSpec(preset=args.preset), progress=lambda f, m: _progress("Demo", f, m))
        print(f"Demo project: {path}  (SYNTHETIC DATA)")
        return 0
    if args.command == "report":
        from . import report
        print(report.build(Path(args.project).parent / "results"))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
