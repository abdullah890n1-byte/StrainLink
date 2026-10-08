"""A quick check that an installed copy really works: `StrainLink --self-test`.

Renders a small synthetic test (a plate that cracks), analyses it end to end,
builds its report and exports its fields, all in a temporary folder, without
opening a window. Used by the installer builds: a bundle that is missing a
library fails here, not on a customer's computer.
"""

from __future__ import annotations

import sys
import tempfile
import time
import traceback
from pathlib import Path

from .system import user_data_dir

LOG_NAME = "selftest.log"


def run() -> int:
    """Run the check; return 0 if everything worked. The outcome is also written to the log file."""
    log = user_data_dir()
    log.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(text: str) -> None:
        lines.append(text)
        print(text, flush=True)

    started = time.perf_counter()
    try:
        from .. import __version__, report
        from ..export import export
        from ..general import run as analyse
        from ..synth.general import GeneralSpec, make_general_test
        from ..validation.known_answers import compare

        say(f"StrainLink {__version__} self-test, Python {sys.version.split()[0]} on {sys.platform}")
        with tempfile.TemporaryDirectory() as folder:
            spec = GeneralSpec(mode="opening", zone_angle_deg=70.0, image_px=(240, 360), plate_mm=(30.0, 18.0),
                               photos=10, onset_photo=5, stretch_rate_mm_s=0.012, photo_interval_s=1.0)
            project = make_general_test(Path(folder), spec)
            say("rendered a small synthetic test (SYNTHETIC DATA)")
            summary = analyse(project)
            say(f"analysed {summary['frames']} photos: {len(summary['zones'])} zone(s) found")
            report.build(Path(folder) / "results")
            export(Path(folder) / "results", "csv")
            result = compare(Path(folder))
            say(f"known-answer check: {result['passed']} of {result['total']} within tolerance")
            kinds = [zone["mode"] for zone in summary["zones"]]
            if "opening" not in kinds:
                raise RuntimeError(f"the crack was not found (zones: {kinds})")
        # The interface: its files must be where the app looks for them, and the window toolkit
        # must load (it is imported, not opened).
        from .main import STATIC
        missing = [name for name in ("index.html", "app.js", "app.css", "plotly.min.js") if not (STATIC / name).is_file()]
        if missing:
            raise RuntimeError(f"interface files missing from {STATIC}: {', '.join(missing)}")
        import webview  # noqa: F401
        say("interface files and window toolkit found")
        say(f"SELF-TEST OK in {time.perf_counter() - started:.0f} s")
        code = 0
    except Exception:           # every failure must reach the log
        say("SELF-TEST FAILED")
        say(traceback.format_exc())
        code = 1
    (log / LOG_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return code
