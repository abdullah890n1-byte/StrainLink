"""Launch the StrainLink desktop app: `python -m strainlink.app` (or the packaged .app)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .system import is_packaged

STATIC = Path(__file__).parent / "static"
WINDOW_SIZE = (1360, 880)
MIN_WINDOW_SIZE = (1100, 720)


def _ensure_chart_library() -> None:
    """Charts use Plotly's bundled JavaScript, copied next to the page (offline)."""
    target = STATIC / "plotly.min.js"
    if target.exists():
        return
    if is_packaged():
        # An installed app must not write inside itself (it would break its signature); the
        # build puts the file there.
        raise RuntimeError("This copy of StrainLink is incomplete (plotly.min.js is missing): install it again.")
    import plotly

    shutil.copyfile(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", target)


def main() -> None:
    if "--self-test" in sys.argv:
        from .selftest import run
        raise SystemExit(run())
    import webview

    from .api import Api
    _ensure_chart_library()
    api = Api()
    window = webview.create_window("StrainLink", str(STATIC / "index.html"), js_api=api,
                                   width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], min_size=MIN_WINDOW_SIZE,
                                   background_color="#f5f6fa")
    api.attach(window)
    webview.start()


if __name__ == "__main__":
    main()
