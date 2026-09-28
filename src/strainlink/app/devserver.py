"""Development server: the app's page in a normal browser, for UI checks.

`python -m strainlink.app.devserver [port]` serves the interface on
http://127.0.0.1:<port> and forwards `pywebview.api` calls over HTTP to the
same Api class the desktop app uses. Native file dialogs are unavailable
here (they need the app window); everything else behaves identically.
Binds to localhost only.
"""

from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .api import Api
from .main import STATIC, _ensure_chart_library

DEFAULT_PORT = 8765


def serve(port: int = DEFAULT_PORT) -> None:
    _ensure_chart_library()
    api = Api()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC), **kwargs)

        def do_POST(self):  # noqa: N802 (http.server naming)
            name = self.path.removeprefix("/api/")
            if name.startswith("_") or not hasattr(api, name):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            args = json.loads(self.rfile.read(length) or b"[]")
            try:
                result = getattr(api, name)(*args)
            except Exception as error:   # report to the page, like the app would
                result = {"ok": False, "error": str(error)}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            # Always serve the current files while developing the interface.
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT)
