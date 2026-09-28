"""Build a double-clickable StrainLink.app for this Mac.

Development build: the app launches StrainLink from this project's Python
environment (.venv), so it always runs the current code. A standalone,
shareable build (bundled Python, signed) is a separate packaging step.

Usage:  python scripts/make_mac_app.py [destination-folder]
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
ICON_SIZES = [16, 32, 64, 128, 256, 512, 1024]
ACCENT = (47, 111, 237)


def _icon_png(size: int) -> Image.Image:
    """The StrainLink mark: a rounded blue square with a white slip curve."""
    scale = 4                                        # draw large, downsample for smooth edges
    s = size * scale
    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = int(0.09 * s)
    draw.rounded_rectangle([margin, margin, s - margin, s - margin], radius=int(0.2 * s), fill=ACCENT + (255,))
    t = np.linspace(0, 1, 60)
    x = margin + (0.18 + 0.64 * t) * (s - 2 * margin)
    y = margin + (0.78 - 0.62 * t**1.6) * (s - 2 * margin)
    draw.line(list(zip(x, y)), fill=(255, 255, 255, 255), width=int(0.07 * s), joint="curve")
    ground_y = margin + 0.84 * (s - 2 * margin)
    draw.line([(margin + 0.14 * (s - 2 * margin), ground_y), (margin + 0.86 * (s - 2 * margin), ground_y)],
              fill=(255, 255, 255, 170), width=int(0.045 * s))
    return image.resize((size, size), Image.LANCZOS)


def _icns(target: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "StrainLink.iconset"
        iconset.mkdir()
        for size in ICON_SIZES[:-1]:
            _icon_png(size).save(iconset / f"icon_{size}x{size}.png")
            _icon_png(size * 2).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)], check=True)


def build(destination: Path) -> Path:
    app = destination / "StrainLink.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    python = PROJECT / ".venv" / "bin" / "python"
    launcher = macos / "StrainLink"
    launcher.write_text(
        "#!/bin/bash\n"
        "# Development build: runs StrainLink from its project environment.\n"
        f'cd "{PROJECT}"\n'
        f'exec "{python}" -m strainlink.app "$@"\n'
    )
    launcher.chmod(0o755)
    _icns(resources / "StrainLink.icns")
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump({
            "CFBundleName": "StrainLink",
            "CFBundleDisplayName": "StrainLink",
            "CFBundleIdentifier": "app.strainlink.desktop",
            "CFBundleVersion": "0.1.0",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleExecutable": "StrainLink",
            "CFBundleIconFile": "StrainLink",
            "CFBundlePackageType": "APPL",
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
        }, f)
    return app


if __name__ == "__main__":
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT)
    print(out)
