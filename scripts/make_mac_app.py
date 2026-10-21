"""Build a double-clickable StrainLink.app for this Mac.

Development build: the app launches StrainLink from this project's Python
environment (.venv), so it always runs the current code. A standalone,
shareable build (bundled Python, signed) is a separate packaging step.

Why a compiled launcher and a signature: macOS privacy protection (TCC)
guards Documents, Desktop and Downloads. It could not identify the earlier
unsigned app whose main program was a shell script, so it silently refused
access (the app died reading .venv) instead of asking. A small Mach-O
launcher, folder-usage texts in Info.plist and an ad-hoc signature give the
app a stable identity: macOS asks once and remembers the answer.

Usage:  python scripts/make_mac_app.py [destination-folder]   (default: ~/Applications)
Launch log: ~/Library/Logs/StrainLink/app.log
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
LOG_DIR = Path.home() / "Library" / "Logs" / "StrainLink"
BUNDLE_ID = "app.strainlink.desktop"
# Shown by macOS when StrainLink first opens a test folder in a protected place.
FOLDER_REASON = "StrainLink reads your test photos and data-logger files, and saves the analysis next to them."

LAUNCHER_C = r"""
// StrainLink launcher: runs the StrainLink desktop app with the project's Python.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    mkdir(LOG_DIR, 0755);
    // Keep a log of the last launch, so a failure is never silent.
    freopen(LOG_DIR "/app.log", "w", stdout);
    freopen(LOG_DIR "/app.log", "a", stderr);
    setenv("PYTHONUNBUFFERED", "1", 1);
    char *args[] = {PYTHON, "-m", "strainlink.app", NULL};
    execv(PYTHON, args);
    perror("StrainLink could not start Python " PYTHON);
    return 1;
}
"""
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "launcher.c"
        source.write_text(LAUNCHER_C, encoding="utf-8")
        subprocess.run(["clang", "-O2", "-Wall", f'-DPYTHON="{python}"', f'-DLOG_DIR="{LOG_DIR}"',
                        str(source), "-o", str(macos / "StrainLink")], check=True)
    _icns(resources / "StrainLink.icns")
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump({
            "CFBundleName": "StrainLink",
            "CFBundleDisplayName": "StrainLink",
            "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleVersion": "0.1.0",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleExecutable": "StrainLink",
            "CFBundleIconFile": "StrainLink",
            "CFBundlePackageType": "APPL",
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSDocumentsFolderUsageDescription": FOLDER_REASON,
            "NSDesktopFolderUsageDescription": FOLDER_REASON,
            "NSDownloadsFolderUsageDescription": FOLDER_REASON,
            "NSRemovableVolumesUsageDescription": FOLDER_REASON,
            "NSNetworkVolumesUsageDescription": FOLDER_REASON,
        }, f)
    # Ad-hoc signature: a stable identity for macOS privacy settings (no developer account needed).
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", "--identifier", BUNDLE_ID, str(app)], check=True)
    return app


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Applications"
    target.mkdir(parents=True, exist_ok=True)
    out = build(target)
    print(out)
