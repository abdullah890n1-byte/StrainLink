"""Build the stand-alone macOS app and its disk image (.dmg).

    .venv/bin/python packaging/build_mac.py

Output: dist/StrainLink.app and dist/StrainLink-<version>-macOS-<arch>.dmg

The app carries its own Python and libraries, so it runs on a Mac with nothing
else installed (same processor type as the Mac it was built on). It is signed
ad hoc: without an Apple Developer ID and notarisation, macOS warns the first
time it is opened on another Mac (right-click, Open). The build ends by running
the app's own self-test.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from icons import write as write_icons  # noqa: E402

from strainlink import __version__  # noqa: E402

DIST = ROOT / "dist"
BUNDLE_ID = "com.strainlink.app"


def run(*command: str) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("build_mac.py runs on a Mac; on Windows use packaging\\build_windows.bat")
    # The charts' JavaScript file must be in the static folder before it is bundled.
    from strainlink.app.main import _ensure_chart_library
    _ensure_chart_library()
    write_icons(ROOT / "build" / "icons")
    app = DIST / "StrainLink.app"
    if app.exists():
        shutil.rmtree(app)
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", str(DIST), "--workpath", str(ROOT / "build" / "pyinstaller"),
        str(Path(__file__).parent / "StrainLink.spec"))
    shutil.rmtree(DIST / "StrainLink", ignore_errors=True)        # the unbundled folder; the .app has it all
    run("codesign", "--force", "--deep", "--sign", "-", "--identifier", BUNDLE_ID, str(app))
    # The built app must work before it is packed.
    run(str(app / "Contents" / "MacOS" / "StrainLink"), "--self-test")
    image = DIST / f"StrainLink-{__version__}-macOS-{platform.machine()}.dmg"
    image.unlink(missing_ok=True)
    staging = ROOT / "build" / "dmg"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    run("ditto", str(app), str(staging / "StrainLink.app"))
    (staging / "Applications").symlink_to("/Applications")        # drag the app onto this to install
    run("hdiutil", "create", "-volname", "StrainLink", "-srcfolder", str(staging), "-ov", "-format", "UDZO", str(image))
    shutil.rmtree(staging)
    # Disk use, as Finder shows it (the bundle links some folders twice; they are stored once).
    size = int(subprocess.run(["du", "-sk", str(app)], capture_output=True, text=True, check=True).stdout.split()[0]) / 1e3
    print(f"\nApp:  {app}  ({size:.0f} MB)\nDisk image:  {image}  ({image.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
