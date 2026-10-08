"""StrainLink's icon as files the installers need: .icns (macOS) and .ico (Windows)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ICNS_SIZES = (16, 32, 64, 128, 256, 512, 1024)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def write(folder: Path) -> dict[str, Path]:
    """Write StrainLink.ico (always) and StrainLink.icns (on a Mac) into folder."""
    from make_mac_app import _icon_png          # the one drawing of the icon
    folder.mkdir(parents=True, exist_ok=True)
    out = {"ico": folder / "StrainLink.ico"}
    _icon_png(256).save(out["ico"], format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    if sys.platform == "darwin":
        out["icns"] = folder / "StrainLink.icns"
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "StrainLink.iconset"
            iconset.mkdir()
            for size in ICNS_SIZES:
                image = _icon_png(size)
                if size <= 512:
                    image.save(iconset / f"icon_{size}x{size}.png")
                if size >= 32:
                    image.save(iconset / f"icon_{size // 2}x{size // 2}@2x.png")
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out["icns"])], check=True)
    return out


if __name__ == "__main__":
    print(write(ROOT / "build" / "icons"))
