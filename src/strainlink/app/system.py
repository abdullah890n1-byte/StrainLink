"""What differs between operating systems, and between running from source and from an installed app."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_NAME = "StrainLink"


def data_dir_for(platform: str, os_name: str, environment: dict, home: Path) -> Path:
    """StrainLink's own folder on a given system (separate from user_data_dir so it can be tested anywhere)."""
    if platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if os_name == "nt":
        return Path(environment.get("APPDATA") or home / "AppData" / "Roaming") / APP_NAME
    return Path(environment.get("XDG_DATA_HOME") or home / ".local" / "share") / APP_NAME


def user_data_dir() -> Path:
    """Where StrainLink keeps its own files (recent list, remembered details, demos, logs)."""
    return data_dir_for(sys.platform, os.name, dict(os.environ), Path.home())


def reveal(path: str | Path) -> None:
    """Show a file or folder in Finder / File Explorer / the file manager."""
    path = Path(path)
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif os.name == "nt":
        # explorer returns a non-zero code even when it works.
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", str(path if path.is_dir() else path.parent)], check=False)


def is_packaged() -> bool:
    """True inside an installed app (a PyInstaller bundle), False when run from source."""
    return bool(getattr(sys, "frozen", False))
