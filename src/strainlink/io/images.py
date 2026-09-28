"""Image sequence discovery and timestamps.

Frames are sorted in natural order ("img2" before "img10"). Each frame
gets a time from, in order of preference: the camera's EXIF capture time
(with sub-seconds if present), a fixed interval given by the user, or the
file modification time (least reliable; flagged). The source is always
reported, so the user knows how the frames were timed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import ExifTags, Image

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TimeSource = Literal["exif", "interval", "file-time", "index"]
_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


@dataclass
class ImageSequence:
    paths: list[Path]
    times_s: np.ndarray        # seconds, relative to the first frame
    time_source: TimeSource
    absolute_start: datetime | None = None

    def __len__(self) -> int:
        return len(self.paths)


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")),
                  key=_natural_key)


def _exif_time(path: Path) -> datetime | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            stamp = sub.get(_EXIF_TAGS.get("DateTimeOriginal")) or exif.get(_EXIF_TAGS.get("DateTime"))
            if not stamp:
                return None
            moment = datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S")
            subsec = sub.get(_EXIF_TAGS.get("SubsecTimeOriginal"))
            if subsec:
                moment = moment.replace(microsecond=int(str(subsec).ljust(6, "0")[:6]))
            return moment
    except (OSError, ValueError, KeyError):
        return None


def load_sequence(folder: str | Path, interval_s: float | None = None) -> ImageSequence:
    """Find the images and time them (see module docstring for the order of preference)."""
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {folder}")
    if interval_s is not None:
        return ImageSequence(paths, np.arange(len(paths)) * float(interval_s), "interval")
    stamps = [_exif_time(p) for p in paths]
    if all(stamps) and len(set(stamps)) == len(stamps):
        start = stamps[0]
        times = np.array([(s - start).total_seconds() for s in stamps])
        if np.all(np.diff(times) > 0):
            return ImageSequence(paths, times, "exif", start)
    mtimes = np.array([p.stat().st_mtime for p in paths])
    if np.all(np.diff(mtimes) > 0):
        return ImageSequence(paths, mtimes - mtimes[0], "file-time", datetime.fromtimestamp(mtimes[0]))
    return ImageSequence(paths, np.arange(len(paths), dtype=float), "index")
