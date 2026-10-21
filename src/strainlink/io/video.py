"""Test videos: pick frames at a chosen spacing and save them as a timed image sequence.

A video of a soil test is turned into a folder of frames once; from then on
it is analysed exactly like a folder of photos (same DIC, same checks).

- Frames are saved as lossless 8-bit grey PNGs (DIC works on grey levels),
  so no second round of compression is added to the video's own.
- Each frame's time comes from the video itself (its presentation time,
  which is right for variable-frame-rate phone videos too) and is written
  to frames.csv; the image loader reads it (time source "video").
- Frame spacing matters for accuracy: too far apart and the grains inside
  a shear band rearrange between frames and lose correlation (found on
  real sand, see Validation.md); too close and the analysis takes longer
  and small tracking errors accumulate over more steps. The default keeps
  at most DEFAULT_MAX_FRAMES frames, never closer than the video's frames.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".wmv"}
FRAMES_FILE = "frames.csv"
SOURCE_FILE = "video_source.json"
DEFAULT_MAX_FRAMES = 200
# Spacing is rounded to a whole number of video frames.
MIN_SPACING_FRAMES = 1


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES and not Path(path).name.startswith(".")


def probe(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open the video {Path(path).name} (format not supported?)")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or count <= 1:
        raise ValueError(f"{Path(path).name} has no readable frame rate or frames")
    return VideoInfo(str(path), fps, count, count / fps, width, height)


def default_spacing_s(info: VideoInfo, max_frames: int = DEFAULT_MAX_FRAMES) -> float:
    """Smallest spacing (whole video frames) that keeps the frame count within max_frames."""
    step = max(MIN_SPACING_FRAMES, int(np.ceil(info.frame_count / max_frames)))
    return step / info.fps


def frames_folder(video: str | Path) -> Path:
    video = Path(video)
    return video.parent / f"{video.stem} frames"


def extract(video: str | Path, every_s: float | None = None, folder: str | Path | None = None,
            progress: Callable[[float, str], None] | None = None) -> tuple[Path, VideoInfo, float]:
    """Save every `every_s` seconds of the video as a grey PNG; returns (folder, info, spacing used).

    Re-uses an earlier extraction with the same spacing. A folder from a
    different spacing is replaced (only files this function wrote).
    """
    info = probe(video)
    step = max(MIN_SPACING_FRAMES, int(round((every_s or default_spacing_s(info)) * info.fps)))
    spacing = step / info.fps
    out = Path(folder) if folder else frames_folder(video)
    source = {"video": Path(video).name, "size_bytes": Path(video).stat().st_size, "step_frames": step}
    if _up_to_date(out, source):
        return out, info, spacing
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("frame_*.png"):
        old.unlink()

    capture = cv2.VideoCapture(str(video))
    rows = []
    try:
        index = 0
        while True:
            # grab() only demuxes; frames that are skipped are never decoded to pixels.
            if not capture.grab():
                break
            if index % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                time_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                name = f"frame_{len(rows):05d}.png"
                cv2.imwrite(str(out / name), grey)
                rows.append((name, index, time_ms / 1000.0 if time_ms > 0 or index == 0 else index / info.fps))
                if progress and len(rows) % 10 == 0:
                    progress(min(index / info.frame_count, 1.0), f"Reading the video: frame {len(rows)}")
            index += 1
    finally:
        capture.release()
    if len(rows) < 2:
        raise ValueError(f"only {len(rows)} frame(s) could be read from {Path(video).name}")

    times = np.array([r[2] for r in rows])
    if not np.all(np.diff(times) > 0):
        # Presentation times missing or unreliable in this file: fall back to the nominal frame rate.
        rows = [(name, i, i / info.fps) for name, i, _ in rows]
    with open(out / FRAMES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "video_frame", "time_s"])
        writer.writerows((name, i, f"{t:.6f}") for name, i, t in rows)
    (out / SOURCE_FILE).write_text(json.dumps({**source, "info": asdict(info), "spacing_s": spacing}, indent=1), encoding="utf-8")
    if progress:
        progress(1.0, f"{len(rows)} frames ready")
    return out, info, spacing


def _up_to_date(folder: Path, source: dict) -> bool:
    try:
        saved = json.loads((folder / SOURCE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(saved.get(k) == v for k, v in source.items()) and (folder / FRAMES_FILE).exists()


def read_frame_times(folder: str | Path) -> dict[str, float] | None:
    """{file name: time (s)} from an extracted-frames folder, or None if it is not one."""
    path = Path(folder) / FRAMES_FILE
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"]: float(row["time_s"]) for row in csv.DictReader(f)}
