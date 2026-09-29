"""Known-answer validation through a compressed VIDEO (SYNTHETIC movement and sensors).

The photos of an analysed demo test are encoded as an H.264 video, the
way a camera or phone records a test. StrainLink then reads the video
(frame extraction and timing, io/video.py), analyses it end-to-end, and
every automatic result is scored against the test's known answers. This
measures what video compression does to accuracy.

Each photo is held for HOLD_FRAMES video frames, so the video's clock
matches the logger's (one photo every interval_s), and frames are taken
at that interval.

Run:  .venv/bin/python scripts/validate_video.py [--source data/synthetic/SYNTH_demo_medium] [--crf 23]
Output: data/synthetic/SYNTH_video_<preset>/
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from strainlink import pipeline, report
from strainlink.io.images import list_images
from strainlink.io.video import extract, frames_folder
from strainlink.validation.known_answers import compare

ROOT = Path(__file__).resolve().parents[1]
HOLD_FRAMES = 10
COPIED = ("logger.csv", "calibration.csv", "truth.json")


def encode(photos: list[Path], video: Path, fps: float) -> None:
    first = cv2.imread(str(photos[0]), cv2.IMREAD_GRAYSCALE)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"avc1"), fps, (first.shape[1], first.shape[0]), False)
    if not writer.isOpened():
        raise RuntimeError("no H.264 encoder available")
    for path in photos:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        for _ in range(HOLD_FRAMES):
            writer.write(frame)
    writer.release()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    return 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "data" / "synthetic" / "SYNTH_demo_medium"))
    args = parser.parse_args()
    source = Path(args.source)
    project = json.loads((source / "project.json").read_text())
    truth = json.loads((source / "truth.json").read_text())
    interval = float(project["images"]["interval_s"])
    photos = list_images(source / project["images"]["folder"])

    folder = ROOT / "data" / "synthetic" / f"SYNTH_video_{truth['preset']}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in COPIED:
        shutil.copy(source / name, folder / name)
    video = folder / "test video.mp4"
    encode(photos, video, HOLD_FRAMES / interval)

    frames, info, spacing = extract(video, every_s=interval)
    extracted = list_images(frames)
    quality = [psnr(cv2.imread(str(a), cv2.IMREAD_GRAYSCALE), cv2.imread(str(b), cv2.IMREAD_GRAYSCALE))
               for a, b in zip(photos, extracted)]
    print(f"Video: {video.stat().st_size / 1e6:.1f} MB (photos {sum(p.stat().st_size for p in photos) / 1e6:.0f} MB), "
          f"{info.frame_count} frames at {info.fps:.3f} fps; {len(extracted)} frames taken every {spacing:.1f} s; "
          f"picture quality vs the photos: PSNR median {np.median(quality):.1f} dB (min {min(quality):.1f})")

    project["name"] = project["name"].replace("(SYNTHETIC DATA)", "— from an H.264 video (SYNTHETIC DATA)")
    project["images"] = {"folder": frames_folder(video).name, "video": video.name, "video_every_s": spacing}
    (folder / "project.json").write_text(json.dumps(project, indent=2))

    started = time.perf_counter()
    summary = pipeline.run(folder / "project.json")
    print(f"Analysed in {time.perf_counter() - started:.0f} s; frames timed from: {summary.__dict__.get('time_source', 'video')}")
    report.build(folder / "results")
    result = compare(folder)
    result["video"] = {"size_mb": video.stat().st_size / 1e6, "psnr_median_db": float(np.median(quality)),
                       "psnr_min_db": float(min(quality))}
    (folder / "results" / "accuracy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nKnown-answer check: {result['passed']} of {result['total']} within tolerance")
    for check in result["checks"]:
        if not check["ok"]:
            print(f"  OUT {check['item']}: found {check['found']}, true {check['true']}, err {check['error']}, tol {check['tolerance']}")


if __name__ == "__main__":
    main()
