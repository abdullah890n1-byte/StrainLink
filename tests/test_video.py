"""Video input: frames are picked at the requested spacing, timed from the video, and loaded like photos."""

import cv2
import numpy as np
import pytest

from strainlink.io.images import load_sequence
from strainlink.io.video import default_spacing_s, extract, is_video, probe


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "test run.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 10, (96, 64), False)
    if not writer.isOpened():
        pytest.skip("no H.264 encoder in this OpenCV build")
    rng = np.random.default_rng(0)
    texture = (rng.random((64, 400)) * 255).astype(np.uint8)
    for k in range(50):                        # 5 s at 10 fps; the pattern moves 2 px per frame
        writer.write(np.ascontiguousarray(texture[:, 2 * k:2 * k + 96]))
    writer.release()
    return path


def test_probe_and_default_spacing(video):
    info = probe(video)
    assert is_video(video) and info.fps == pytest.approx(10) and info.frame_count == 50
    assert default_spacing_s(info, max_frames=20) == pytest.approx(0.3)   # ceil(50/20) = 3 frames


def test_extract_times_and_reuse(video):
    folder, info, spacing = extract(video, every_s=0.5)
    seq = load_sequence(folder)
    assert len(seq) == 10 and seq.time_source == "video"
    assert np.allclose(seq.times_s, np.arange(10) * 0.5, atol=1e-3)
    first = cv2.imread(str(seq.paths[0]), cv2.IMREAD_UNCHANGED)
    assert first.ndim == 2 and first.shape == (64, 96)
    stamp = seq.paths[0].stat().st_mtime_ns
    extract(video, every_s=0.5)                 # same spacing: re-used, not rewritten
    assert seq.paths[0].stat().st_mtime_ns == stamp
    folder, _, spacing = extract(video, every_s=1.0)
    assert spacing == pytest.approx(1.0) and len(load_sequence(folder)) == 5
