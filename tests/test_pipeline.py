"""End-to-end smoke test: a small synthetic test folder through the whole pipeline (SYNTHETIC)."""

import json

import pytest

from strainlink import pipeline, report
from strainlink.synth.demo import DemoSpec, make_demo


@pytest.fixture(scope="module")
def small_project(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    # Coarse and short so the test runs quickly: 3 mm/px, a photo every 10 mm of push.
    return make_demo(folder, DemoSpec(preset="dense", mm_per_px=3.0, push_step_mm=10.0))


def test_pipeline_runs_and_writes_results(small_project):
    summary = pipeline.run(small_project)
    results = small_project.parent / "results"
    for name in ("summary.json", "fields.npz", "bands.json", "per_photo.csv"):
        assert (results / name).exists()
    truth = json.loads((small_project.parent / "truth.json").read_text())
    assert summary.mechanics["passive"]["peak"] == pytest.approx(truth["peak_force_kN"], rel=0.02)
    assert summary.sync["source"] == "dic"
    # Coarse, short test: the photo-to-logger match must be within its *stated* uncertainty.
    error = abs(summary.frame_push_mm[-1] - 60.0)
    assert error <= 3 * summary.sync["push_sigma_mm"] + 0.05
    if summary.sync["push_sigma_mm"] > 0.25:
        assert any("matched to the logger" in f for f in summary.flags)
    assert report.build(results).exists()

    from strainlink.validation.known_answers import compare
    accuracy = compare(small_project.parent)
    assert accuracy["total"] >= 5
    json.dumps(accuracy)      # must be sendable to the app
    sensor_checks = [c for c in accuracy["checks"] if c["group"] == "Sensor results"]
    assert all(c["ok"] for c in sensor_checks)


def test_missing_sensor_file_gives_a_clear_message(tmp_path):
    project = tmp_path / "project.json"
    project.write_text(json.dumps({"calibration": {"mm_per_px": 1, "toe_px": [10, 10], "wall_height_mm": 100},
                                   "images": {"folder": "nowhere"}}))
    with pytest.raises(pipeline.ProjectError, match="No photos found"):
        pipeline.run(project)
