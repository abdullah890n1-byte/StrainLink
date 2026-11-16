"""The failure-zone model's building blocks, and the model itself on clear cases (SYNTHETIC)."""

import numpy as np
import pytest

from strainlink.ml import features
from strainlink.ml.model import MODEL_FILE, MIN_ZONE_SPAN_STEPS, load, zones
from strainlink.ml.scenes import CORRIDOR_PX, Scene, Zone, random_scene

QUIET = {"exx": 0.002, "eyy": -0.001, "exy": 0.0, "rotation_deg": 0.0, "curvature": 0.0, "waves": [], "hole_strain": 0.0}


def _crack_scene(jump_px: float, seed: int = 3) -> Scene:
    zone = Zone(start=np.array([-50.0, 150.0]), direction=np.array([1.0, 0.0]), length=1000.0,
                jump=np.array([0.0, jump_px]), width=2.0, taper=40.0, mode="opening")
    return Scene(shape=(300, 400), background=QUIET, zones=[zone], noise_grey=1.0, seed=seed)


def test_truth_marks_the_zone_corridor_and_leaves_a_margin():
    scene = _crack_scene(1.0)
    y = np.array([150.0, 150.0 + CORRIDOR_PX - 1, 150.0 + 1.5 * CORRIDOR_PX, 290.0])
    labels = scene.truth(np.full(4, 200.0), y)
    assert list(labels) == [1, 1, -1, 0]
    assert scene.jump_size(np.array([200.0]), np.array([150.0]))[0] == pytest.approx(1.0)


def test_zone_jump_fades_to_zero_beyond_the_tip():
    zone = Zone(start=np.array([0.0, 0.0]), direction=np.array([1.0, 0.0]), length=100.0,
                jump=np.array([0.0, 1.0]), width=2.0, taper=20.0, mode="opening")
    assert zone.amount(np.array([50.0, 90.0, 100.0, 150.0])).tolist() == [1.0, 0.5, 0.0, 0.0]


def test_random_scenes_are_reproducible_and_varied():
    a, b = random_scene(42), random_scene(42)
    assert a.zones and [z.jump.tolist() for z in a.zones] == [z.jump.tolist() for z in b.zones] or not a.zones
    kinds = {len(random_scene(s).zones) for s in range(40)}
    assert kinds >= {0, 1}


def test_step_in_movement_is_zero_for_smooth_fields_and_large_for_a_jump():
    rows, cols = np.mgrid[0:30, 0:30].astype(float)
    smooth = 0.01 * cols + 0.003 * rows + 0.2
    step = np.where(rows > 15, 1.0, 0.0)
    assert np.nanmax(features._plane_residual(smooth, 5)) < 1e-6     # rounding only; DIC resolves ~1e-3 px
    assert np.nanmax(features._plane_residual(step, 5)) > 0.25      # a plane fit leaves 0.28 of a unit step


def test_features_have_one_column_per_name_and_skip_unmeasured_points():
    rows, cols = np.mgrid[0:40, 0:50].astype(float)
    u, v = 0.01 * cols, -0.003 * rows
    u[:5, :] = np.nan
    v[:5, :] = np.nan
    X, have = features.compute(u, v, np.isfinite(u), 5.0)
    assert X.shape[1] == len(features.NAMES)
    assert not have[:5].any() and np.all(np.isfinite(X))


def test_zone_grouping_keeps_lines_and_drops_specks():
    p = np.zeros((40, 40))
    p[20, 5:35] = 0.9                               # a line: 30 steps long
    p[5:7, 5:7] = 0.9                               # a speck
    mask, found = zones(p)
    assert len(found) == 1 and found[0]["span_steps"] >= MIN_ZONE_SPAN_STEPS
    assert mask[20, 10] and not mask[5, 5]


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="no trained model (run scripts/ml_train.py)")
def test_model_finds_a_clear_crack_and_stays_quiet_on_intact_material():
    from strainlink.ml.measure import measure
    model = load()
    cracked = measure(_crack_scene(0.5))
    p = model.probability(cracked["u"], cracked["v"], cracked["valid"], cracked["step_px"])
    on_line = (cracked["truth"] == 1) & np.isfinite(p)
    far = (cracked["truth"] == 0) & np.isfinite(p)
    assert np.nanmean(p[on_line]) > 0.8 and np.nanmean(p[far] >= 0.5) < 0.02
    intact = measure(Scene(shape=(300, 400), background=QUIET, noise_grey=1.0, seed=5))
    q = model.probability(intact["u"], intact["v"], intact["valid"], intact["step_px"])
    assert not zones(q)[1]


def test_zones_too_faint_to_measure_are_not_scored():
    faint, clear = _crack_scene(0.01), _crack_scene(1.0)
    x, y = np.array([200.0]), np.array([150.0])
    # 0.01 px over a 31 px subset is ~3e-4 strain: below 3 x a noise of 1e-3, so "don't care".
    assert faint.truth(x, y, faint_below=1e-3)[0] == -1
    assert clear.truth(x, y, faint_below=1e-3)[0] == 1


def test_compressed_noisy_scenes_render_and_measure():
    from strainlink.ml.measure import measure
    scene = random_scene(7)
    scene.jpeg_quality, scene.noise_grey = 40, 8.0
    m = measure(scene)
    assert np.isfinite(m["u"]).mean() > 0.8 and m["noise"] > 0


class _AlwaysZoneOnColumn:
    """A stand-in estimator: says "zone" for every point (tests the plumbing, not the model)."""

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def test_model_output_is_an_add_on_beside_the_analysis(tmp_path):
    import json
    from strainlink.export import export
    from strainlink.fields import FieldStore
    from strainlink.ml.apply import run_on_analysis
    from strainlink.ml.model import ZoneModel
    results = tmp_path / "results"
    results.mkdir()
    x, z = np.meshgrid(np.arange(30.0), np.arange(20.0)[::-1])
    ux = np.stack([0 * x, 0.01 * x]); uz = np.stack([0 * z, -0.003 * z])
    np.savez_compressed(results / "fields.npz", x_mm=x, z_mm=z, ux=ux, uz=uz, push_mm=np.array([0.0, 1.0]))
    (results / "summary.json").write_text(json.dumps({"frame": {"mm_per_px": 0.2}, "frame_push_mm": [0.0, 1.0]}))
    before = FieldStore(results)
    assert not before.available("zone_probability") and np.isnan(before.get("zone_probability", 1)).all()
    out = run_on_analysis(tmp_path / "project.json", ZoneModel(_AlwaysZoneOnColumn(), {"model": "stand-in", "threshold": 0.5}))
    assert out["first_photo_with_zone"] == 2 and out["threshold"] == 0.5
    after = FieldStore(results)
    assert after.available("zone_probability") and np.nanmax(after.get("zone_probability", 1)) == 1.0
    # The analysis's own fields are untouched.
    for key in ("ux", "uz"):
        assert np.array_equal(before.data[key], after.data[key])
    import pandas as pd
    table = pd.read_csv(export(results, "csv") / "photo_0002.csv")
    assert "zone_probability" in table.columns
