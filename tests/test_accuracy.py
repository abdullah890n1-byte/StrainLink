"""Stated accuracies must be honest: checked against exact answers (SYNTHETIC)."""

import json

import numpy as np
import pytest

from strainlink import accuracy
from strainlink.fields import FieldStore, accuracy_statement
from strainlink.general import run
from strainlink.pipeline import accuracy_flags
from strainlink.synth import cases as C

STRAIN, POISSON = 0.01, 0.3
SHAPE = (260, 380)


def test_scatter_measures_the_noise_not_the_trend():
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[0:60, 0:80].astype(float)
    trend = 0.002 + 1e-5 * cols + 2e-5 * rows                  # a strain that varies smoothly
    noisy = trend + rng.normal(0, 1e-4, trend.shape)
    found = accuracy.scatter(noisy)
    assert found.typical == pytest.approx(1e-4, rel=0.15)
    assert accuracy.scatter(trend).typical < 2e-6               # no noise: nothing to report


def test_relative_accuracy_wording():
    assert accuracy.describe(0.01, 0.0001) == "within 1.0% of the value"
    assert accuracy.describe(1e-5, 1e-4) == "smaller than the measurement noise"
    assert accuracy.relative(0.0, 1e-4) is None


def test_average_is_more_accurate_than_a_point_but_keeps_a_systematic_part():
    point = accuracy.COVERAGE_FACTOR * 1e-4
    whole = accuracy.average_uncertainty(0.01, 1e-4, points=8000)
    assert whole < point / 5
    assert whole >= np.hypot(accuracy.BIAS_FLOOR_STRAIN, accuracy.BIAS_FRACTION_LOSSLESS * 0.01)
    assert accuracy.average_uncertainty(0.01, 1e-4, 8000, compressed=True) > 3 * whole


@pytest.fixture(scope="module")
def tension(tmp_path_factory):
    folder = tmp_path_factory.mktemp("accuracy")
    centre = ((SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2)
    case = C.Case("tension", "Tension 1%", C.uniaxial(STRAIN, POISSON, centre), shape=SHAPE, photos=4)
    summary = run(C.render_case(case, folder))
    return case, folder, summary


def test_stated_accuracy_contains_the_truth(tension):
    case, folder, summary = tension
    f = np.load(folder / "results" / "fields.npz")
    k = len(f["exx"]) - 1
    true = {"exx": 0.5 * ((1 + STRAIN) ** 2 - 1), "ezz": 0.5 * ((1 - POISSON * STRAIN) ** 2 - 1), "exz": 0.0}
    store = FieldStore(folder / "results")
    for key, value in true.items():
        field = f[key][k]
        known = np.isfinite(field)
        inside = np.abs(field - value)[known] <= store.uncertainty(key, k)[known]
        assert inside.mean() >= 0.95, key
    statement = summary["accuracy"]
    for key in ("exx", "ezz"):
        whole = statement["whole_object"][key]
        assert abs(whole["value"] - true[key]) <= whole["u95"]
        assert whole["relative"] < 0.02                      # and it is a useful statement, not a huge one
    assert statement["point"]["strain"] < 5e-4


def test_displacement_accuracy_contains_the_truth(tension):
    case, folder, summary = tension
    f = np.load(folder / "results" / "fields.npz")
    k = len(f["ux"]) - 1
    mm = case.mm_per_px
    X = f["x_mm"] / mm
    Y = summary["frame"]["height_px"] - f["z_mm"] / mm
    tu, tv = case.field(X, Y, 1.0)
    store = FieldStore(folder / "results")
    for key, found, true in (("ux", f["ux"][k] / mm, tu), ("uz", -f["uz"][k] / mm, tv)):
        known = np.isfinite(found)
        stated_px = store.uncertainty(key, k)[known] / mm
        assert (np.abs(found - true)[known] <= stated_px).mean() >= 0.95, key
        assert np.median(stated_px) < 0.02


def test_flag_when_the_strain_is_below_the_noise():
    statement = {"point": {"strain": 2e-4}, "peak_strain": 1e-4, "peak_relative": None}
    assert "smaller than the accuracy at a point" in accuracy_flags(statement)[0]
    assert accuracy_flags({"point": {"strain": 2e-4}, "peak_strain": 0.02, "peak_relative": 0.01}) == []


def test_statement_is_saved_and_survives_json(tension):
    _, folder, _ = tension
    saved = json.loads((folder / "results" / "summary.json").read_text())["accuracy"]
    assert saved["confidence"] == "95%" and saved["not_included"]
    again = accuracy_statement(FieldStore(folder / "results"), False, 0.05)
    assert again["point"]["strain"] == pytest.approx(saved["point"]["strain"])
