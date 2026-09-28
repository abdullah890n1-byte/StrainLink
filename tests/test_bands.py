"""Band detection on synthetic retaining-wall experiments with known bands (SYNTHETIC)."""

import numpy as np
import pytest

from strainlink.bands import classify, detect_bands
from strainlink.strain import compute_strain
from strainlink.synth.wall import WallExperiment, preset, primary_curve, with_intermediate_bands
from strainlink.validation.bands import mean_distance

SPACING = 5.0
X, Z = np.meshgrid(np.arange(0, 900 + 1e-9, SPACING), np.arange(390, -150 - 1e-9, -SPACING))


def _detect(experiment, delta, noise=0.0, seed=0):
    ux, uz = experiment.displacement(X, Z, delta)
    if noise:
        rng = np.random.default_rng(seed)
        ux = ux + rng.normal(0, noise, ux.shape)
        uz = uz + rng.normal(0, noise, uz.shape)
    gamma = compute_strain(ux, uz, SPACING, -SPACING, 5).gamma_max
    return detect_bands(gamma, X, Z, experiment.spec.wall_height_mm, uz=uz), experiment.truth(delta)


@pytest.mark.parametrize("name,delta", [("dense", 20), ("medium", 30), ("loose", 60)])
def test_fully_developed_primary_band_is_measured(name, delta):
    found, truth = _detect(WallExperiment(preset(name)), delta, noise=0.05)
    primary = found.by_label("primary")
    assert len(primary) == 1
    assert mean_distance(primary[0].points, truth.bands[0].centreline) < 3.0
    assert primary[0].reached_end
    assert found.primary.lift_off_deg == pytest.approx(truth.lift_off_span_deg, abs=3.0)
    assert found.primary.length_mm == pytest.approx(truth.length_mm, abs=20.0)
    assert found.primary.area_mm2 == pytest.approx(truth.area_mm2, rel=0.05)


def test_radial_band_is_labelled():
    found, _ = _detect(WallExperiment(preset("medium")), 30, noise=0.05)
    assert [b.label for b in found.bands].count("radial") == 1


def test_intermediate_bands_are_found():
    """The Log-Spiral-Hyperbolic question: intermediate bands must be detected when present."""
    found, _ = _detect(WallExperiment(with_intermediate_bands(preset("medium"))), 12, noise=0.05)
    assert len(found.by_label("intermediate")) == 2


def test_no_intermediate_bands_under_translation():
    found, _ = _detect(WallExperiment(preset("medium")), 12, noise=0.05)
    assert found.by_label("intermediate") == []


def test_partial_band_is_flagged_not_reached():
    found, _ = _detect(WallExperiment(preset("loose")), 40, noise=0.05)
    assert "not-reached-surface" in found.primary.flags
    assert np.isnan(found.primary.length_mm)


def test_nothing_detected_before_localisation():
    found, _ = _detect(WallExperiment(preset("medium")), 5.0, noise=0.05)
    assert found.by_label("primary") == []


def test_shape_classifier_on_exact_curves():
    rng = np.random.default_rng(1)
    straight = primary_curve(33.5, 33.5, 1.0, 43.8, 390.0).points[::10]
    result = classify(straight + rng.normal(0, 0.3, straight.shape), 20.0)
    assert result.label == "linear"
    medium = WallExperiment(preset("medium"))
    composite = medium.primary(30).points[::10]
    result = classify(composite + rng.normal(0, 0.3, composite.shape), 20.0)
    assert result.label == "composite" and not result.ambiguous


def test_tracking_gives_onset_and_surface_reach():
    from strainlink.bands.track import track
    experiment = WallExperiment(preset("medium"))
    deltas = [5, 7.5, 10, 12.5, 15, 20, 25, 30]
    frames = [_detect(experiment, d, noise=0.05, seed=i)[0] for i, d in enumerate(deltas)]
    history = track(frames, deltas)
    # band-1 is >= 50 mm long from delta 7.5 mm (medium preset); reaches the surface at 15 mm.
    assert history.primary.onset_mm in (7.5, 10.0)
    assert history.primary.surface_reached_mm == 15.0
    radial = history.by_label("radial")
    assert len(radial) == 1 and radial[0].onset_mm >= 15.0
