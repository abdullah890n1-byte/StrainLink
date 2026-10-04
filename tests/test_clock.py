"""Matching the photos to the logger clock in the any-object mode (SYNTHETIC signals)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from strainlink.general import run
from strainlink.general.pipeline import MIN_SYNC_R2, _fit_offset
from strainlink.synth.general import CLOCK_OFFSET_S, GeneralSpec, make_general_test

TRUE_OFFSET_S = 12.5
PHOTO_T = np.arange(0.0, 60.0, 2.0)              # 30 photos, 2 s apart
LOGGER_T = np.arange(0.0, 80.0, 0.1)
ONSET_S = 30.0                                   # the zone forms (photo time)


def _load(photo_time: np.ndarray) -> np.ndarray:
    """Rises with the loading, drops by 30% when the zone forms, then holds."""
    return np.where(photo_time < ONSET_S, np.clip(photo_time, 0.0, None), 0.7 * ONSET_S)


def test_steady_ramp_cannot_fix_the_clock_and_says_so():
    rng = np.random.default_rng(0)
    stretch = 0.01 * PHOTO_T + rng.normal(0, 1e-4, PHOTO_T.size)
    crosshead = 0.0104 * (LOGGER_T - 5.0)            # already moving, at a steady rate throughout
    fit = _fit_offset(PHOTO_T, stretch, LOGGER_T, crosshead)
    assert fit["open_ended"]
    assert fit["sigma"] > 2.0                        # several photo intervals: not a usable match


def test_load_drop_fixes_the_clock():
    rng = np.random.default_rng(1)
    strain = 1e-4 * _load(PHOTO_T) + rng.normal(0, 2e-6, PHOTO_T.size)
    load = 3.0 * _load(LOGGER_T - TRUE_OFFSET_S) + rng.normal(0, 0.02, LOGGER_T.size)
    fit = _fit_offset(PHOTO_T, strain, LOGGER_T, load)
    assert abs(fit["offset"] - TRUE_OFFSET_S) < 0.25
    assert fit["sigma"] < 0.5 and not fit["open_ended"]
    assert fit["r2"] > 0.99


def test_unrelated_channel_is_not_accepted_for_matching():
    rng = np.random.default_rng(2)
    strain = 1e-4 * _load(PHOTO_T)
    unrelated = np.cumsum(rng.normal(0, 1.0, LOGGER_T.size))      # a wandering channel
    assert _fit_offset(PHOTO_T, strain, LOGGER_T, unrelated)["r2"] < MIN_SYNC_R2


def test_a_half_without_timing_information_does_not_inflate_the_uncertainty():
    # The first half of the test is a pure ramp (it cannot fix the clock by itself); the drop is
    # in the second half. The halves must not be counted as disagreeing.
    strain = 1e-4 * _load(PHOTO_T)
    load = 3.0 * _load(LOGGER_T - TRUE_OFFSET_S)
    fit = _fit_offset(PHOTO_T, strain, LOGGER_T, load)
    assert fit["sigma"] < 0.5


# --- through the whole any-object pipeline (a small cracking plate, steady loading rate) ---

SMALL = GeneralSpec(mode="opening", zone_angle_deg=70.0, image_px=(240, 360), plate_mm=(30.0, 18.0),
                    photos=16, onset_photo=8, stretch_rate_mm_s=0.012, photo_interval_s=1.0)


@pytest.fixture(scope="module")
def with_load(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clock_load")
    return run(make_general_test(folder, SMALL)), folder


def test_pipeline_matches_the_clock_from_the_load(with_load):
    summary, _ = with_load
    sync = summary["sync"]
    assert sync["matched_channel"].startswith("Load")
    assert abs(sync["clock_offset_s"] - CLOCK_OFFSET_S) <= 0.25 * SMALL.photo_interval_s
    assert not any("matched to the logger" in flag for flag in summary["flags"])
    caveats = [c for f in summary["correlation"]["findings"] if f["title"].startswith("Load") for c in f["caveats"]]
    assert any("matched to the logger clock" in c for c in caveats)


def test_pipeline_flags_the_clock_when_only_a_steady_ramp_is_logged(with_load, tmp_path):
    _, source = with_load
    folder = tmp_path / "no_load"
    shutil.copytree(source, folder, ignore=shutil.ignore_patterns("results"))
    logger = pd.read_csv(folder / "logger.csv").drop(columns=["Load (kN)"])
    logger.to_csv(folder / "logger.csv", index=False)
    summary = run(folder / "project.json")
    assert summary["sync"]["matched_channel"].startswith("Crosshead")
    assert any("matched to the logger" in flag for flag in summary["flags"])
    # The stated range must contain the truth even though the estimate cannot be sharp.
    assert abs(summary["sync"]["clock_offset_s"] - CLOCK_OFFSET_S) <= summary["sync"]["offset_sigma_s"] + SMALL.photo_interval_s
