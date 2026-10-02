"""Any-object mode: zone kinematics, sensor events, honest screening and a small end-to-end run (SYNTHETIC)."""

import json

import numpy as np
import pandas as pd
import pytest

from strainlink.bands.detect import DetectedBand
from strainlink.correlate import screen
from strainlink.general import run
from strainlink.general.events import channel_events
from strainlink.general.zones import kinematics
from strainlink.synth.general import GeneralSpec, make_general_test
from strainlink.validation.known_answers import compare


def _band_field(jump: np.ndarray):
    """A vertical zone at x = 50 mm with a known displacement jump across it (+ uniform strain)."""
    x, z = np.meshgrid(np.arange(0, 100.1, 1.0), np.arange(60, -0.1, -1.0))
    step = 0.5 * (1 + np.tanh((x - 50) / 1.5))
    ux = 0.002 * x + jump[0] * step
    uz = -0.0006 * z + jump[1] * step
    band = DetectedBand(name="Z1", label="zone", points=np.column_stack([np.full(40, 50.0), np.linspace(10, 50, 40)]),
                        width_mm=3.0, width_profile_mm=np.full(40, 3.0), peak_gamma=0.1, snr=50, reached_end=False)
    return band, x, z, ux, uz


@pytest.mark.parametrize("jump, mode", [((0.5, 0.0), "opening"), ((0.0, 0.5), "shear"), ((0.3, 0.3), "mixed")])
def test_zone_mode_from_the_jump_across_it(jump, mode):
    band, x, z, ux, uz = _band_field(np.array(jump))
    k = kinematics(band, x, z, ux, uz)
    assert k.mode == mode
    assert k.angle_deg == pytest.approx(90.0, abs=1.0)
    if mode == "opening":
        assert k.opening_mm == pytest.approx(0.5, rel=0.05)      # the background strain is removed
    if mode == "shear":
        assert k.slip_mm == pytest.approx(0.5, rel=0.05)


def test_sudden_drop_found_and_random_walk_ignored():
    rng = np.random.default_rng(0)
    axis = np.linspace(0, 1, 2000)
    load = np.where(axis < 0.6, 10 * axis, 4.2 + 2 * (axis - 0.6)) + rng.normal(0, 0.01, axis.size)
    keys = {e.key for e in channel_events("Load", axis, load, 0.001)}
    assert {"Load.peak", "Load.drop"} <= keys
    drop = next(e for e in channel_events("Load", axis, load, 0.001) if e.key == "Load.drop")
    assert drop.push_mm == pytest.approx(0.6, abs=0.01)
    walk = 21 + np.cumsum(rng.normal(0, 0.01, axis.size))
    assert channel_events("Temperature", axis, walk, 0.001) == []


def test_screening_cannot_claim_significance_from_too_few_photos():
    # 12 photos: only ~6 distinct circular shifts exist, so p can never go below 0.05.
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(size=12))
    frames = pd.DataFrame({"sensor": a, "object": a + rng.normal(0, 0.01, 12)})
    assert screen(frames, ["sensor"], ["object"]) == []


@pytest.fixture(scope="module")
def small_crack_test(tmp_path_factory):
    folder = tmp_path_factory.mktemp("general")
    spec = GeneralSpec(mode="opening", zone_angle_deg=70.0, image_px=(240, 360), plate_mm=(30.0, 18.0),
                       photos=16, onset_photo=8, stretch_rate_mm_s=0.012, photo_interval_s=1.0)
    project = make_general_test(folder, spec)
    run(project)
    return folder


def test_small_crack_test_end_to_end(small_crack_test):
    summary = json.loads((small_crack_test / "results" / "summary.json").read_text())
    assert summary["kind"] == "general"
    zones = summary["zones"]
    assert zones and zones[0]["mode"] == "opening"
    result = compare(small_crack_test)
    checks = {c["item"]: c for c in result["checks"]}
    assert checks["Type of zone"]["ok"]
    assert checks["Direction of the zone"]["ok"]
    assert checks["Zone appears at"]["ok"]
    assert all(c["ok"] for item, c in checks.items() if item.startswith("Strain"))


def test_object_findings_without_a_logger():
    from strainlink.correlate.events import Event
    from strainlink.general.findings import formation_order, zone_growth
    axis = np.arange(20.0)
    opening = np.where(axis >= 10, 0.02 * (axis - 9), np.nan)
    growth = zone_growth([{"name": "Z1", "mode": "opening", "onset": 10.0}], {"Z1 opening (mm)": opening}, axis, ("time", "s"))
    assert growth[0].strength == "strong" and "keeps opening" in growth[0].statement
    together = formation_order([Event("Z1.onset", "Z1 (crack) appears", "object", 10.0, 0.5, "Z1"),
                                Event("Z2.onset", "Z2 (crack) appears", "object", 10.2, 0.5, "Z2")], ("time", "s"))
    assert "same time" in together[0].statement
    ordered = formation_order([Event("Z1.onset", "Z1 (crack) appears", "object", 10.0, 0.5, "Z1"),
                               Event("Z2.onset", "Z2 (shear band) appears", "object", 16.0, 0.5, "Z2")], ("time", "s"))
    assert ordered[0].strength == "strong" and "forms first" in ordered[0].statement
