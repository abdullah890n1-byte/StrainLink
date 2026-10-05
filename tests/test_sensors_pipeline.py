"""Sensor import, clock sync, mechanics and the correlation engine on synthetic logs (SYNTHETIC)."""

import numpy as np
import pandas as pd
import pytest

from strainlink import correlate, mechanics, sync
from strainlink.io.sensors import load_sensors
from strainlink.synth.sensors import generate, sensor_preset
from strainlink.synth.wall import WallExperiment, preset


@pytest.fixture(scope="module")
def dense():
    spec = sensor_preset("dense")
    return spec, generate(WallExperiment(preset("dense")), spec)


def test_messy_export_is_read_and_converted(tmp_path, dense):
    _, logs = dense
    frame = logs.logger.rename(columns={"time_s": "Time [s]", "actuator_force_kN": "Load (N)",
                                        "actuator_disp_mm": "Stroke [mm]", "PT5_kPa": "EPC5 [Pa]"})
    frame["Load (N)"] *= 1000
    frame["EPC5 [Pa]"] *= 1000
    path = tmp_path / "export.csv"
    with open(path, "w") as f:
        f.write("Logger TDS-540\nTest RW-DS\n\n")
        frame.to_csv(f, index=False, sep=";")
    table = load_sensors(path)
    roles = {c.role for c in table.channels}
    assert {"time", "force", "displacement", "pressure"} <= roles
    force = table.values(table.by_role("force")[0])
    assert np.nanmax(force) == pytest.approx(logs.logger.actuator_force_kN.max(), rel=1e-6)
    pt5 = next(c for c in table.channels if c.name == "PT5")
    assert np.nanmax(table.values(pt5)) == pytest.approx(logs.logger.PT5_kPa.max(), rel=1e-6)


def test_clock_offset_from_wall_push(dense):
    spec, logs = dense
    frames = logs.camera_times_s - logs.camera_times_s[0]
    dic_push = logs.frame_push_mm + np.random.default_rng(0).normal(0, 0.02, len(frames))
    timeline = sync.sync_by_dic(frames, dic_push, logs.logger.time_s.to_numpy(), logs.logger.actuator_disp_mm.to_numpy())
    true_offset = logs.camera_times_s[0] - spec.camera_clock_offset_s
    assert abs(timeline.clock_offset_s - true_offset) < 3 * max(timeline.offset_sigma_s, 0.1)
    assert np.max(np.abs(timeline.frame_push_mm - logs.frame_push_mm)) < 0.05


def _mech(spec, logs):
    L = logs.logger
    cells = {k: (logs.pt_depths_mm[k], L[k + "_kPa"].to_numpy()) for k in logs.pt_depths_mm}
    return mechanics.analyse(L.actuator_disp_mm.to_numpy(), L.actuator_force_kN.to_numpy(),
                             unit_weight_kn_m3=spec.unit_weight_kn_m3, wall_height_mm=390, wall_width_mm=700,
                             calibration=(logs.calibration.push_mm.to_numpy(), logs.calibration.force_kN.to_numpy()),
                             cells=cells, uplift_mm=L.LVDT_uplift_mm.to_numpy(), wall_weight_kn=spec.wall_weight_kn)


def test_mechanics_recovers_known_values(dense):
    spec, logs = dense
    m = _mech(spec, logs)
    assert m.passive.peak == pytest.approx(logs.truth["peak_force_kN"], rel=0.01)
    assert m.kp == pytest.approx(spec.kp, rel=0.01)
    assert m.passive.softens
    assert m.passive.push_range_mm[0] <= spec.delta_p_mm <= m.passive.push_range_mm[1]
    assert m.uplift_onset_mm == pytest.approx(logs.truth["uplift_start_mm"], abs=0.3)
    assert m.delta_mob_deg == pytest.approx(spec.delta_mob_deg, abs=0.5)
    assert abs(m.thrust_mismatch_fraction) < 0.3


def test_plateau_peak_position_range_contains_truth():
    spec = sensor_preset("loose")
    logs = generate(WallExperiment(preset("loose")), spec)
    m = _mech(spec, logs)
    assert m.passive.plateau
    assert m.passive.push_range_mm[0] <= spec.delta_p_mm <= m.passive.push_range_mm[1]


def test_cell_softening_hypothesis_is_found_when_planted():
    """Cells soften right after the tip reaches them: the engine must grade it strong."""
    levels = {"PT2": 273.0, "PT3": 195.0, "PT4": 117.0, "PT5": 39.0}
    reach = {"PT2": 13.3, "PT3": 12.1, "PT4": 10.8, "PT5": 9.1}
    tips = [correlate.Event(f"band-1.reaches.{k}", "", "soil", v, 0.5, k) for k, v in reach.items()]
    onsets = [correlate.Event(f"{k}.softening", "", "sensor", v + 0.5 + 0.05 * i, 0.25, k)
              for i, (k, v) in enumerate(reach.items())]
    report = correlate.analyse(onsets, [], tips, levels, pd.DataFrame({"push_mm": []}), [], [])
    finding = next(f for f in report.findings if f.key == "cells-vs-tip")
    assert finding.strength == "strong"
    assert finding.evidence["mean lag (mm of push)"] == pytest.approx(0.575, abs=0.01)


def test_cell_softening_hypothesis_rejected_when_unrelated():
    """All cells soften together, whatever the tip does: not a tip effect."""
    levels = {"PT2": 273.0, "PT3": 195.0, "PT4": 117.0, "PT5": 39.0}
    reach = {"PT2": 13.3, "PT3": 12.1, "PT4": 10.8, "PT5": 9.1}
    tips = [correlate.Event(f"band-1.reaches.{k}", "", "soil", v, 0.5, k) for k, v in reach.items()]
    onsets = [correlate.Event(f"{k}.softening", "", "sensor", 20.0 + 0.1 * i, 0.25, k) for i, k in enumerate(reach)]
    report = correlate.analyse(onsets, [], tips, levels, pd.DataFrame({"push_mm": []}), [], [])
    finding = next(f for f in report.findings if f.key == "cells-vs-tip")
    assert finding.strength in ("not supported", "weak")


def test_wall_push_is_read_at_the_wall_face_not_in_the_lagging_soil():
    """Soil a few cm from the wall lags it, more and more through the test; at the face it does not."""
    from strainlink import sync
    height = 400.0
    xs, zs = np.meshgrid(np.arange(10.0, 200.0, 5.0), np.arange(20.0, 380.0, 5.0))
    x, z = xs.ravel(), zs.ravel()
    pushes = np.linspace(0.0, 60.0, 31)
    # Lag: none at the wall, growing with distance (curved) and out of proportion to the push.
    lag = lambda push: (push / 60.0) ** 2 * (0.012 * x + 0.0004 * x**2)   # noqa: E731
    rng = np.random.default_rng(0)
    ux = np.array([p - lag(p) + rng.normal(0, 0.003, x.size) for p in pushes])
    found = sync.wall_push_from_dic(ux, x, z, height)
    assert np.max(np.abs(found - pushes)) < 0.01
    strip = np.array([np.median(u[(x <= 25.0) & (z >= 0.2 * height) & (z <= 0.8 * height)]) for u in ux])
    assert np.max(np.abs(strip - pushes)) > 0.2           # what the old measure would have read

    # With too few columns measured, it falls back to the strip (and does not fail).
    few = x <= 25.0
    assert np.all(np.isfinite(sync.wall_push_from_dic(ux[:, few], x[few], z[few], height)))
