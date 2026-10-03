"""The complete analysis: photos + sensor logs -> engineering results and findings.

A project is a small JSON file (written by the app's wizard) that points
to the photo folder and the logger file and records the scale, the wall
position and the test details. `run` executes every stage, reporting
progress, and writes the results next to the project:

    results/summary.json   every number, event and finding (what the app and report show)
    results/fields.npz     per-photo displacement and strain fields on the DIC grid
    results/bands.json     per-photo band centrelines for the overlays

Errors are raised as ProjectError with a message written for the user,
saying what is wrong and where to fix it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__, correlate, mechanics, sync
from .bands import DetectionSettings, detect_bands
from .bands.track import TestBands, track
from .calib import WallFrame
from .dic import DICSettings, Grid, load_grayscale, track_sequence
from .io.images import load_sequence
from .io.sensors import Channel, SensorTable, load_sensors, read_table
from .strain import compute_strain, fill_displacement_gaps

Progress = Callable[[str, float, str], None]

# DIC defaults for soil photos: 21 px subsets on a 5 px grid; photos are
# taken often enough that frame-to-frame motion stays within a few pixels.
DEFAULT_SUBSET_RADIUS_PX = 10
DEFAULT_STEP_PX = 5
DEFAULT_SEARCH_RADIUS_PX = 4
DEFAULT_STRAIN_WINDOW = 5
# Uncertainty of band-1's tip position used for timing its arrival at the
# cells (median tip error in the synthetic validation, see Validation.md).
TIP_POSITION_SIGMA_MM = 15.0
# Photo-to-logger matching looser than this (mm of push) is flagged for review.
PUSH_SYNC_TOLERANCE_MM = 0.25
STAGES = ("Loading", "Measuring soil movement (DIC)", "Finding shear bands", "Reading sensors",
          "Synchronising clocks", "Engineering results", "Correlation engine", "Saving results")


class ProjectError(Exception):
    """A problem the user can fix; the message says what and where."""


class LazyImages:
    """Photos loaded one at a time when needed.

    A test can have hundreds of photos; holding them all in memory (as
    float64) stalled an 8 GB machine in testing. DIC only ever needs the
    reference and the current photo.
    """

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.shape = load_grayscale(paths[0]).shape

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, k: int) -> np.ndarray:
        return load_grayscale(self.paths[k])


@dataclass
class RunSummary:
    project: str
    synthetic: bool
    version: str
    finished: str
    seconds: float
    frames: int
    frame_push_mm: list[float]
    sync: dict
    mechanics: dict
    bands: dict
    correlation: dict
    flags: list[str] = field(default_factory=list)


def lens_corrector(project: dict):
    """The lens correction of a project (calibration.lens, from a calibration-grid photo), or None."""
    lens = project.get("calibration", {}).get("lens")
    if not lens:
        return None
    from .lens import LensModel, Undistorter
    return Undistorter(LensModel.from_json(lens))


def corrected_images(project: dict, images):
    """Photos corrected for the lens when the project has a lens model (otherwise unchanged)."""
    undistort = lens_corrector(project)
    if undistort is None:
        return images
    from .lens import CorrectedImages
    try:
        undistort(images[0])
    except ValueError as error:
        raise ProjectError(str(error)) from error
    return CorrectedImages(images, undistort)


def load_project(path: str | Path) -> dict:
    path = Path(path)
    try:
        project = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectError(f"Could not read the project file {path.name}: {error}") from error
    project["_dir"] = str(path.parent)
    return project


def _resolve(project: dict, relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else Path(project["_dir"]) / p


def run(project_path: str | Path, progress: Progress | None = None) -> RunSummary | dict:
    """Analyse a project. Retaining-wall tests return a RunSummary; any other test (kind "general") its summary dict."""
    project = load_project(project_path)
    if project.get("kind") == "general":
        from .general import run as run_general
        return run_general(project_path, progress)
    return _run_wall(project_path, project, progress)


def _run_wall(project_path: str | Path, project: dict, progress: Progress | None) -> RunSummary:
    started = time.perf_counter()
    report = progress or (lambda stage, fraction, message: None)
    out_dir = Path(project["_dir"]) / "results"
    out_dir.mkdir(exist_ok=True)

    # 1. Photos and calibration.
    report(STAGES[0], 0.0, "Reading photos")
    cal = project.get("calibration", {})
    try:
        frame = WallFrame(mm_per_px=float(cal["mm_per_px"]), toe_col=float(cal["toe_px"][0]),
                          toe_row=float(cal["toe_px"][1]), wall_height_mm=float(cal["wall_height_mm"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectError("The scale and wall position are not set. Complete the Calibrate step.") from error
    images_cfg = project.get("images", {})
    try:
        sequence = load_sequence(_resolve(project, images_cfg.get("folder", "images")), images_cfg.get("interval_s"))
    except FileNotFoundError as error:
        raise ProjectError(f"No photos found: {error}. Choose the photo folder in the Photos step.") from error
    if len(sequence) < 2:
        raise ProjectError("At least two photos are needed (the first must show the soil before the push).")
    images = corrected_images(project, LazyImages(sequence.paths))

    # 2. DIC.
    dic_cfg = project.get("dic", {})
    # Each photo is correlated against the previous one (reference updated every
    # photo): inside a shear band the soil shears too much between distant
    # photos for subsets to match (they were lost at ~10 mm of push when the
    # first photo was kept as reference).
    settings = DICSettings(subset_radius=int(dic_cfg.get("subset_radius_px", DEFAULT_SUBSET_RADIUS_PX)),
                           step=int(dic_cfg.get("step_px", DEFAULT_STEP_PX)),
                           search_radius=int(dic_cfg.get("search_radius_px", DEFAULT_SEARCH_RADIUS_PX)),
                           reference_mode="incremental", update_every=int(dic_cfg.get("update_every", 1)))
    grid = Grid.over_roi(_soil_roi(project, frame, images.shape), settings.step, settings.subset_radius)
    if grid.x.size == 0:
        raise ProjectError("The measurement region is empty. Check the wall toe position and the region in Calibrate.")
    tracked = track_sequence(images, grid, settings,
                             progress=lambda k, n: report(STAGES[1], k / n, f"Photo {k + 1} of {n + 1}"))
    x_mm, z_mm = frame.to_wall(grid.x, grid.y)
    spacing = settings.step * frame.mm_per_px

    # 3. Strain and bands, photo by photo.
    window = int(dic_cfg.get("strain_window", DEFAULT_STRAIN_WINDOW))
    n = len(images)
    gamma = np.full((n,) + grid.shape, np.nan)
    ux_all = np.full((n,) + grid.shape, np.nan)
    uz_all = np.full((n,) + grid.shape, np.nan)
    per_frame = []
    det_settings = DetectionSettings(**project.get("detection", {}))
    for k in range(n):
        report(STAGES[2], k / n, f"Photo {k + 1} of {n}")
        ux, uz = frame.displacement_to_wall(tracked.field(k, "u"), tracked.field(k, "v"))
        ux_all[k], uz_all[k] = ux, uz
        if k == 0:
            per_frame.append(detect_bands(np.zeros(grid.shape), x_mm, z_mm, frame.wall_height_mm, det_settings))
            gamma[k] = 0.0
            continue
        # Points tracked before but lost now (correlation lost inside an active band)
        # are refilled from their neighbours' displacement, so the band isn't cut there.
        lost = (tracked.valid[1:k + 1].any(axis=0) & ~tracked.valid[k]).reshape(grid.shape)
        ux_f, uz_f, refilled = fill_displacement_gaps(ux, uz, lost)
        strain = compute_strain(ux_f, uz_f, spacing, -spacing, window)
        gamma[k] = strain.gamma_max
        per_frame.append(detect_bands(strain.gamma_max, x_mm, z_mm, frame.wall_height_mm, det_settings, uz=uz_f,
                                      reconstructed=refilled))

    # 4. Sensors.
    report(STAGES[3], 0.0, "Reading the logger file")
    table, cal_run = _load_sensor_files(project)
    time_s = table.time_s
    push_ch, force_ch = table.by_role("displacement"), table.by_role("force")
    if time_s is None:
        raise ProjectError(f"No time column found in {table.path.name}. Assign it in the Sensors step.")
    if not push_ch or not force_ch:
        missing = "wall displacement (actuator/LVDT)" if not push_ch else "force"
        raise ProjectError(f"No {missing} column found in {table.path.name}. Assign it in the Sensors step.")
    push = table.values(push_ch[0])
    force = table.values(force_ch[0])

    # 5. Clock synchronisation.
    report(STAGES[4], 0.0, "Matching photos to the logger clock")
    timeline = _synchronise(project, sequence.times_s, tracked, x_mm, z_mm, frame, time_s, push)

    # Band tracks against the synchronised wall push.
    history = track(per_frame, timeline.frame_push_mm)

    # 6. Mechanics.
    report(STAGES[5], 0.0, "Peak force, Kp, pressure cells")
    test = project.get("test", {})
    cells = _cells(table, test)
    uplift = table.by_role("uplift")
    try:
        mech = mechanics.analyse(push, force, unit_weight_kn_m3=float(test["unit_weight_kn_m3"]),
                                 wall_height_mm=frame.wall_height_mm, wall_width_mm=float(test["wall_width_mm"]),
                                 calibration=cal_run, cells=cells,
                                 uplift_mm=table.values(uplift[0]) if uplift else None,
                                 wall_weight_kn=test.get("wall_weight_kn"))
    except KeyError as error:
        raise ProjectError(f"Test detail missing: {error.args[0]}. Complete the Test details step.") from error

    # 7. Correlation.
    report(STAGES[6], 0.0, "Finding and testing relationships")
    corr, frames_table = _correlate(mech, history, timeline, table, time_s, cells, frame.wall_height_mm, push)

    # 8. Save.
    report(STAGES[7], 0.0, "Writing results")
    np.savez_compressed(out_dir / "fields.npz", x_mm=x_mm, z_mm=z_mm, gamma=gamma, ux=ux_all, uz=uz_all,
                        valid=tracked.valid.reshape((n,) + grid.shape), push_mm=timeline.frame_push_mm)
    (out_dir / "bands.json").write_text(json.dumps(_bands_json(per_frame), default=_json_default))
    frames_table.to_csv(out_dir / "per_photo.csv", index=False)
    summary = RunSummary(
        project=project.get("name", Path(project_path).stem), synthetic=bool(project.get("synthetic", False)),
        version=__version__, finished=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.perf_counter() - started,
        frames=n, frame_push_mm=timeline.frame_push_mm.tolist(),
        sync={"source": timeline.source, "clock_offset_s": timeline.clock_offset_s, "offset_sigma_s": timeline.offset_sigma_s,
              "soil_to_wall_ratio": timeline.soil_to_wall_ratio, "push_sigma_mm": timeline.push_sigma_mm,
              "photo_time_source": sequence.time_source},
        mechanics=_mechanics_json(mech),
        bands=_history_json(history),
        correlation={"findings": [f.as_dict() for f in corr.findings],
                     "events": [asdict(e) for e in corr.events], "warnings": corr.warnings},
        flags=_flags(history, mech, timeline, sequence.time_source),
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=1, default=_json_default))
    report(STAGES[7], 1.0, "Done")
    return summary


# --- stages ------------------------------------------------------------------

def _soil_roi(project: dict, frame: WallFrame, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Measurement region (px): the soil from the wall face to the image edge, surface to bottom."""
    if "roi_px" in project.get("calibration", {}):
        return tuple(int(v) for v in project["calibration"]["roi_px"])   # type: ignore[return-value]
    surface_row = frame.toe_row - frame.wall_height_mm / frame.mm_per_px
    return (int(math.ceil(frame.toe_col)), int(max(surface_row, 0)), shape[1], shape[0])


def _load_sensor_files(project: dict) -> tuple[SensorTable, tuple[np.ndarray, np.ndarray] | None]:
    cfg = project.get("sensors", {})
    if "file" not in cfg:
        raise ProjectError("No sensor file chosen. Choose the logger export in the Sensors step.")
    path = _resolve(project, cfg["file"])
    try:
        table = load_sensors(path)
    except (OSError, ValueError) as error:
        raise ProjectError(f"Could not read {path.name}: {error}") from error
    if "channels" in cfg:      # the user's confirmed mapping overrides the automatic guess
        table.channels = [Channel(**c) for c in cfg["channels"]]
    cal_run = None
    if cfg.get("calibration_file"):
        cal_frame = read_table(_resolve(project, cfg["calibration_file"]))
        if cal_frame.shape[1] < 2:
            raise ProjectError("The calibration file needs two columns: push (mm) and force (kN).")
        cal_run = (cal_frame.iloc[:, 0].to_numpy(float), cal_frame.iloc[:, 1].to_numpy(float))
    return table, cal_run


def _synchronise(project, frame_times, tracked, x_mm, z_mm, frame, time_s, push) -> sync.Timeline:
    cfg = project.get("sync", {"mode": "dic"})
    mode = cfg.get("mode", "dic")
    if mode == "offset":
        return sync.sync_by_offset(frame_times, float(cfg["offset_s"]), time_s, push)
    if mode == "push-step":
        return sync.sync_by_push_step(len(frame_times), float(cfg["push_step_mm"]), time_s, push)
    ux_mm = tracked.u * frame.mm_per_px
    try:
        wall_push = sync.wall_push_from_dic(ux_mm, x_mm.ravel(), z_mm.ravel(), frame.wall_height_mm, tracked.valid)
        return sync.sync_by_dic(frame_times, wall_push, time_s, push)
    except ValueError as error:
        raise ProjectError(f"Automatic clock matching failed: {error}. Enter the clock offset or the push per "
                           f"photo in the Sync step.") from error


def _cells(table: SensorTable, test: dict) -> dict[str, tuple[float, np.ndarray]]:
    depths = test.get("cell_depths_mm", {})
    cells = {}
    for channel in table.by_role("pressure"):
        if channel.name in depths:
            cells[channel.name] = (float(depths[channel.name]), table.values(channel))
    return cells


def _correlate(mech, history: TestBands, timeline, table, time_s, cells, wall_height, push):
    logger_step = float(np.nanmedian(np.diff(push[push > 0]))) if (push > 0).sum() > 2 else 0.01
    s_events = correlate.sensor_events(mech, max(logger_step, 1e-3))
    d_events = correlate.soil_events(history)
    levels = {name: wall_height - depth for name, (depth, _) in cells.items()}
    t_events = correlate.tip_reach_events(history, levels, TIP_POSITION_SIGMA_MM)
    frames = pd.DataFrame({"push_mm": timeline.frame_push_mm})
    frames["force"] = timeline.at_frames(time_s, mech.passive_kn)
    for name, (_, values) in cells.items():
        frames[name] = timeline.at_frames(time_s, values)
    frames["band-1 length"] = history.primary.developed_length_mm
    frames["band-1 tip height"] = history.primary.tip_z_mm
    frames["heave"] = history.primary.heave_mm
    sensor_cols = ["force"] + list(cells)
    soil_cols = ["band-1 length", "band-1 tip height", "heave"]
    return correlate.analyse(s_events, d_events, t_events, levels, frames, sensor_cols, soil_cols), frames


# --- serialisation -------------------------------------------------------------

def _json_default(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_default(v) if isinstance(v, (float, np.floating)) else v for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not serialisable: {type(value)}")


def _clean(value):
    """NaN/inf -> None recursively (JSON has no NaN)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def _mechanics_json(mech: mechanics.MechanicsResults) -> dict:
    return _clean({
        "push_mm": mech.push_mm[::max(len(mech.push_mm) // 2000, 1)],
        "passive_kn": mech.passive_kn[::max(len(mech.push_mm) // 2000, 1)],
        "passive": asdict(mech.passive), "kp": mech.kp, "kp_sigma": mech.kp_sigma,
        "cells": [asdict(c) for c in mech.cells],
        "thrust_at_rest_kn": mech.thrust_at_rest_kn, "thrust_from_cells_at_peak_kn": mech.thrust_from_cells_at_peak_kn,
        "thrust_mismatch_fraction": mech.thrust_mismatch_fraction, "uplift_onset_mm": mech.uplift_onset_mm,
        "delta_mob_deg": mech.delta_mob_deg, "notes": mech.notes,
    })


def _history_json(history: TestBands) -> dict:
    p = history.primary
    primary = {k: v for k, v in asdict(p).items()}
    primary.update(onset_mm=p.onset_mm, surface_reached_mm=p.surface_reached_mm,
                   surface_reached_range_mm=list(p.surface_reached_range_mm))
    tracks = [{"name": t.name, "label": t.label, "onset_mm": t.onset_mm, "push_mm": t.delta_mm,
               "length_mm": [b.length_mm for b in t.bands], "width_mm": [b.width_mm for b in t.bands]}
              for t in history.tracks]
    return _clean({"primary": primary, "tracks": tracks})


def _bands_json(per_frame) -> list:
    return _clean([[{"name": b.name, "label": b.label, "points": b.points.round(2), "width_mm": b.width_mm,
                     "flags": b.flags} for b in fb.bands] for fb in per_frame])


def _flags(history: TestBands, mech, timeline, time_source: str) -> list[str]:
    flags = []
    if time_source == "file-time":
        flags.append("Photo times come from file modification times (least reliable). Prefer camera EXIF times "
                     "or enter the photo interval.")
    if timeline.source == "dic" and not timeline.push_sigma_mm <= PUSH_SYNC_TOLERANCE_MM:
        flags.append(f"Photos are matched to the logger only within ±{timeline.push_sigma_mm:.2f} mm of wall push "
                     f"(±{timeline.offset_sigma_s:.1f} s). Photos taken more often, or a known clock offset "
                     f"(Sync step), would tighten this.")
    review_frames = int(np.sum(history.primary.needs_review))
    if review_frames:
        flags.append(f"{review_frames} photo(s) have band measurements flagged for review.")
    flags.extend(mech.notes)
    if not np.isfinite(history.primary.onset_mm):
        flags.append("No primary slip surface (band-1) was detected in this test.")
    return flags
