"""Complete synthetic test folders: photos, logger CSV, calibration run, project file (SYNTHETIC).

What a real test leaves on disk, generated with known ground truth:

    <folder>/
      images/SYNTH_0000.tif ...   photos of the soil through the glass (8-bit)
      logger.csv                  data logger export (time, force, push, LVDTs, PT1..5)
      calibration.csv             pushover run without soil (base friction)
      project.json                everything StrainLink needs to run
      truth.json                  ground truth, for validation only (the app never reads it)

The photo shows the wall (plain grey) on the left, the speckled soil, and
air (dark) above the surface. As the wall advances its face moves into the
picture and the surface heaves, exactly as in a real rig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
from scipy import ndimage

from ..calib import WallFrame
from .sensors import SensorSpec, generate, sensor_preset
from .speckle import SpeckleTexture, _pull_back
from .wall import WallExperiment, preset

WALL_GREY = 150.0
AIR_GREY = 25.0
# Picture extent around the soil (mm): wall strip on the left, air above.
WALL_STRIP_MM = 60.0
AIR_STRIP_MM = 60.0
SOIL_EXTENT_X_MM = 900.0
SOIL_DEPTH_BELOW_TOE_MM = 150.0
# Sub-samples per pixel side for sensor integration. The texture itself is
# defined at 4x resolution; 2x2 integration is ample for 8-bit photos and
# makes the demo four times faster to generate.
PIXEL_SAMPLES = 2
# Spacing (px) of the grid on which the displacement model is evaluated per photo.
MODEL_GRID_PX = 2


@dataclass(frozen=True)
class DemoSpec:
    preset: str = "medium"
    mm_per_px: float = 1.0
    push_step_mm: float = 1.0          # one photo per this much wall push
    image_noise_grey: float = 1.5      # camera noise (grey levels)
    grain_radius_px: float = 2.0
    seed: int = 11


def _frame(spec: DemoSpec, wall_height: float) -> tuple[WallFrame, tuple[int, int]]:
    width = int(round((WALL_STRIP_MM + SOIL_EXTENT_X_MM) / spec.mm_per_px))
    height = int(round((AIR_STRIP_MM + wall_height + SOIL_DEPTH_BELOW_TOE_MM) / spec.mm_per_px))
    frame = WallFrame(mm_per_px=spec.mm_per_px, toe_col=WALL_STRIP_MM / spec.mm_per_px,
                      toe_row=(AIR_STRIP_MM + wall_height) / spec.mm_per_px, wall_height_mm=wall_height)
    return frame, (height, width)


def render_photo(texture: SpeckleTexture, experiment: WallExperiment, frame: WallFrame, shape: tuple[int, int],
                 push_mm: float, noise: float, rng: np.random.Generator, samples: int = PIXEL_SAMPLES) -> np.ndarray:
    """One photo: soil deformed by the experiment's displacement field at this push."""
    height, width = shape
    # The model is evaluated on a coarse grid (bands are ~20 px wide, so a
    # 2 px grid resolves them) and interpolated during the fixed-point solve.
    rows, cols = np.mgrid[0:height + MODEL_GRID_PX:MODEL_GRID_PX, 0:width + MODEL_GRID_PX:MODEL_GRID_PX].astype(float)
    x, z = frame.to_wall(cols, rows)
    ux, uz = experiment.displacement(x, z, push_mm) if push_mm > 0 else (np.zeros_like(x), np.zeros_like(z))
    u_px, v_px = ux / frame.mm_per_px, -uz / frame.mm_per_px

    def displacement(cx: np.ndarray, ry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = np.stack([ry.ravel() / MODEL_GRID_PX, cx.ravel() / MODEL_GRID_PX])
        return (ndimage.map_coordinates(u_px, coords, order=1, mode="nearest").reshape(cx.shape),
                ndimage.map_coordinates(v_px, coords, order=1, mode="nearest").reshape(cx.shape))

    sub = (np.arange(samples) + 0.5) / samples - 0.5
    sy = (np.arange(height)[:, None] + sub[None, :]).ravel()
    sx = (np.arange(width)[:, None] + sub[None, :]).ravel()
    gx, gy = np.meshgrid(sx, sy)
    ref_x, ref_y = _pull_back(gx, gy, displacement)
    ref_wall_x, ref_wall_z = frame.to_wall(ref_x, ref_y)
    soil = (ref_wall_x >= 0) & (ref_wall_z <= frame.wall_height_mm)
    face_x, _ = frame.to_wall(gx, gy)
    wall = ~soil & (face_x < push_mm)
    value = np.where(soil, 40.0 + 170.0 * texture.evaluate(ref_x, ref_y), np.where(wall, WALL_GREY, AIR_GREY))
    image = value.reshape(height, samples, width, samples).mean(axis=(1, 3))
    image = image + rng.normal(0.0, noise, image.shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def make_demo(folder: str | Path, spec: DemoSpec | None = None,
              progress: Callable[[float, str], None] | None = None) -> Path:
    """Write a complete synthetic test folder and return the project file path."""
    spec = spec or DemoSpec()
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    experiment = WallExperiment(preset(spec.preset))
    sensors: SensorSpec = sensor_preset(spec.preset)
    sensors = SensorSpec(**{**sensors.__dict__, "camera_interval_s": spec.push_step_mm / sensors.push_rate_mm_s})
    logs = generate(experiment, sensors)
    wall_height = experiment.spec.wall_height_mm
    frame, shape = _frame(spec, wall_height)
    texture = SpeckleTexture.granular(shape, grain_radius_px=spec.grain_radius_px, seed=spec.seed)
    rng = np.random.default_rng(spec.seed)

    for k, push in enumerate(logs.frame_push_mm):
        if progress:
            progress(k / len(logs.frame_push_mm), f"Rendering photo {k + 1} of {len(logs.frame_push_mm)}")
        image = render_photo(texture, experiment, frame, shape, float(push), spec.image_noise_grey, rng)
        tifffile.imwrite(folder / "images" / f"SYNTH_{k:04d}.tif", image)

    logs.logger.to_csv(folder / "logger.csv", index=False)
    logs.calibration.to_csv(folder / "calibration.csv", index=False)
    project = {
        "name": f"Synthetic demo — {spec.preset} sand (SYNTHETIC DATA)",
        "synthetic": True,
        "images": {"folder": "images", "interval_s": float(logs.camera_times_s[1] - logs.camera_times_s[0])},
        "sensors": {"file": "logger.csv", "calibration_file": "calibration.csv"},
        "calibration": {"mm_per_px": frame.mm_per_px, "toe_px": [frame.toe_col, frame.toe_row],
                        "wall_height_mm": wall_height},
        "test": {"unit_weight_kn_m3": sensors.unit_weight_kn_m3, "wall_width_mm": sensors.wall_width_m * 1000,
                 "wall_weight_kn": sensors.wall_weight_kn,
                 "cell_depths_mm": {k: round(v, 3) for k, v in logs.pt_depths_mm.items()}},
        "sync": {"mode": "dic"},
    }
    truths = [experiment.truth(float(p)) for p in logs.frame_push_mm]
    truth = {
        **logs.truth,
        "preset": spec.preset,
        "frame_push_mm": logs.frame_push_mm.tolist(),
        "lift_off_span_deg": [t.lift_off_span_deg for t in truths],
        "shape": [t.shape for t in truths],
        "length_mm": [t.length_mm for t in truths],
        "area_mm2": [t.area_mm2 for t in truths],
        "bands": [[{"name": b.name, "label": b.label, "points": b.centreline[::20].round(2).tolist()} for b in t.bands]
                  for t in truths],
    }
    (folder / "project.json").write_text(json.dumps(project, indent=2))
    (folder / "truth.json").write_text(json.dumps(truth, indent=1, default=lambda v: None))
    if progress:
        progress(1.0, "Demo test ready")
    return folder / "project.json"
