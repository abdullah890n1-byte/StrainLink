"""A synthetic test on any object, with exact known answers (SYNTHETIC DATA).

A speckled plate on a plain background is stretched along x. Part-way
through the test a localisation zone forms along a known inclined line
through the plate: a crack that opens (mode "opening") or a shear band
that slides (mode "shear"). From that moment the plate unloads
elastically and the zone takes the extra movement, as in a real test.

The logger records the crosshead displacement, the load (which drops
suddenly when the zone forms: the planted relationship) and an unrelated
temperature channel (which must not be linked to anything). The logger's
clock starts CLOCK_OFFSET_S before the first photo, so matching the clocks
is part of the test.

All files are SYNTHETIC and labelled so. truth.json holds the answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .speckle import SpeckleTexture, _pull_back

BACKGROUND_GREY = 30.0
PIXEL_SAMPLES = 2
# Logger clock: it starts this long before the first photo, sampling at LOGGER_HZ.
CLOCK_OFFSET_S = 12.5
LOGGER_HZ = 10.0
# Load drop when the zone forms (share of the load), then gradual unloading.
LOAD_DROP_FRACTION = 0.3


@dataclass(frozen=True)
class GeneralSpec:
    mode: str = "opening"             # "opening" (crack) or "shear" (shear band)
    mm_per_px: float = 0.1
    plate_mm: tuple[float, float] = (80.0, 50.0)
    image_px: tuple[int, int] = (600, 900)     # rows, cols
    photos: int = 41
    photo_interval_s: float = 2.0
    stretch_rate_mm_s: float = 0.0125          # crosshead speed
    onset_photo: int = 20                      # first photo showing the zone
    zone_angle_deg: float = 70.0               # zone direction from horizontal
    zone_width_mm: float = 3.0              # wide enough that the rendered jump never folds the image
    youngs_modulus_mpa: float = 2000.0
    thickness_mm: float = 10.0
    poisson: float = 0.3
    image_noise_grey: float = 1.5
    seed: int = 7
    markers: int = 0                           # dark marker dots painted on the plate (move with it)
    marker_radius_px: float = 6.0


def _geometry(spec: GeneralSpec):
    rows, cols = spec.image_px
    centre = np.array([cols / 2 * spec.mm_per_px, rows / 2 * spec.mm_per_px])      # mm, z up from bottom
    angle = np.radians(spec.zone_angle_deg)
    tangent = np.array([np.cos(angle), np.sin(angle)])
    # The normal points to the +x side, so that side is the one the crosshead pulls away.
    normal = np.array([tangent[1], -tangent[0]])
    return centre, tangent, normal


def history(spec: GeneralSpec) -> dict[str, np.ndarray]:
    """Crosshead displacement, elastic strain and zone jump at every photo."""
    t = np.arange(spec.photos) * spec.photo_interval_s
    crosshead = spec.stretch_rate_mm_s * t
    length = spec.plate_mm[0]
    t_onset = spec.onset_photo * spec.photo_interval_s
    eps_onset = spec.stretch_rate_mm_s * t_onset / length
    # After onset the plate unloads by LOAD_DROP_FRACTION at once, then holds that strain.
    elastic = np.where(t < t_onset, crosshead / length, (1 - LOAD_DROP_FRACTION) * eps_onset)
    jump = np.clip(crosshead - elastic * length, 0.0, None)      # the zone takes the rest (mm, along x)
    return {"t": t, "crosshead": crosshead, "elastic": elastic, "jump_x": jump}


def displacement(spec: GeneralSpec, eps: float, jump_x: float):
    """(u, v) in px at reference pixel positions (cols, rows), v downward."""
    centre, tangent, normal = _geometry(spec)
    rows = spec.image_px[0]
    # The zone's jump vector: its x-component is jump_x (the crosshead's extra movement).
    # Opening: the +x side moves away along the normal. Shear: it slides along the zone,
    # in the direction that also carries it towards +x.
    direction = normal if spec.mode == "opening" else (tangent if tangent[0] > 0 else -tangent)
    jump = direction * (jump_x / direction[0]) if abs(direction[0]) > 1e-9 else direction * 0.0

    def field(cx: np.ndarray, ry: np.ndarray):
        x = cx * spec.mm_per_px - centre[0]
        z = (rows - ry) * spec.mm_per_px - centre[1]
        side = x * normal[0] + z * normal[1]
        step = 0.5 * (1 + np.tanh(2.0 * side / spec.zone_width_mm))   # smooth jump across the zone
        ux = eps * x + jump[0] * (step - 0.5)
        uz = -spec.poisson * eps * z + jump[1] * (step - 0.5)
        return ux / spec.mm_per_px, -uz / spec.mm_per_px

    return field, jump


MARKER_GREY = 8.0


def marker_centres(spec: GeneralSpec) -> np.ndarray:
    """Reference positions (col, row px) of the marker dots: random on the plate, away from its edges."""
    if not spec.markers:
        return np.empty((0, 2))
    rng = np.random.default_rng(spec.seed + 100)
    rows, cols = spec.image_px
    centre, _, _ = _geometry(spec)
    half = np.array(spec.plate_mm) / 2 / spec.mm_per_px - 4 * spec.marker_radius_px
    cpx = np.array([centre[0] / spec.mm_per_px, rows - centre[1] / spec.mm_per_px])
    points: list[np.ndarray] = []
    while len(points) < spec.markers:
        p = cpx + rng.uniform(-1, 1, 2) * half
        if all(np.hypot(*(p - q)) > 6 * spec.marker_radius_px for q in points):
            points.append(p)
    return np.asarray(points)


def render(spec: GeneralSpec, texture: SpeckleTexture, eps: float, jump_x: float, rng) -> np.ndarray:
    rows, cols = spec.image_px
    field, _ = displacement(spec, eps, jump_x)
    sub = (np.arange(PIXEL_SAMPLES) + 0.5) / PIXEL_SAMPLES - 0.5
    sy = (np.arange(rows)[:, None] + sub[None, :]).ravel()
    sx = (np.arange(cols)[:, None] + sub[None, :]).ravel()
    gx, gy = np.meshgrid(sx, sy)
    ref_x, ref_y = _pull_back(gx, gy, field)
    centre, _, _ = _geometry(spec)
    half = np.array(spec.plate_mm) / 2 / spec.mm_per_px
    cpx = np.array([centre[0] / spec.mm_per_px, rows - centre[1] / spec.mm_per_px])
    on_plate = (np.abs(ref_x - cpx[0]) <= half[0]) & (np.abs(ref_y - cpx[1]) <= half[1])
    value = np.where(on_plate, 40.0 + 170.0 * texture.evaluate(ref_x, ref_y), BACKGROUND_GREY)
    for mx, my in marker_centres(spec):
        # A marker is part of the material (drawn at its reference position), with a soft
        # one-pixel edge as a camera lens gives it: hard-edged dots jitter by ~0.1 px as they
        # cross the pixel grid, which real markers don't.
        cover = np.clip(spec.marker_radius_px + 0.5 - np.hypot(ref_x - mx, ref_y - my), 0.0, 1.0)
        value = cover * MARKER_GREY + (1.0 - cover) * value
    image = value.reshape(rows, PIXEL_SAMPLES, cols, PIXEL_SAMPLES).mean(axis=(1, 3))
    image = image + rng.normal(0.0, spec.image_noise_grey, image.shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def make_general_test(folder: str | Path, spec: GeneralSpec | None = None) -> Path:
    spec = spec or GeneralSpec()
    folder = Path(folder)
    (folder / "photos").mkdir(parents=True, exist_ok=True)
    h = history(spec)
    rng = np.random.default_rng(spec.seed)
    texture = SpeckleTexture.granular(spec.image_px, grain_radius_px=2.0, seed=spec.seed)
    for k in range(spec.photos):
        tifffile.imwrite(folder / "photos" / f"SYNTH_{k:04d}.tif", render(spec, texture, h["elastic"][k], h["jump_x"][k], rng))

    # Logger on its own clock, starting CLOCK_OFFSET_S before the first photo.
    duration = h["t"][-1] + CLOCK_OFFSET_S + 5.0
    lt = np.arange(0.0, duration, 1.0 / LOGGER_HZ)
    t_photo = lt - CLOCK_OFFSET_S
    crosshead = np.clip(spec.stretch_rate_mm_s * t_photo, 0.0, None)
    area = spec.plate_mm[1] * spec.thickness_mm
    t_onset = spec.onset_photo * spec.photo_interval_s
    eps_onset = spec.stretch_rate_mm_s * t_onset / spec.plate_mm[0]
    elastic = np.where(t_photo < t_onset, crosshead / spec.plate_mm[0], (1 - LOAD_DROP_FRACTION) * eps_onset)
    load_kn = spec.youngs_modulus_mpa * elastic * area / 1000.0
    noise = np.random.default_rng(spec.seed + 1)
    logger = pd.DataFrame({
        "Time (s)": lt.round(3),
        "Crosshead (mm)": (crosshead + noise.normal(0, 0.0005, lt.size)).round(5),
        "Load (kN)": (load_kn + noise.normal(0, 0.002, lt.size)).round(4),
        "Temperature (C)": (21.0 + np.cumsum(noise.normal(0, 0.01, lt.size))).round(3),
    })
    logger.to_csv(folder / "logger.csv", index=False)

    centre, tangent, normal = _geometry(spec)
    half = np.array(spec.plate_mm) / 2
    # Zone centreline clipped to the plate (reference positions, mm).
    s = np.linspace(-100, 100, 4001)
    line = centre + s[:, None] * tangent
    on = (np.abs(line[:, 0] - centre[0]) <= half[0]) & (np.abs(line[:, 1] - centre[1]) <= half[1])
    _, jump = displacement(spec, 0.0, 1.0)
    project = {
        "kind": "general", "name": f"Synthetic plate — {'crack' if spec.mode == 'opening' else 'shear band'} (SYNTHETIC DATA)",
        "synthetic": True,
        "images": {"folder": "photos", "interval_s": spec.photo_interval_s},
        "calibration": {"mm_per_px": spec.mm_per_px},
        "sensors": {"file": "logger.csv"},
        "sync": {"mode": "auto"},
    }
    (folder / "project.json").write_text(json.dumps(project, indent=2))
    truth = {
        "kind": "general", "synthetic": True, "mode": spec.mode, "onset_photo": spec.onset_photo,
        "zone_angle_deg": spec.zone_angle_deg, "zone_points_mm": line[on][::40].round(3).tolist(),
        "mean_exx": h["elastic"].tolist(), "mean_ezz": (-spec.poisson * h["elastic"]).tolist(),
        "jump_x_mm": h["jump_x"].tolist(),
        "opening_mm": (h["jump_x"] * (jump @ normal)).tolist() if spec.mode == "opening" else None,   # positive = opening
        "slip_mm": (h["jump_x"] * abs(jump @ tangent)).tolist() if spec.mode == "shear" else None,
        "clock_offset_s": CLOCK_OFFSET_S, "crosshead_at_photos_mm": h["crosshead"].tolist(),
        "load_drop_at_crosshead_mm": float(spec.stretch_rate_mm_s * t_onset),
        "planted_link": ["Load (kN)", "zone onset"], "unrelated_channel": "Temperature (C)",
        "note": "Mean strains are over the plate away from the zone (the elastic strain).",
    }
    if spec.markers:
        centres = marker_centres(spec)
        moves = []
        for k in range(spec.photos):
            field, _ = displacement(spec, h["elastic"][k], h["jump_x"][k])
            u, v = field(centres[:, 0], centres[:, 1])
            moves.append(np.column_stack([u, v]).round(5).tolist())
        truth["markers"] = {"centres_px": centres.round(4).tolist(), "radius_px": spec.marker_radius_px,
                            "displacement_px": moves}
        project["markers"] = {"polarity": "dark", "diameter_px": [0.8 * 2 * spec.marker_radius_px,
                                                                  2.5 * 2 * spec.marker_radius_px]}
        (folder / "project.json").write_text(json.dumps(project, indent=2))
    (folder / "truth.json").write_text(json.dumps(truth, indent=1))
    return folder / "project.json"
