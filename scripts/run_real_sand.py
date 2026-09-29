"""Run StrainLink's DIC, strain and band detection on real sand-box photos (PyPIV).

This is a real-soil test without known answers: it shows whether the
automatic chain finds the thrust faults (shear bands) a person sees in the
photos. It is not a retaining-wall test, so bands are reported unlabelled
(no toe / wall-top labelling) and no wall measurements are made.

Data: data/external/pypiv_sandbox/ (3 photos of a sand wedge pushed against
a backstop, Canon EOS 6D Mark II, 1890x846 px; PyPIV, GPL-3.0). See
Datasets.md for provenance.

Output: reports/real_sand/ (figure + summary.json). Run:
    .venv/bin/python scripts/run_real_sand.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks

from strainlink.bands.detect import DetectionSettings, detect_bands
from strainlink.dic.engine import DICSettings, Grid, track_sequence
from strainlink.strain import compute_strain

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "pypiv_sandbox"
OUT = ROOT / "reports" / "real_sand"
PHOTOS = ["subduction-4_0034.jpg", "subduction-4_0038.jpg", "subduction-4_0043.jpg"]

MM_PER_INCH = 25.4
# Tape rows (px) where the graduation ticks are crisp, and the column of the 12-inch mark,
# left of which the tape is graduated in 1/32 in and right of which in 1/16 in.
TAPE_ROWS = (725, 745)
FOOT_MARK_COL = 934
# Everything below this row is the tape measure, not soil.
SOIL_BOTTOM_ROW = 700
# In-focus soil: local grain contrast (std of the Laplacian over a subset)
# must exceed this fraction of the typical in-focus value. Defocused sand
# further back in the box is 3D and would give meaningless 2D motion.
SHARPNESS_WINDOW_PX = 31
SHARPNESS_FRACTION = 0.45
# Grains are ~5-8 px across, so a 41 px subset holds ~30 grains.
DIC = DICSettings(subset_radius=20, step=6, search_radius=20, min_zncc=0.7)
STRAIN_WINDOW = 5


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / name).convert("L"), dtype=np.float64)


def tape_scale(image: np.ndarray) -> tuple[float, float]:
    """mm per px from the tape ticks, with the spread between the two graduated halves as uncertainty."""
    profile = image[TAPE_ROWS[0]:TAPE_ROWS[1]].mean(axis=0)
    ticks, _ = find_peaks(-profile, prominence=40, distance=12)
    fine = ticks[ticks <= FOOT_MARK_COL]
    coarse = ticks[ticks >= FOOT_MARK_COL]
    px_per_inch = []
    for cols, per_inch in ((fine, 32), (coarse, 16)):
        slope = np.polyfit(np.arange(len(cols)), cols, 1)[0]
        px_per_inch.append(slope * per_inch)
    mean = float(np.mean(px_per_inch))
    return MM_PER_INCH / mean, MM_PER_INCH / mean * abs(px_per_inch[0] - px_per_inch[1]) / (2 * mean)


def in_focus_soil(image: np.ndarray) -> np.ndarray:
    lap = ndimage.laplace(ndimage.gaussian_filter(image, 1.0))
    mean = ndimage.uniform_filter(lap, SHARPNESS_WINDOW_PX)
    sharp = np.sqrt(np.maximum(ndimage.uniform_filter(lap**2, SHARPNESS_WINDOW_PX) - mean**2, 0))
    rows = np.arange(image.shape[0])[:, None]
    lower = sharp[SOIL_BOTTOM_ROW // 2:SOIL_BOTTOM_ROW]
    typical = float(np.percentile(lower, 75))
    mask = (sharp > SHARPNESS_FRACTION * typical) & (rows < SOIL_BOTTOM_ROW)
    # Keep the one connected in-focus soil body and close grain-scale gaps.
    mask = ndimage.binary_opening(mask, iterations=5)
    labels, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(mask)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = [load(p) for p in PHOTOS]
    mm_per_px, mm_sigma = tape_scale(images[0])
    soil = in_focus_soil(images[0])

    grid = Grid.over_roi((0, 0, images[0].shape[1], SOIL_BOTTOM_ROW), DIC.step, DIC.subset_radius + 2)
    # Only subsets that lie fully inside the in-focus soil are tracked.
    eroded = ndimage.binary_erosion(soil, iterations=DIC.subset_radius // 2)
    inside = eroded[grid.y.astype(int), grid.x.astype(int)]
    grid = Grid(x=np.where(inside, grid.x, np.nan), y=np.where(inside, grid.y, np.nan))

    t0 = time.perf_counter()
    track = track_sequence(images, grid, DIC)
    seconds = time.perf_counter() - t0

    ny, nx = grid.shape
    spacing_mm = DIC.step * mm_per_px
    # Soil coordinates: x to the right, z up from the tape edge (mm).
    x_mm = np.nan_to_num(grid.x, nan=0) * mm_per_px
    gx = np.arange(nx) * DIC.step + DIC.subset_radius + 2
    gy = np.arange(ny) * DIC.step + DIC.subset_radius + 2
    X, Y = np.meshgrid(gx, gy)
    x_mm, z_mm = X * mm_per_px, (SOIL_BOTTOM_ROW - Y) * mm_per_px

    results = {"photos": PHOTOS, "mm_per_px": mm_per_px, "mm_per_px_sigma": mm_sigma,
               "grid_points_in_soil": int(inside.sum()), "dic_seconds": round(seconds, 1), "steps": []}
    fields = []
    settings = DetectionSettings(min_width_mm=0.5, max_width_mm=8.0, min_length_mm=5.0, attach_tolerance_mm=2.0,
                                 lift_off_span_mm=5.0)
    for label, a, b in (("photo 34 → 38", 0, 1), ("photo 38 → 43", 1, 2), ("photo 34 → 43 (total)", 0, 2)):
        ok = track.valid[a] & track.valid[b]
        u = np.where(ok, track.x[b] - track.x[a], np.nan).reshape(ny, nx) * mm_per_px
        v = np.where(ok, track.y[b] - track.y[a], np.nan).reshape(ny, nx) * mm_per_px
        strain = compute_strain(u, v, spacing_mm, spacing_mm, STRAIN_WINDOW)
        gamma = strain.gamma_max
        found = detect_bands(gamma, x_mm, z_mm, wall_height_mm=float(np.nanmax(z_mm)), settings=settings, uz=-v)
        bands = [{"points_mm": np.round(bd.points, 2).tolist(), "length_mm": round(bd.length_mm, 1),
                  "width_mm": round(bd.width_mm, 2), "peak_gamma": round(bd.peak_gamma, 4),
                  "snr": round(bd.snr, 1), "flags": bd.flags} for bd in found.bands]
        tracked = int(ok.sum())
        results["steps"].append({
            "step": label, "tracked_points": tracked,
            "tracked_fraction": round(tracked / max(int(inside.sum()), 1), 3),
            "median_zncc": round(float(np.nanmedian(track.zncc[b][ok])), 3),
            "max_displacement_mm": round(float(np.nanmax(np.hypot(u, v))), 3),
            "bands": bands,
        })
        fields.append((label, u, v, gamma, found))

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    figure(images, soil, X, Y, fields, mm_per_px)
    print(json.dumps({k: v for k, v in results.items() if k != "steps"}, indent=2))
    for s in results["steps"]:
        print(s["step"], "tracked", s["tracked_points"], f"({100 * s['tracked_fraction']:.0f}%)",
              "zncc", s["median_zncc"], "max |u|", s["max_displacement_mm"], "mm")
        for bd in s["bands"]:
            p = np.asarray(bd["points_mm"])
            print(f"   band: length {bd['length_mm']} mm, width {bd['width_mm']} mm, SNR {bd['snr']}, "
                  f"from ({p[0, 0]:.1f}, {p[0, 1]:.1f}) to ({p[-1, 0]:.1f}, {p[-1, 1]:.1f}) mm, flags {bd['flags']}")


def figure(images, soil, X, Y, fields, mm_per_px) -> None:
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 5.2 * len(fields)))
    for ax, (label, u, v, gamma, found) in zip(axes, fields):
        ax.imshow(images[0], cmap="gray", vmin=0, vmax=255)
        shown = np.ma.masked_invalid(gamma)
        im = ax.pcolormesh(X, Y, shown, cmap="inferno", alpha=0.75, shading="auto",
                           vmin=0, vmax=np.nanpercentile(gamma, 99))
        step = 8
        ax.quiver(X[::step, ::step], Y[::step, ::step], u[::step, ::step] / mm_per_px, v[::step, ::step] / mm_per_px,
                  color="cyan", angles="xy", scale_units="xy", scale=0.25, width=0.0015)
        for bd in found.bands:
            cols = bd.points[:, 0] / mm_per_px
            rows = SOIL_BOTTOM_ROW - bd.points[:, 1] / mm_per_px
            ax.plot(cols, rows, color="lime", lw=2.2)
        ax.contour(soil, levels=[0.5], colors="white", linewidths=0.6)
        ax.set_title(f"{label}: max shear strain γmax, displacement ×4 (cyan), detected bands (green)")
        ax.set_xlim(0, images[0].shape[1])
        ax.set_ylim(SOIL_BOTTOM_ROW + 20, 150)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="γmax")
    fig.tight_layout()
    fig.savefig(OUT / "real_sand_bands.png", dpi=110)


if __name__ == "__main__":
    main()
