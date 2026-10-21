"""Validate lens-distortion correction (SYNTHETIC DATA, exact answers).

A strong barrel lens (k1 = -0.25, k2 = 0.08: photo corners moved ~40 px) is
applied to ideal rendered photos of two classic cases:

- rigid translation (12 px): the true strain is zero everywhere;
- tension 1% (ν = 0.3): the true strain is uniform.

Each is analysed twice, without and with the correction fitted from a
checkerboard photo taken through the same lens, and scored: the strain
over the whole photo (median and its spread) and, with the correction,
the displacement against the exact answer.

Run:  .venv/bin/python scripts/validate_lens.py
Output: data/synthetic/LENS/, reports/lens_validation.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile

from strainlink.general import run
from strainlink.io.images import list_images
from strainlink.lens import LensModel, calibrate, distort
from strainlink.synth import cases as C

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic" / "LENS"
W, H = 900, 600
LENS = LensModel(np.array([[900.0, 0, (W - 1) / 2], [0, 900.0, (H - 1) / 2], [0, 0, 1]]),
                 np.array([-0.25, 0.08, 0.0, 0.0, 0.0]), (W, H), 0.0, 0, (0, 0))
SQUARE_PX = 55
MM_PER_PX = 0.05
CENTRE = ((W - 1) / 2, (H - 1) / 2)


def checkerboard() -> np.ndarray:
    board = np.full((H, W), 230.0)
    x0, y0 = (W - 14 * SQUARE_PX) // 2, (H - 9 * SQUARE_PX) // 2
    for i in range(14):
        for j in range(9):
            if (i + j) % 2 == 0:
                board[y0 + j * SQUARE_PX:y0 + (j + 1) * SQUARE_PX, x0 + i * SQUARE_PX:x0 + (i + 1) * SQUARE_PX] = 25.0
    board = cv2.GaussianBlur(board, (0, 0), 0.8)
    return distort(board, LENS) + np.random.default_rng(3).normal(0, 1.0, board.shape)


def through_lens(folder: Path) -> None:
    """Replace the ideal photos by what the barrel lens records."""
    for path in list_images(folder / "photos"):
        ideal = tifffile.imread(path).astype(float)
        tifffile.imwrite(path, np.clip(np.round(distort(ideal, LENS)), 0, 255).astype(np.uint8))


def score(case: C.Case, folder: Path, corrected: bool) -> dict:
    f = np.load(folder / "results" / "fields.npz")
    k = len(f["ux"]) - 1
    from scipy import ndimage
    ok = ndimage.binary_erosion(np.isfinite(f["exx"][k]), iterations=4) & np.isfinite(f["exx"][k])
    exx, ezz = f["exx"][k][ok], f["ezz"][k][ok]
    m = case.meta
    true_exx = 0.5 * ((1 + m.get("strain", 0.0)) ** 2 - 1)
    true_ezz = 0.5 * ((1 - m.get("poisson", 0.0) * m.get("strain", 0.0)) ** 2 - 1)
    out = {"case": case.name, "corrected": corrected,
           "exx_median": float(np.median(exx)), "exx_true": true_exx,
           "exx_spread_p5_p95": float(np.percentile(exx, 95) - np.percentile(exx, 5)),
           "worst_exx_error": float(np.max(np.abs(exx - true_exx))),
           "ezz_median": float(np.median(ezz)), "ezz_true": true_ezz,
           "worst_ezz_error": float(np.max(np.abs(ezz - true_ezz)))}
    if corrected:
        # In corrected coordinates the positions are the ideal ones, so the exact answer applies.
        summary = json.loads((folder / "results" / "summary.json").read_text(encoding="utf-8"))
        X, Y = f["x_mm"] / MM_PER_PX, summary["frame"]["height_px"] - f["z_mm"] / MM_PER_PX
        tu, tv = case.field(X, Y, 1.0)
        u, v = f["ux"][k] / MM_PER_PX, -f["uz"][k] / MM_PER_PX
        meas = np.isfinite(u)
        out["disp_rms_px"] = float(np.sqrt(np.mean((u[meas] - tu[meas]) ** 2 + (v[meas] - tv[meas]) ** 2)))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = checkerboard()
    tifffile.imwrite(OUT / "calibration_grid.tif", np.clip(np.round(grid), 0, 255).astype(np.uint8))
    model = calibrate([grid], square_mm=SQUARE_PX * MM_PER_PX)
    print(f"Lens fitted: {model.board} board, fit {model.rms_px:.3f} px RMS, k1 {model.distortion.ravel()[0]:.4f}, "
          f"k2 {model.distortion.ravel()[1]:.4f}; scale from the grid {model.mm_per_px:.5f} mm/px (true {MM_PER_PX})")
    cases = [C.Case("translation", "Rigid translation 12 px through a barrel lens", C.translation(12.0, 0.0),
                    shape=(H, W), photos=5, meta={}),
             C.Case("tension", "Tension 1% through a barrel lens", C.uniaxial(0.01, 0.3, CENTRE), shape=(H, W), photos=5,
                    meta={"strain": 0.01, "poisson": 0.3})]
    results = []
    for case in cases:
        base = OUT / case.name
        C.render_case(case, base / "raw")
        through_lens(base / "raw")
        for corrected in (False, True):
            folder = base / ("corrected" if corrected else "uncorrected")
            if folder.exists():
                shutil.rmtree(folder)
            shutil.copytree(base / "raw", folder)
            project = json.loads((folder / "project.json").read_text(encoding="utf-8"))
            if corrected:
                project["calibration"]["lens"] = model.to_json()
            (folder / "project.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
            run(folder / "project.json")
            r = score(case, folder, corrected)
            results.append(r)
            print(f"{case.name:11s} {'with' if corrected else 'without':7s} correction: εxx {r['exx_median'] * 1e6:7.0f} µε "
                  f"(true {r['exx_true'] * 1e6:.0f}), spread {r['exx_spread_p5_p95'] * 1e6:6.0f} µε, worst εxx error "
                  f"{r['worst_exx_error'] * 1e6:6.0f} µε, worst εzz error {r['worst_ezz_error'] * 1e6:6.0f} µε"
                  + (f", displacement {r['disp_rms_px']:.4f} px RMS" if corrected else ""))
        shutil.rmtree(base / "raw")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "lens_validation.json").write_text(json.dumps(
        {"lens": model.to_json(), "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
