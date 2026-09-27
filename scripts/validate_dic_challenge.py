"""Validate the DIC engine on the 2D DIC Challenge 1.0 rigid-shift sets.

Ground truth comes from the challenge's own set definitions
(set_descriptions.xlsx and file names). Reference: Reu et al. (2018),
Exp. Mech. 58(7):1067-1099, https://doi.org/10.1007/s11340-017-0349-0.

Usage:  python scripts/validate_dic_challenge.py [--subset-radius 10]
Prints a Markdown table (bias, noise, RMSE per set) for Validation.md.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strainlink.dic import DICSettings, Grid, SplineImage, correlate_pair, load_grayscale

DATA = Path(__file__).resolve().parents[1] / "data" / "external" / "dic_challenge_1"
ROI_MARGIN_PX = 40
GRID_STEP_PX = 10


@dataclass
class ShiftCase:
    image: Path
    shift_x: float
    shift_y: float


def _cases_numbered(folder: str, pattern: str, px_per_index: float) -> tuple[Path, list[ShiftCase]]:
    files = sorted((DATA / folder).glob("*.tif"))
    indexed = []
    for f in files:
        match = re.search(pattern, f.name)
        if match:
            indexed.append((int(match.group(1)), f))
    indexed.sort()
    reference = indexed[0][1]
    cases = [ShiftCase(f, i * px_per_index, i * px_per_index) for i, f in indexed[1:]]
    return reference, cases


def _cases_named(folder: str) -> tuple[Path, list[ShiftCase]]:
    files = sorted((DATA / folder).glob("*.tif"))
    reference = next(f for f in files if "Reference" in f.name)
    cases = []
    for f in files:
        match = re.search(r"X(\d+\.\d+) Y(\d+\.\d+)", f.name)
        if match and float(match.group(1)) > 0:
            cases.append(ShiftCase(f, float(match.group(1)), float(match.group(2))))
    return reference, cases


def _cases_sample6_7(folder: str) -> tuple[Path, list[ShiftCase]]:
    files = sorted((DATA / folder).glob("*.tif"))
    reference = next(f for f in files if "Reference" in f.name)
    cases = []
    for f in files:
        match = re.search(r"Y(\d+) X(\d+)", f.name)
        if match and int(match.group(2)) > 0:
            cases.append(ShiftCase(f, 0.1 * int(match.group(2)), 0.1 * int(match.group(1))))
    return reference, cases


SETS = {
    "Sample1": lambda: _cases_numbered("Sample1", r"_(\d+)\.tif$", 0.05),
    "Sample2": lambda: _cases_numbered("Sample2", r"_(\d+)\.tif$", 0.05),
    "Sample3": lambda: _cases_named("Sample3"),
    "Sample4": lambda: _cases_named("Sample4"),
    "Sample5": lambda: _cases_named("Sample5"),
    "Sample6": lambda: _cases_sample6_7("Sample6"),
    "Sample7": lambda: _cases_sample6_7("Sample7"),
}


def evaluate_set(name: str, settings: DICSettings) -> dict:
    reference_path, cases = SETS[name]()
    reference_pixels = load_grayscale(reference_path)
    reference = SplineImage.from_array(reference_pixels)
    h, w = reference_pixels.shape
    grid = Grid.over_roi((0, 0, w, h), GRID_STEP_PX, ROI_MARGIN_PX)
    errors_u, errors_v, valid_fraction = [], [], []
    signs = []
    started = time.perf_counter()
    for case in cases:
        current = SplineImage.from_array(load_grayscale(case.image), with_gradients=False)
        pair = correlate_pair(reference, current, grid.x.ravel(), grid.y.ravel(), settings)
        u, v = pair.u[pair.valid], pair.v[pair.valid]
        # Sign conventions differ between sets (Sample1/2 shift content in -y),
        # so the sign is detected per axis from the data rather than assumed.
        sign_u = np.sign(np.median(u)) if case.shift_x else 1.0
        sign_v = np.sign(np.median(v)) if case.shift_y else 1.0
        signs.append((sign_u, sign_v))
        errors_u.append(u - sign_u * case.shift_x)
        errors_v.append(v - sign_v * case.shift_y)
        valid_fraction.append(pair.valid.mean())
    elapsed = time.perf_counter() - started
    eu, ev = np.concatenate(errors_u), np.concatenate(errors_v)
    per_image_bias = [np.mean(e) for e in errors_u]
    return {
        "set": name,
        "images": len(cases),
        "points": grid.x.size,
        "sign": _describe_signs(np.array(signs)),
        "bias_u": np.mean(eu), "bias_v": np.mean(ev),
        "max_abs_image_bias": np.max(np.abs(per_image_bias)),
        "noise_u": np.mean([np.std(e) for e in errors_u]),
        "noise_v": np.mean([np.std(e) for e in errors_v]),
        "rmse": np.sqrt(np.mean(np.concatenate([eu, ev]) ** 2)),
        "valid": np.mean(valid_fraction),
        "seconds": elapsed,
    }


def _describe_signs(signs: np.ndarray) -> str:
    def axis(column: np.ndarray) -> str:
        if np.all(column > 0):
            return "+"
        return "-" if np.all(column < 0) else "mixed"

    return f"u{axis(signs[:, 0])} v{axis(signs[:, 1])}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-radius", type=int, default=10)
    parser.add_argument("--min-zncc", type=float, default=0.8)
    parser.add_argument("--sets", nargs="*", default=list(SETS))
    args = parser.parse_args()
    settings = DICSettings(subset_radius=args.subset_radius, search_radius=5, reference_mode="fixed",
                           min_zncc=args.min_zncc)
    side = 2 * args.subset_radius + 1
    print(f"Subset {side}x{side} px, grid step {GRID_STEP_PX} px, ROI margin {ROI_MARGIN_PX} px, "
          f"min ZNCC {args.min_zncc}\n")
    print("| Set | Images | Points/img | Sign | Bias u (px) | Bias v (px) | Max per-image bias (px) "
          "| Noise σu (px) | Noise σv (px) | RMSE (px) | Valid | Time (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in args.sets:
        r = evaluate_set(name, settings)
        print(f"| {r['set']} | {r['images']} | {r['points']} | {r['sign']} | {r['bias_u']:+.4f} | "
              f"{r['bias_v']:+.4f} | {r['max_abs_image_bias']:.4f} | {r['noise_u']:.4f} | "
              f"{r['noise_v']:.4f} | {r['rmse']:.4f} | {100 * r['valid']:.1f}% | {r['seconds']:.1f} |",
              flush=True)


if __name__ == "__main__":
    main()
