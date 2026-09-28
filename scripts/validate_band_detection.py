"""Validate automatic band detection against synthetic ground truth (SYNTHETIC data).

Sweeps test type (loose / medium / dense sand, medium with intermediate
bands), wall displacement and displacement noise. Displacement fields are
generated on a DIC-like grid, noise is added to mimic DIC measurement
error, strain is computed exactly as for real data, and the detector is
scored against the known bands.

Usage:  python scripts/validate_band_detection.py [--noise 0 0.02 0.05 0.1] [--csv out.csv]
Prints Markdown tables for Validation.md.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from strainlink.bands import detect_bands
from strainlink.strain import compute_strain
from strainlink.synth.wall import WallExperiment, preset, with_intermediate_bands
from strainlink.validation.bands import score_frame, truth_heave

GRID_SPACING_MM = 5.0          # DIC grid step (e.g. 5 px at 1 mm/px)
DOMAIN_X_MM = (0.0, 900.0)
DOMAIN_Z_MM = (-150.0, 390.0)
STRAIN_WINDOW = 5
DELTAS_MM = [2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60]
SEED = 2026


def experiments() -> dict[str, WallExperiment]:
    return {
        "loose": WallExperiment(preset("loose")),
        "medium": WallExperiment(preset("medium")),
        "dense": WallExperiment(preset("dense")),
        "medium+intermediate": WallExperiment(with_intermediate_bands(preset("medium"))),
    }


def run(noise_levels: list[float]) -> pd.DataFrame:
    xs = np.arange(DOMAIN_X_MM[0], DOMAIN_X_MM[1] + 1e-9, GRID_SPACING_MM)
    zs = np.arange(DOMAIN_Z_MM[1], DOMAIN_Z_MM[0] - 1e-9, -GRID_SPACING_MM)
    x, z = np.meshgrid(xs, zs)
    rng = np.random.default_rng(SEED)
    rows = []
    for name, experiment in experiments().items():
        height = experiment.spec.wall_height_mm
        for delta in DELTAS_MM:
            ux, uz = experiment.displacement(x, z, delta)
            truth = experiment.truth(delta)
            heave = truth_heave(experiment, truth, xs, zs[0], experiment.spec.band_width_mm)
            for noise in noise_levels:
                nx = ux + rng.normal(0.0, noise, ux.shape) if noise else ux
                nz = uz + rng.normal(0.0, noise, uz.shape) if noise else uz
                strain = compute_strain(nx, nz, GRID_SPACING_MM, -GRID_SPACING_MM, STRAIN_WINDOW)
                started = time.perf_counter()
                found = detect_bands(strain.gamma_max, x, z, height, uz=nz)
                elapsed = time.perf_counter() - started
                score = asdict(score_frame(truth, found, heave))
                score.update(test=name, noise_mm=noise, seconds=elapsed)
                rows.append(score)
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 2) -> str:
    return "–" if not np.isfinite(value) else f"{value:.{digits}f}"


def summarise(df: pd.DataFrame) -> str:
    out = []
    out.append("### Detection (bands with developed length ≥ 50 mm)\n")
    out.append("| Test | Noise (mm) | True bands | Detected | Recall | False bands | Primary found / expected |")
    out.append("|---|---|---|---|---|---|---|")
    for (test, noise), g in df.groupby(["test", "noise_mm"], sort=False):
        tb, dt = g.true_bands.sum(), g.detected_true.sum()
        out.append(f"| {test} | {noise} | {tb} | {dt} | {100 * dt / max(tb, 1):.1f}% | {g.false_bands.sum()} | "
                   f"{(g.primary_found & g.primary_expected).sum()} / {g.primary_expected.sum()} |")

    out.append("\n### Primary band (band-1) measurements, where found\n")
    out.append("| Noise (mm) | Centreline mean err (mm) | Tip err median, bands still forming (mm) | Lift-off err |median| / max (°) "
               "| L err |median| / max (mm) | Area err |median| / max | Heave err |median| (mm) | Width median (mm) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for noise, g in df.groupby("noise_mm"):
        found = g[g.primary_found]
        lift = found.lift_off_err_deg.abs()
        length = found.length_err_mm.abs()
        area = found.area_err_frac.abs()
        out.append(
            f"| {noise} | {_fmt(found.centreline_err_mm.mean())} | {_fmt(found.tip_err_mm.median(), 1)} | "
            f"{_fmt(lift.median())} / {_fmt(lift.max())} | {_fmt(length.median(), 1)} / {_fmt(length.max(), 1)} | "
            f"{_fmt(100 * area.median(), 1)}% / {_fmt(100 * area.max(), 1)}% | {_fmt(found.heave_err_mm.abs().median())} | "
            f"{_fmt(found.width_mm.median(), 1)} |")

    out.append("\n### Shape classification (primary band)\n")
    out.append("| Noise (mm) | Frames | Correct | Wrong but flagged ambiguous | Wrong and not flagged (silent) |")
    out.append("|---|---|---|---|---|")
    for noise, g in df.groupby("noise_mm"):
        found = g[g.primary_found]
        correct = (found.shape_found == found.shape_true)
        flagged = ~correct & found.shape_ambiguous
        silent = ~correct & ~found.shape_ambiguous
        out.append(f"| {noise} | {len(found)} | {correct.sum()} | {flagged.sum()} | {silent.sum()} |")

    out.append("\n### Silent errors (outside tolerance and not flagged for review)\n")
    out.append("| Noise (mm) | Frames | Frames flagged for review | Silent errors | Details |")
    out.append("|---|---|---|---|---|")
    for noise, g in df.groupby("noise_mm"):
        silent = g[g.silent_errors.map(len) > 0]
        details = "; ".join(f"{r.test} δ={r.delta_mm}: {','.join(r.silent_errors)}" for r in silent.itertuples())
        out.append(f"| {noise} | {len(g)} | {g.needs_review.sum()} | {len(silent)} | {details or 'none'} |")
    out.append(f"\nMedian detection time per frame: {df.seconds.median():.2f} s (Apple M2, CPU).")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.02, 0.05, 0.1])
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    df = run(args.noise)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(summarise(df))


if __name__ == "__main__":
    main()
