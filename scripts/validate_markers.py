"""Validate the marker check (SYNTHETIC DATA with exact answers).

A plate test with dark marker dots painted on the material (they move with
it) is rendered, analysed with markers switched on, and scored three ways:

1. marker tracker vs the exact truth — is the reference itself trustworthy?
2. DIC vs the markers — what a user sees on a real test;
3. DIC vs the exact truth at the same points — so (2) can be checked as a
   measure of DIC's real error.

Run:  .venv/bin/python scripts/validate_markers.py [--analyse-only]
Output: data/synthetic/SYNTH_markers/ (results/markers.csv, accuracy in the printout)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strainlink.general import run
from strainlink.synth.general import GeneralSpec, make_general_test

ROOT = Path(__file__).resolve().parents[1]
SPEC = GeneralSpec(mode="opening", zone_angle_deg=70.0, markers=12, marker_radius_px=6.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "data" / "synthetic" / "SYNTH_markers"
    project = folder / "project.json"
    if not (args.analyse_only and project.exists()):
        project = make_general_test(folder, SPEC)
    summary = run(project)
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))["markers"]
    true_centres = np.asarray(truth["centres_px"])
    true_moves = np.asarray(truth["displacement_px"])               # (frames, markers, 2)
    m = summary["markers"]
    tracks = pd.read_csv(folder / "results" / "markers.csv")
    found = np.column_stack([m["first_positions_px"], m["first_rows_px"]])
    # Match each detected marker to the nearest true one.
    match = [int(np.argmin(np.hypot(*(true_centres - f).T))) for f in found]
    offsets = [float(np.min(np.hypot(*(true_centres - f).T))) for f in found]

    tracker_err, dic_truth_err = [], []
    fields = np.load(folder / "results" / "fields.npz")
    mm = json.loads(project.read_text(encoding="utf-8"))["calibration"]["mm_per_px"]
    x_mm, z_mm = fields["x_mm"], fields["z_mm"]
    grid_cols = x_mm[0] / mm
    height = summary["frame"]["height_px"]
    grid_rows = height - z_mm[:, 0] / mm
    from scipy import ndimage
    from strainlink.bands.ridges import fill_missing
    in_zone = set(m["in_zones"])
    for i, t in enumerate(match):
        if i in in_zone:
            continue
        x = tracks[f"M{i + 1} x (px)"].to_numpy()
        y = tracks[f"M{i + 1} y (px)"].to_numpy()
        ok = np.isfinite(x)
        tracker_err.append(np.hypot(x[ok] - x[0] - true_moves[ok, t, 0], y[ok] - y[0] - true_moves[ok, t, 1]))
        coords = np.array([[(true_centres[t, 1] - grid_rows[0]) / (grid_rows[1] - grid_rows[0])],
                           [(true_centres[t, 0] - grid_cols[0]) / (grid_cols[1] - grid_cols[0])]])
        for k in range(1, len(x)):
            u = ndimage.map_coordinates(fill_missing(fields["ux"][k] / mm), coords, order=1)[0]
            v = ndimage.map_coordinates(fill_missing(-fields["uz"][k] / mm), coords, order=1)[0]
            dic_truth_err.append(np.hypot(u - true_moves[k, t, 0], v - true_moves[k, t, 1]))
    tracker = np.concatenate(tracker_err)
    dic_truth = np.asarray(dic_truth_err)
    print(f"Markers painted: {len(true_centres)}; found: {m['found']} (first-photo position error "
          f"median {np.median(offsets):.3f} px); followed to the end: {m['followed_to_end']}; "
          f"in the crack zone (reported separately): {len(in_zone)}")
    print(f"1. Marker tracker vs truth: RMS {np.sqrt(np.mean(tracker**2)):.4f} px, max {tracker.max():.4f} px")
    print(f"2. DIC vs markers:          RMS {m['rms_px']:.4f} px ({m['rms_mm'] * 1000:.2f} µm), max {m['max_px']:.4f} px")
    print(f"3. DIC vs truth (same pts): RMS {np.sqrt(np.mean(dic_truth**2)):.4f} px, max {dic_truth.max():.4f} px")
    if m.get("in_zones"):
        print(f"   markers in the crack zone: DIC vs markers RMS {m['rms_in_zones_px']:.3f} px (a point vs a subset across a crack)")
    (folder / "results" / "marker_validation.json").write_text(json.dumps({
        "painted": len(true_centres), "found": m["found"], "tracker_rms_px": float(np.sqrt(np.mean(tracker**2))),
        "dic_vs_markers_rms_px": m["rms_px"], "dic_vs_truth_rms_px": float(np.sqrt(np.mean(dic_truth**2))),
        "in_zones": m["in_zones"], "rms_in_zones_px": m["rms_in_zones_px"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
