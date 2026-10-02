"""StrainLink vs DICe on the same images: the published Brazilian-test videos (REAL images).

Arza-García et al. (2022, open access) published two DICe result videos of
their BTS test on a compacted soil disc: the speckled images with a
semi-transparent colour layer of DICe's DISPLACEMENT_X (video 1) and
DISPLACEMENT_Y (video 2). StrainLink analyses video 1's frames itself
(any-object mode, project in data/external/springer_bts/StrainLink analysis).

DICe's answer is recovered from the colours: blending a colour with the
grey speckle changes its saturation and brightness but not its hue, so
each pixel's hue, read through the colour bar, gives DICe's value. The
comparison then fits DICe's change since the first frame to StrainLink's
displacement (u, v in px) at the same material points, away from the
crack, which also tells which component and unit DICe used.

Run:  .venv/bin/python scripts/compare_springer_dice.py [--project other/project.json --out folder]
Output: reports/springer_dice/ (summary.json, figure)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "springer_bts"
PROJECT = DATA / "StrainLink analysis" / "project.json"   # any analysis of video 1 (see --project)
RESULTS = PROJECT.parent / "results"
OUT = ROOT / "reports" / "springer_dice"
VIDEOS = {"DISPLACEMENT_X": (DATA / "43452_2021_325_MOESM1_ESM.mp4", (1.0, 2.3)),
          "DISPLACEMENT_Y": (DATA / "43452_2021_325_MOESM2_ESM.mp4", (1.5, 2.6))}
# Colour bar position in the video (px) and its value range (read from its labels).
BAR_COLUMNS = (668, 678)
BAR_ROWS = (42, 672)
# Pixels this pale or this dark carry no reliable hue (white/black speckle, the crack).
MIN_SATURATION = 60
MIN_VALUE = 60
WINDOW = 4                 # half-size (px) of the window whose hues are pooled per point
CRACK_EXCLUSION_MM = 2.0   # points this close to the crack are left out of the comparison


def hue(bgr: np.ndarray) -> np.ndarray:
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(float)
    angle = h[..., 0]
    return np.where(angle > 150, angle - 180, angle), h[..., 1], h[..., 2]      # red wraps to negative


def colour_scale(frame: np.ndarray, value_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(hue, value) pairs along the colour bar, sorted by hue for interpolation."""
    bar = frame[BAR_ROWS[0]:BAR_ROWS[1] + 1, BAR_COLUMNS[0]:BAR_COLUMNS[1] + 1]
    h, _, _ = hue(bar)
    hues = np.median(h, axis=1)
    values = np.linspace(value_range[1], value_range[0], len(hues))       # top of the bar = maximum
    order = np.argsort(hues)
    return hues[order], values[order]


def read_frames(video: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames, wanted, k = {}, set(indices), 0
    while wanted and cap.grab():
        if k in wanted:
            frames[k] = cap.retrieve()[1]
            wanted.discard(k)
        k += 1
    cap.release()
    return frames


def decode(frame: np.ndarray, cols: np.ndarray, rows: np.ndarray, scale) -> np.ndarray:
    h, s, v = hue(frame)
    out = np.full(cols.shape, np.nan)
    for i, (c, r) in enumerate(zip(np.round(cols).astype(int), np.round(rows).astype(int))):
        if not (WINDOW <= r < frame.shape[0] - WINDOW and WINDOW <= c < frame.shape[1] - WINDOW):
            continue
        win = (slice(r - WINDOW, r + WINDOW + 1), slice(c - WINDOW, c + WINDOW + 1))
        good = (s[win] > MIN_SATURATION) & (v[win] > MIN_VALUE)
        if good.sum() >= 8:
            out[i] = np.interp(np.median(h[win][good]), *scale)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    project = json.loads(PROJECT.read_text())
    mm_per_px = project["calibration"]["mm_per_px"]
    fields = np.load(RESULTS / "fields.npz")
    summary = json.loads((RESULTS / "summary.json").read_text())
    bands = json.loads((RESULTS / "bands.json").read_text())
    height = summary["frame"]["height_px"]
    with open(PROJECT.parent / project["images"]["folder"] / "frames.csv") as f:
        video_frames = [int(r["video_frame"]) for r in csv.DictReader(f)]

    x, z = fields["x_mm"], fields["z_mm"]
    ref_col, ref_row = x / mm_per_px, height - z / mm_per_px
    n = len(video_frames)
    compare_frames = list(range(0, n, 4)) + [n - 1]
    # The crack's position (final zone), to leave its neighbourhood out.
    crack = np.vstack([np.asarray(b["points"], float) for b in bands[-1]]) if bands[-1] else np.empty((0, 2))
    from scipy.spatial import cKDTree
    near_crack = (cKDTree(crack).query(np.column_stack([x.ravel(), z.ravel()]))[0].reshape(x.shape) < CRACK_EXCLUSION_MM
                  if len(crack) else np.zeros(x.shape, bool))

    result = {"frames_compared": len(compare_frames), "mm_per_px": mm_per_px, "components": {}}
    for name, (video, value_range) in VIDEOS.items():
        frames = read_frames(video, [video_frames[k] for k in compare_frames])
        scale = colour_scale(frames[video_frames[0]], value_range)
        base = decode(frames[video_frames[0]], ref_col.ravel(), ref_row.ravel(), scale)
        # Each component is compared with the StrainLink component it matches (X with u, Y with v),
        # allowing a sign (axis direction), a scale (units) and one offset per frame (reference).
        # Colours pinned at either end of the colour bar are clipped values and are left out.
        comp = 0 if name.endswith("X") else 1
        lo, hi = value_range
        margin = 0.01 * (hi - lo)
        xs, ds, frame_ids = [], [], []
        for k in compare_frames[1:]:
            u_px = fields["ux"][k] / mm_per_px
            v_px = -fields["uz"][k] / mm_per_px        # image rows point down
            ok = np.isfinite(u_px) & ~near_crack
            cols, rows = (ref_col + np.nan_to_num(u_px))[ok], (ref_row + np.nan_to_num(v_px))[ok]
            value = decode(frames[video_frames[k]], cols, rows, scale)
            good = np.isfinite(value) & (value > lo + margin) & (value < hi - margin)
            xs.append((u_px if comp == 0 else v_px)[ok][good])
            ds.append(value[good] - base.reshape(x.shape)[ok][good])
            frame_ids.append(np.full(good.sum(), k))
        sl, d, fid = np.concatenate(xs), np.concatenate(ds), np.concatenate(frame_ids)
        # Remove each frame's mean (the offset), then fit the scale through the origin.
        sl_c = sl - np.array([sl[fid == k].mean() for k in fid])
        d_c = d - np.array([d[fid == k].mean() for k in fid])
        a = float(sl_c @ d_c / (sl_c @ sl_c))
        fitted, residual = a * sl_c, d_c - a * sl_c
        r2 = 1 - residual.var() / d_c.var()
        result["components"][name] = {
            "compared_with": "u (horizontal)" if comp == 0 else "v (vertical, image rows down)",
            "points": int(len(d)), "dice_units_per_video_px": a, "sign": "same" if a > 0 else "opposite",
            "r2": float(r2), "rms_difference_dice_units": float(np.sqrt(np.mean(residual**2))),
            "rms_difference_video_px": float(np.sqrt(np.mean(residual**2)) / abs(a)),
            "dice_change_range": [float(np.percentile(d_c, 1)), float(np.percentile(d_c, 99))],
        }
        d = d_c
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.plot(fitted, d, ",", alpha=0.3, color="#2f6fed")
        lim = [min(fitted.min(), d.min()), max(fitted.max(), d.max())]
        ax.plot(lim, lim, color="#e0452b", lw=1)
        ax.set_xlabel(f"StrainLink (as {name}, fitted units)")
        ax.set_ylabel(f"DICe {name} change")
        ax.set_title(f"{name}: R² = {r2:.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{name.lower()}.png", dpi=120)
        plt.close(fig)
    result["crack"] = crack_timing(fields, bands, summary, video_frames, ref_col, ref_row, x, z, mm_per_px)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def crack_timing(fields, bands, summary, video_frames, ref_col, ref_row, x, z, mm_per_px) -> dict:
    """DICe's X jump across the crack (strips 2-5 mm either side) vs StrainLink's crack opening, per photo."""
    import pandas as pd
    final = [b for b in bands[-1] if b.get("mode") == "opening"]
    if not final:
        return {}
    line = np.asarray(final[0]["points"], float)
    line = line[np.argsort(line[:, 1])]
    side = x - np.interp(z, line[:, 1], line[:, 0])
    left, right = (side < -2) & (side > -5), (side > 2) & (side < 5)
    video, value_range = VIDEOS["DISPLACEMENT_X"]
    frames = read_frames(video, video_frames)
    scale = colour_scale(frames[video_frames[0]], value_range)
    jumps = []
    for k, vf in enumerate(video_frames):
        u, v = fields["ux"][k] / mm_per_px, -fields["uz"][k] / mm_per_px
        d = decode(frames[vf], (ref_col + np.nan_to_num(u)).ravel(), (ref_row + np.nan_to_num(v)).ravel(), scale).reshape(x.shape)
        jumps.append(float(np.nanmedian(d[left]) - np.nanmedian(d[right])))
    jumps = np.array(jumps) - jumps[0]
    per_photo = pd.read_csv(RESULTS / "per_photo.csv")
    opening = per_photo[f"{final[0]['name']} opening (mm)"].to_numpy()
    t = per_photo["time_s"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(t, jumps, "o-", ms=3, color="#868e96", label="DICe: X jump across the crack (DICe units)")
    ax.plot(t, opening, "o-", ms=3, color="#e8590c", label="StrainLink: crack opening (mm)")
    ax.set_xlabel("time in the video (s)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "crack_timing.png", dpi=120)
    plt.close(fig)
    zone = next(z_ for z_ in summary["zones"] if z_["name"] == final[0]["name"])
    return {"strainlink_onset_s": zone["onset"], "time_s": t.tolist(), "dice_x_jump": jumps.round(4).tolist(),
            "strainlink_opening_mm": [None if not np.isfinite(o) else round(float(o), 4) for o in opening]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="another StrainLink analysis of video 1 (default: the reference one)")
    parser.add_argument("--out", help="output folder")
    args = parser.parse_args()
    if args.project:
        PROJECT = Path(args.project).resolve()
        RESULTS = PROJECT.parent / "results"
    if args.out:
        OUT = Path(args.out)
    main()
