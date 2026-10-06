"""StrainLink on the head-to-head package, with the common settings, exported in the package's CSV format.

Its DIC engine and strain calculation are run on the stated rectangular area of
interest (the automatic area finder is not used, so every program measures the
same area). Other settings are StrainLink's defaults.

    .venv/bin/python scripts/head_to_head/run_strainlink.py
Output: data/head_to_head/results/strainlink/<case>.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import PACKAGE, RESULTS, STEP_PX, STRAIN_WINDOW_POINTS, SUBSET_PX, aoi, cases

from strainlink.dic import DICSettings, Grid, load_grayscale, track_sequence
from strainlink.io.images import list_images
from strainlink.strain import compute_strain


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", choices=["incremental", "fixed"], default="incremental",
                        help="incremental: updated every photo (StrainLink's default); fixed: always the first photo")
    args = parser.parse_args()
    out = RESULTS / ("strainlink" if args.reference == "incremental" else "strainlink_fixed")
    out.mkdir(parents=True, exist_ok=True)
    radius = SUBSET_PX // 2
    settings = DICSettings(subset_radius=radius, step=STEP_PX, search_radius=10, reference_mode=args.reference,
                           update_every=1 if args.reference == "incremental" else 0)
    for case in cases():
        started = time.perf_counter()
        images = [load_grayscale(p) for p in list_images(PACKAGE / case.name)]
        grid = Grid.over_roi(aoi(case.shape), STEP_PX, radius)          # subsets stay inside the area of interest
        tracked = track_sequence(images, grid, settings)
        k = len(images) - 1
        u, v = tracked.field(k, "u"), tracked.field(k, "v")
        strain = compute_strain(u, v, STEP_PX, STEP_PX, STRAIN_WINDOW_POINTS, order=2)
        table = pd.DataFrame({"x": grid.x.ravel(), "y": grid.y.ravel(), "u": u.ravel(), "v": v.ravel(),
                              "exx": strain.exx.ravel(), "eyy": strain.eyy.ravel(), "exy": strain.exy.ravel()})
        table[np.isfinite(table["u"])].to_csv(out / f"{case.name}.csv", index=False, float_format="%.8g")
        print(f"{case.name:13s} {np.isfinite(u).sum()} points in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
