"""µDIC (open source, Olufsen et al., SoftwareX 2020) on the head-to-head package.

Run with µDIC's own environment (it needs an older Python and NumPy). To create it:
    /usr/bin/python3 -m venv data/external/mudic_env
    data/external/mudic_env/bin/python -m pip install --upgrade pip
    data/external/mudic_env/bin/python -m pip install numpy==1.23.5 "scipy<1.11" "matplotlib<3.8" \
        "scikit-image<0.22" "Pillow<10.1" numba dill natsort nose tifffile
    data/external/mudic_env/bin/python -m pip install --no-deps muDIC==0.2.1
Then:
    data/external/mudic_env/bin/python scripts/head_to_head/run_mudic.py [--update-reference]
Output: data/head_to_head/results/mudic/<case>.csv

µDIC is a global (finite-element) DIC code: it has elements, not subsets. The
element size is set to the package's subset size (31 px, bilinear Q4 elements,
its default), over the same area of interest; other settings are µDIC's
defaults, with no reference update. Displacement and Green-Lagrange strain are
sampled inside the elements (UPSCALE points per element side).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import muDIC as dic
import numpy as np

from common import AOI_INSET_PX, PACKAGE, RESULTS, SUBSET_PX

UPSCALE = 6          # sampling of the fields inside each element (about one point per 5 px)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-reference", action="store_true", help="update the reference at every photo")
    parser.add_argument("--element", type=int, default=SUBSET_PX, help="element size, px")
    parser.add_argument("--only", nargs="*", help="only these cases")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    name = "mudic_updated" if args.update_reference else "mudic"
    out = RESULTS / (name if args.element == SUBSET_PX else f"{name}_element{args.element}")
    out.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in Path(PACKAGE).iterdir() if p.is_dir()):
        if args.only and folder.name not in args.only:
            continue
        started = time.perf_counter()
        stack = dic.image_stack_from_folder(str(folder) + "/", file_type=".tif")
        rows, cols = stack[0].shape
        x0, y0, x1, y1 = AOI_INSET_PX, AOI_INSET_PX, cols - AOI_INSET_PX, rows - AOI_INSET_PX
        n_elx, n_ely = round((x1 - x0) / args.element), round((y1 - y0) / args.element)
        mesh = dic.Mesher(deg_e=1, deg_n=1, type="q4").mesh(stack, Xc1=x0, Xc2=x1, Yc1=y0, Yc2=y1,
                                                             n_elx=n_elx, n_ely=n_ely, GUI=False)
        try:
            updates = list(range(1, len(stack))) if args.update_reference else []
            results = dic.DICAnalysis(dic.DICInput(mesh, stack, ref_update_frames=updates, maxit=40)).run()
            fields = dic.Fields(results, upscale=UPSCALE)
        except Exception as error:                      # a case the program cannot do is part of the result
            print(f"{folder.name:13s} FAILED: {type(error).__name__}: {error}", flush=True)
            continue
        coords, disp, strain = fields.coords(), fields.disp(), fields.green_strain()
        # µDIC reads photos bottom row first (its y axis points up, from the photo's last row).
        # Converted here to the package's convention: y down from the first row, v and exy with it.
        table = {"x": coords[0, 0, :, :, 0].ravel(), "y": (rows - 1) - coords[0, 1, :, :, 0].ravel(),
                 "u": disp[0, 0, :, :, -1].ravel(), "v": -disp[0, 1, :, :, -1].ravel(),
                 "exx": strain[0, 0, 0, :, :, -1].ravel(), "eyy": strain[0, 1, 1, :, :, -1].ravel(),
                 "exy": -strain[0, 0, 1, :, :, -1].ravel()}
        ok = np.isfinite(table["u"])
        header = ",".join(table)
        np.savetxt(out / f"{folder.name}.csv", np.column_stack([v[ok] for v in table.values()]), delimiter=",",
                   header=header, comments="", fmt="%.8g")
        print(f"{folder.name:13s} {ok.sum()} points, {n_elx} x {n_ely} elements, {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
