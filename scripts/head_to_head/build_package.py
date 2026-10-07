"""Build the head-to-head package: the photos, the settings sheet and the export format.

Run after scripts/vic2d_tasks.py (it renders the photos):
    .venv/bin/python scripts/head_to_head/build_package.py
Output: data/head_to_head/package/ (one folder of photos per case, SETTINGS.md)
"""

from __future__ import annotations

import shutil

from common import (AOI_INSET_PX, PACKAGE, STEP_PX, STRAIN_WINDOW_POINTS, SUBSET_PX, aoi, cases, suite_folder)

from strainlink.io.images import list_images
from strainlink.synth import cases as C


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases():
        if "wavelength_px" in case.meta and not (suite_folder(case) / "photos").exists():
            C.render_case(case, suite_folder(case))
        photos = list_images(suite_folder(case) / "photos")
        target = PACKAGE / case.name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        for path in photos:
            shutil.copy2(path, target / path.name)
        x0, y0, x1, y1 = aoi(case.shape)
        rows.append(f"| `{case.name}` | {case.title} | {len(photos)} | {case.shape[1]} × {case.shape[0]} | "
                    f"x {x0}–{x1}, y {y0}–{y1} |")
    sheet = f"""# Head-to-head DIC comparison: settings sheet (SYNTHETIC DATA, exact answers)

Every program analyses the same photos with the same settings; every result
is scored the same way (`scripts/head_to_head/score.py`).

## Settings (identical for every program)

| Setting | Value | In Vic-2D |
|---|---|---|
| Reference photo | the first photo of each folder (`img_000`) | reference image |
| Photo scored | the last photo of each folder | last deformed image |
| Area of interest | the photo less {AOI_INSET_PX} px on every side (rectangle, see table) | AOI rectangle |
| Subset size | {SUBSET_PX} px | Subset = {SUBSET_PX} |
| Step | {STEP_PX} px | Step = {STEP_PX} |
| Strain filter | {STRAIN_WINDOW_POINTS} points ({STRAIN_WINDOW_POINTS * STEP_PX} px) | Filter size = {STRAIN_WINDOW_POINTS} |
| Strain measure | Green–Lagrange | Tensor type = Lagrange |
| Units | pixels (no scale calibration) | leave the scale uncalibrated |
| Everything else | the program's defaults (write down what they were) | interpolation, criterion, weights, incremental correlation |

## Cases

| Folder | What it is | Photos | Size (px) | Area of interest |
|---|---|---|---|---|
{chr(10).join(rows)}

## What to export (one CSV file per case, named `<folder>.csv`)

The data of the **last** photo, one row per measurement point, with columns

    x, y, u, v, exx, eyy, exy

- `x`, `y`: position in the reference photo, pixels (x to the right, y down).
- `u`, `v`: displacement, pixels.
- `exx`, `eyy`, `exy`: strain (optional; a program without strain is scored on displacement only).

Vic-2D: Data → Export → CSV, all variables, for the last image. Its own column
names and its `sigma` column (−1 marks points it could not correlate) are understood,
and either sign convention for `v` and `exy`.

## Scoring

    .venv/bin/python scripts/head_to_head/score.py "Vic-2D 7" path/to/folder/with/csv/files

writes `reports/head_to_head.md`: displacement error (RMS, px) over the whole
area, strain error (RMS, µε) away from its edge, and the share of the area
each program measured, next to every other program scored so far.
"""
    (PACKAGE / "SETTINGS.md").write_text(sheet)
    print(f"Package: {PACKAGE} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
