"""Head-to-head comparison of DIC programs on identical images with exact answers (SYNTHETIC DATA).

Every program analyses the same photos with the same settings, and every
result goes through the same scorer (score.py). The cases are the
full-frame cases of the Vic-2D task suite (scripts/vic2d_tasks.py): a
speckled plate deformed by an analytical field, so the true displacement
and strain are known at every point.

Conventions of the package (what a program must export, one CSV per case):
columns x, y, u, v and optionally exx, eyy, exy; pixels; x to the right, y
DOWN (image rows); positions in the first (reference) photo; displacement
of the last photo relative to the first; Green-Lagrange (Lagrangian)
strain. The scorer accepts Vic-2D's column names and either sign of v.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

PACKAGE = ROOT / "data" / "head_to_head" / "package"
RESULTS = ROOT / "data" / "head_to_head" / "results"
REPORT = ROOT / "reports" / "head_to_head.md"

# Settings every program uses.
SUBSET_PX = 31            # subset (correlation window) size
STEP_PX = 5               # spacing of measurement points
STRAIN_WINDOW_POINTS = 5  # strain filter: 5 points = 25 px
AOI_INSET_PX = 20         # area of interest: the photo less this margin on every side
# Strain is scored away from the AOI's edge (every program smooths differently there).
STRAIN_MARGIN_PX = 30

# Full-frame cases only (the same rectangular area of interest suits every program).
CASE_NAMES = ("translation", "rotation", "tension", "compression", "shear", "bending", "thermal", "static",
              "large_strain", "noise_high", "low_contrast", "blur", "lighting")


# Fine-detail cases: a displacement wave of this amplitude at each wavelength (px). How much of
# the amplitude a program recovers shows how much it blurs; the static case shows its noise.
WAVE_AMPLITUDE_PX = 0.5
WAVELENGTHS_PX = (200, 100, 60, 40)
WAVE_PHOTOS = 3


def wave_cases() -> list:
    from strainlink.synth import cases as C
    return [C.Case(f"wave_{w}", f"Displacement wave, wavelength {w} px (amplitude {WAVE_AMPLITUDE_PX} px)",
                   C.wave(WAVE_AMPLITUDE_PX, float(w)), photos=WAVE_PHOTOS, seed=3, meta={"wavelength_px": w})
            for w in WAVELENGTHS_PX]


def cases() -> list:
    import vic2d_tasks
    wanted = {c.name: c for c in vic2d_tasks.cases()}
    return [wanted[name] for name in CASE_NAMES] + wave_cases()


def suite_folder(case) -> Path:
    """Where a case's photos come from: the task suite's folder, or (wave cases) one rendered here."""
    import vic2d_tasks
    if "wavelength_px" in case.meta:
        return ROOT / "data" / "head_to_head" / "rendered" / case.name
    return vic2d_tasks.OUT / case.name


def aoi(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Area of interest (x0, y0, x1, y1), px, end exclusive."""
    rows, cols = shape
    return AOI_INSET_PX, AOI_INSET_PX, cols - AOI_INSET_PX, rows - AOI_INSET_PX


def truth(case, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """True displacement (px) and Green-Lagrange strain at reference points (image axes, y down)."""
    from strainlink.synth import cases as C
    u, v = case.field(x, y, 1.0)
    return {"u": u, "v": v, **C.true_strain(case.field, x, y, 1.0)}
