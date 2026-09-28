"""Scoring band detection against synthetic ground truth.

Used by the validation script and the tests. Tolerances below define what
counts as a correct measurement; a measurement outside tolerance that
carries no review flag is a *silent error*, the failure mode the accuracy
policy forbids (see Validation.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..bands.detect import FrameBands
from ..synth.wall import FrameTruth

# A detected band matches a true band if its mean distance to it is within this (mm).
MATCH_DISTANCE_MM = 15.0
# True bands shorter than this are not expected to be detectable (detector min length).
DETECTABLE_LENGTH_MM = 50.0
# Tolerances for "correct" measurements.
LIFT_OFF_TOL_DEG = 3.0
LENGTH_TOL_MM = 20.0
AREA_TOL_FRACTION = 0.05
HEAVE_TOL_MM = 1.0
CENTRELINE_TOL_MM = 3.0

# Which review flags cover which measurement (a flagged error is not silent).
_COVERS = {
    "lift_off": {"lift-off-uncertain", "band-does-not-reach-toe", "low-contrast", "near-data-gap"},
    "shape": {"shape-ambiguous", "low-contrast"},
    "length": {"not-reached-surface", "low-contrast", "near-data-gap"},
    "area": {"not-reached-surface", "low-contrast", "near-data-gap"},
}


def _polyline_length(points: np.ndarray) -> float:
    return float(np.hypot(*np.diff(points, axis=0).T).sum()) if len(points) > 1 else 0.0


def _densify(points: np.ndarray, step: float = 0.5) -> np.ndarray:
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    grid = np.arange(0.0, s[-1] + step, step)
    return np.column_stack([np.interp(grid, s, points[:, 0]), np.interp(grid, s, points[:, 1])])


def mean_distance(points: np.ndarray, reference: np.ndarray) -> float:
    """Mean distance from points to a reference polyline (mm)."""
    distance, _ = cKDTree(_densify(reference)).query(points)
    return float(np.mean(distance))


def truth_heave(experiment, truth: FrameTruth, x_top: np.ndarray, z_top: float, width_mm: float) -> float:
    """Ground-truth heave with the detector's definition: max uz on the top data row over the wedge."""
    reach = (truth.length_mm if np.isfinite(truth.length_mm) else
             (truth.bands[0].centreline[-1, 0] if truth.bands else 0.0)) + width_mm
    xs = x_top[(x_top >= 0) & (x_top <= reach)]
    if len(xs) == 0:
        return np.nan
    _, uz = experiment.displacement(xs, np.full_like(xs, z_top), truth.wall_displacement_mm)
    return float(np.max(uz))


@dataclass
class FrameScore:
    delta_mm: float
    true_bands: int
    detected_true: int
    false_bands: int
    primary_expected: bool
    primary_found: bool
    centreline_err_mm: float
    tip_err_mm: float
    lift_off_err_deg: float
    shape_true: str
    shape_found: str
    shape_ambiguous: bool
    reached_true: bool
    reached_found: bool
    length_err_mm: float
    area_err_frac: float
    heave_err_mm: float
    width_mm: float
    needs_review: bool
    silent_errors: list[str]


def score_frame(truth: FrameTruth, found: FrameBands, heave_true: float) -> FrameScore:
    expected = [b for b in truth.bands if _polyline_length(b.centreline) >= DETECTABLE_LENGTH_MM]
    candidates = [b for b in found.bands if b.label != "wall-interface"]
    used: set[int] = set()
    matched = 0
    for tb in expected:
        best = None
        for i, db in enumerate(candidates):
            if i in used or db.label != tb.label:
                continue
            d = mean_distance(db.points, tb.centreline)
            if d <= MATCH_DISTANCE_MM and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            used.add(best[1])
            matched += 1
    false_bands = len(candidates) - len(used)

    primary_truth = next((b for b in truth.bands if b.label == "primary"), None)
    primary_expected = primary_truth is not None and _polyline_length(primary_truth.centreline) >= DETECTABLE_LENGTH_MM
    primary_band = next((b for b in found.bands if b.label == "primary"), None)
    nan = float("nan")
    score = dict(centreline=nan, tip=nan, lift=nan, length=nan, area=nan, heave=nan)
    shape_found, ambiguous, reached_found, width = "", False, False, nan
    silent: list[str] = []
    if primary_band is not None and primary_truth is not None and found.primary is not None:
        pm = found.primary
        flags = set(pm.flags) | set(primary_band.flags)
        score["centreline"] = mean_distance(primary_band.points, primary_truth.centreline)
        if not primary_truth.reached_end:
            # The tip is only defined while the band is still forming; once it
            # reaches the surface its end is the exit point, scored through L.
            score["tip"] = float(np.hypot(*(primary_band.tip - primary_truth.centreline[-1])))
        score["lift"] = pm.lift_off_deg - truth.lift_off_span_deg
        shape_found, ambiguous = pm.shape.label, pm.shape.ambiguous
        reached_found = primary_band.reached_end
        width = primary_band.width_mm
        if np.isfinite(truth.length_mm) and np.isfinite(pm.length_mm):
            score["length"] = pm.length_mm - truth.length_mm
            score["area"] = (pm.area_mm2 - truth.area_mm2) / truth.area_mm2
        if np.isfinite(heave_true) and np.isfinite(pm.heave_mm):
            score["heave"] = pm.heave_mm - heave_true

        def check(name: str, bad: bool) -> None:
            if bad and not (flags & _COVERS[name]):
                silent.append(name)

        check("lift_off", abs(score["lift"]) > LIFT_OFF_TOL_DEG)
        check("shape", shape_found != truth.shape)
        check("length", np.isfinite(score["length"]) and abs(score["length"]) > LENGTH_TOL_MM)
        check("area", np.isfinite(score["area"]) and abs(score["area"]) > AREA_TOL_FRACTION)
        if primary_truth.reached_end != reached_found and reached_found:
            silent.append("reached-surface")
    if primary_expected and primary_band is None:
        silent.append("missed-primary")

    return FrameScore(
        delta_mm=truth.wall_displacement_mm, true_bands=len(expected), detected_true=matched,
        false_bands=false_bands, primary_expected=primary_expected, primary_found=primary_band is not None,
        centreline_err_mm=score["centreline"], tip_err_mm=score["tip"], lift_off_err_deg=score["lift"],
        shape_true=truth.shape, shape_found=shape_found, shape_ambiguous=ambiguous,
        reached_true=bool(primary_truth.reached_end) if primary_truth else False, reached_found=reached_found,
        length_err_mm=score["length"], area_err_frac=score["area"], heave_err_mm=score["heave"],
        width_mm=width, needs_review=found.needs_review, silent_errors=silent,
    )
