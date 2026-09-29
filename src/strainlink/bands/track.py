"""Following shear bands through a whole test.

Per-frame detections become band *tracks*: the same physical band
identified in every frame, with its measurements as series against wall
displacement. The primary band (band-1) is unique per frame. Other bands
are matched frame to frame by label and position (mean distance between
centrelines). A band's onset is the first wall displacement from which it
is detected in ONSET_CONFIRM_FRAMES consecutive frames, so a single
spurious detection cannot set an onset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .detect import DetectedBand, FrameBands

# Consecutive frames a band must be seen in before its onset is confirmed.
ONSET_CONFIRM_FRAMES = 2
# Bands in consecutive frames are the same band if their centrelines are
# within this many band widths of each other on average.
MATCH_WIDTHS = 2.0


@dataclass
class BandTrack:
    name: str
    label: str
    frames: list[int] = field(default_factory=list)          # frame indices where seen
    delta_mm: list[float] = field(default_factory=list)
    bands: list[DetectedBand] = field(default_factory=list)

    @property
    def onset_mm(self) -> float:
        """First wall displacement of a confirmed (consecutive) detection run; NaN if none."""
        run = 1
        for k in range(1, len(self.frames)):
            run = run + 1 if self.frames[k] == self.frames[k - 1] + 1 else 1
            if run >= ONSET_CONFIRM_FRAMES:
                return self.delta_mm[k - ONSET_CONFIRM_FRAMES + 1]
        return self.delta_mm[0] if ONSET_CONFIRM_FRAMES <= 1 and self.delta_mm else float("nan")

    def series(self, attribute: str) -> np.ndarray:
        return np.array([getattr(b, attribute) for b in self.bands], dtype=float)


@dataclass
class PrimaryHistory:
    """band-1 measurements against wall displacement (NaN where not measurable)."""

    delta_mm: np.ndarray
    detected: np.ndarray
    developed_length_mm: np.ndarray
    tip_x_mm: np.ndarray
    tip_z_mm: np.ndarray
    lift_off_deg: np.ndarray
    lift_off_sigma_deg: np.ndarray
    length_mm: np.ndarray
    area_mm2: np.ndarray
    heave_mm: np.ndarray
    width_mm: np.ndarray
    reached_surface: np.ndarray
    surface_confirmed: np.ndarray   # reached, with the band traced right up to the top of the data
    shape: list[str]
    needs_review: np.ndarray

    @property
    def onset_mm(self) -> float:
        return _confirmed_onset(self.delta_mm, self.detected)

    @property
    def surface_reached_mm(self) -> float:
        return _confirmed_onset(self.delta_mm, self.reached_surface)

    @property
    def surface_reached_range_mm(self) -> tuple[float, float]:
        """Earliest and latest push at which band-1 can have reached the surface.

        From the first photo where the tip enters the unresolvable top zone,
        to the first photo where it is confirmed at the top of the data (or
        the end of the test, if it never is).
        """
        first = self.surface_reached_mm
        if not np.isfinite(first):
            return float("nan"), float("nan")
        confirmed = _confirmed_onset(self.delta_mm, self.surface_confirmed)
        return first, confirmed if np.isfinite(confirmed) else float(self.delta_mm[-1])


@dataclass
class TestBands:
    primary: PrimaryHistory
    tracks: list[BandTrack]

    def by_label(self, label: str) -> list[BandTrack]:
        return [t for t in self.tracks if t.label == label]


def _confirmed_onset(delta: np.ndarray, flags: np.ndarray) -> float:
    run = 0
    for k, ok in enumerate(flags):
        run = run + 1 if ok else 0
        if run >= ONSET_CONFIRM_FRAMES:
            return float(delta[k - ONSET_CONFIRM_FRAMES + 1])
    return float("nan")


def _mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    distance, _ = cKDTree(b).query(a)
    return float(np.mean(distance))


def track(frames: list[FrameBands], delta_mm: list[float] | np.ndarray) -> TestBands:
    """Link per-frame detections (ordered by wall displacement) into band tracks."""
    delta = np.asarray(delta_mm, dtype=float)
    if len(frames) != len(delta):
        raise ValueError("one wall displacement per frame is required")
    return TestBands(primary=_primary_history(frames, delta), tracks=_secondary_tracks(frames, delta))


def _primary_history(frames: list[FrameBands], delta: np.ndarray) -> PrimaryHistory:
    n = len(frames)
    nan = np.full(n, np.nan)
    columns = {k: nan.copy() for k in (
        "developed_length_mm", "tip_x_mm", "tip_z_mm", "lift_off_deg", "lift_off_sigma_deg",
        "length_mm", "area_mm2", "heave_mm", "width_mm")}
    detected = np.zeros(n, dtype=bool)
    reached = np.zeros(n, dtype=bool)
    confirmed = np.zeros(n, dtype=bool)
    review = np.zeros(n, dtype=bool)
    shapes = [""] * n
    for k, frame in enumerate(frames):
        review[k] = frame.needs_review
        band = next((b for b in frame.bands if b.label == "primary"), None)
        if band is None or frame.primary is None:
            continue
        p = frame.primary
        detected[k] = True
        reached[k] = band.reached_end
        confirmed[k] = band.reached_end and "surface-reach-unconfirmed" not in p.flags
        shapes[k] = p.shape.label + ("?" if p.shape.ambiguous else "")
        columns["developed_length_mm"][k] = p.developed_length_mm
        columns["tip_x_mm"][k], columns["tip_z_mm"][k] = band.tip
        columns["lift_off_deg"][k] = p.lift_off_deg
        columns["lift_off_sigma_deg"][k] = p.lift_off_sigma_deg
        columns["length_mm"][k] = p.length_mm
        columns["area_mm2"][k] = p.area_mm2
        columns["heave_mm"][k] = p.heave_mm
        columns["width_mm"][k] = band.width_mm
    return PrimaryHistory(delta_mm=delta, detected=detected, reached_surface=reached, surface_confirmed=confirmed,
                          shape=shapes, needs_review=review, **columns)


def _secondary_tracks(frames: list[FrameBands], delta: np.ndarray) -> list[BandTrack]:
    tracks: list[BandTrack] = []
    counters: dict[str, int] = {}
    prefixes = {"radial": "R", "intermediate": "I", "wall-interface": "W", "unattached": "U"}
    for k, frame in enumerate(frames):
        claimed: set[int] = set()
        for band in sorted((b for b in frame.bands if b.label != "primary"), key=lambda b: -b.length_mm):
            best = None
            for t_index, t in enumerate(tracks):
                if t_index in claimed or t.label != band.label or t.frames[-1] < k - 1:
                    continue
                d = _mean_distance(band.points, t.bands[-1].points)
                if d <= MATCH_WIDTHS * band.width_mm and (best is None or d < best[0]):
                    best = (d, t_index)
            if best is None:
                counters[band.label] = counters.get(band.label, 0) + 1
                tracks.append(BandTrack(name=f"{prefixes[band.label]}{counters[band.label]}", label=band.label))
                t_index = len(tracks) - 1
            else:
                t_index = best[1]
            claimed.add(t_index)
            t = tracks[t_index]
            t.frames.append(k)
            t.delta_mm.append(float(delta[k]))
            t.bands.append(band)
    return tracks
