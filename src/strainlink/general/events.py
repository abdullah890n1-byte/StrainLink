"""Events for any test: when a sensor channel peaks or drops suddenly, when a zone appears or runs through.

Every event is placed on the test's common axis (the loading displacement
if the logger records one, otherwise time since the first photo) with a
1-sigma uncertainty, so the correlation engine can compare any sensor
event with any object event.
"""

from __future__ import annotations

import numpy as np

from ..bands.track import BandTrack
from ..correlate.events import Event

# A peak must be followed by a fall of at least this many times the channel's
# own random wandering over the rest of the record (noise * sqrt(samples)),
# and must not sit in the last END_FRACTION of the record. Judging against
# the wandering, not a single step, keeps a drifting record (temperature,
# a random walk) from producing "peaks".
PEAK_DROP_NOISE = 3.0
END_FRACTION = 0.05
# A sudden drop: a fall of at least this share of the channel's range within
# DROP_WINDOW_FRACTION of the axis range, and at least DROP_NOISE times the
# channel's random wandering over that window (noise * sqrt(samples in it)).
DROP_RANGE_FRACTION = 0.1
DROP_WINDOW_FRACTION = 0.02
DROP_NOISE = 5.0
MODE_NAMES = {"opening": "crack", "shear": "shear band", "mixed": "mixed-mode zone"}


def noise_level(values: np.ndarray) -> float:
    """Robust noise of a record from its sample-to-sample differences."""
    diffs = np.diff(values[np.isfinite(values)])
    if len(diffs) < 3:
        return 0.0
    return float(1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2.0))


def channel_events(name: str, axis: np.ndarray, values: np.ndarray, axis_sigma: float) -> list[Event]:
    """Peak and largest sudden drop of one logger channel, on the common axis."""
    ok = np.isfinite(axis) & np.isfinite(values)
    a, v = axis[ok], values[ok]
    if len(v) < 10 or np.ptp(v) == 0:
        return []
    noise = noise_level(v)
    events = []
    span = float(np.ptp(a))
    k = int(np.argmax(v))
    wander_after = max(noise, 1e-12) * np.sqrt(max(len(v) - k, 1))
    if a[k] < a.max() - END_FRACTION * span and (v[k] - np.min(v[k:])) > PEAK_DROP_NOISE * wander_after:
        events.append(Event(f"{name}.peak", f"{name} reaches its peak", "sensor", float(a[k]),
                            max(axis_sigma, _peak_width(a, v, k, noise)), name, f"the peak of {name}"))
    drop = _largest_drop(a, v, span * DROP_WINDOW_FRACTION)
    if drop is not None:
        start, size = drop
        window_samples = max(DROP_WINDOW_FRACTION * len(v), 1.0)
        wander = max(noise, 1e-12) * np.sqrt(window_samples)
        if size >= DROP_RANGE_FRACTION * np.ptp(v) and size >= DROP_NOISE * wander:
            share = 100.0 * size / max(abs(v[:max(np.searchsorted(a, start), 1)]).max(), 1e-12)
            events.append(Event(f"{name}.drop", f"{name} drops suddenly by {share:.0f}%", "sensor", float(start),
                                axis_sigma, name, f"the sudden drop of {name}"))
    return events


def _peak_width(a: np.ndarray, v: np.ndarray, k: int, noise: float) -> float:
    """Half the axis span over which the channel is within noise of its peak."""
    near = np.flatnonzero(v >= v[k] - 2.0 * noise)
    return 0.5 * float(a[near].max() - a[near].min()) if len(near) else 0.0


def _largest_drop(a: np.ndarray, v: np.ndarray, window: float) -> tuple[float, float] | None:
    """(axis position where it starts, size) of the largest fall within `window` of axis."""
    best = None
    j = 0
    for i in range(len(v)):
        while a[j] < a[i] - window:
            j += 1
        size = float(v[j:i + 1].max() - v[i])
        if size > 0 and (best is None or size > best[1]):
            start = j + int(np.argmax(v[j:i + 1]))
            best = (float(a[start]), size)
    return best


def zone_events(tracks: list[BandTrack], modes: dict[str, str], crossing: dict[str, float],
                frame_sigma: float) -> list[Event]:
    """Object events: each zone's confirmed appearance, and when it first runs across the object."""
    events = []
    for t in tracks:
        kind = MODE_NAMES.get(modes.get(t.name, ""), "zone")
        if np.isfinite(t.onset_mm):
            events.append(Event(f"{t.name}.onset", f"{t.name} ({kind}) appears", "object", float(t.onset_mm),
                                frame_sigma, t.name, f"the appearance of {t.name} ({kind})"))
        if np.isfinite(crossing.get(t.name, np.nan)):
            events.append(Event(f"{t.name}.through", f"{t.name} ({kind}) runs across the object", "object",
                                float(crossing[t.name]), frame_sigma, t.name, f"{t.name} running across the object"))
    return events
