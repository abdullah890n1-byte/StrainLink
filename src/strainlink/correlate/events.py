"""Events: the moments in a test when something changes, each with an uncertainty.

Sensor events come from the mechanics results (peaks, plateau starts,
softening onsets, uplift). Soil events come from the band tracks (band
onsets, band-1 reaching the surface) and from combining the two: the push
at which band-1's tip reaches the level of each pressure cell.

All events are expressed in wall push (mm), the common axis of the test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..bands.track import TestBands
from ..mechanics import MechanicsResults

Source = Literal["sensor", "soil"]


@dataclass(frozen=True)
class Event:
    key: str            # stable identifier, e.g. "PT3.softening"
    label: str          # human-readable, e.g. "PT3 starts to soften"
    source: Source
    push_mm: float
    sigma_mm: float     # 1-sigma uncertainty of the push
    subject: str = ""   # channel or band the event belongs to
    phrase: str = ""    # noun phrase for sentences, e.g. "the appearance of R1 (radial band)"


def _frame_sigma(push: np.ndarray) -> float:
    """Timing resolution of a frame-based event: half the photo spacing."""
    steps = np.diff(push[np.isfinite(push)])
    return float(0.5 * np.median(steps)) if len(steps) else float("nan")


def sensor_events(mech: MechanicsResults, logger_step_mm: float) -> list[Event]:
    events = []
    p = mech.passive
    width = p.push_range_mm[1] - p.push_range_mm[0]
    peak_sigma = max(width / 4.0, logger_step_mm) if not p.plateau else max((p.push_at_peak_mm - p.push_range_mm[0]) / 2, logger_step_mm)
    events.append(Event("force.peak" if not p.plateau else "force.plateau",
                        "Passive force reaches its peak" if not p.plateau else "Passive force reaches its plateau",
                        "sensor", p.push_at_peak_mm, peak_sigma, "force"))
    if p.softens and np.isfinite(p.softening_onset_mm):
        events.append(Event("force.softening", "Passive force starts to soften", "sensor",
                            p.softening_onset_mm, max(logger_step_mm, 0.25), "force"))
    for cell in mech.cells:
        if cell.softens and np.isfinite(cell.softening_onset_mm):
            events.append(Event(f"{cell.name}.softening", f"{cell.name} starts to soften", "sensor",
                                cell.softening_onset_mm, max(logger_step_mm, 0.25), cell.name))
        elif cell.plateau:
            events.append(Event(f"{cell.name}.plateau", f"{cell.name} reaches a plateau", "sensor",
                                cell.push_at_peak_mm, max((cell.push_range_mm[1] - cell.push_range_mm[0]) / 4, logger_step_mm), cell.name))
    if np.isfinite(mech.uplift_onset_mm):
        events.append(Event("uplift.onset", "Wall starts to lift", "sensor", mech.uplift_onset_mm,
                            max(logger_step_mm, 0.25), "uplift"))
    return events


def soil_events(history: TestBands) -> list[Event]:
    events = []
    primary = history.primary
    sigma = _frame_sigma(primary.delta_mm)
    if np.isfinite(primary.onset_mm):
        events.append(Event("band-1.onset", "Band-1 (primary slip surface) appears", "soil", primary.onset_mm, sigma, "band-1",
                            "the appearance of band-1 (the primary slip surface)"))
    if np.isfinite(primary.surface_reached_mm):
        lo, hi = primary.surface_reached_range_mm
        # An unconfirmed breakthrough is uncertain over its whole range.
        events.append(Event("band-1.surface", "Band-1 reaches the soil surface", "soil",
                            primary.surface_reached_mm, max(sigma, (hi - lo) / 2), "band-1", "band-1 reaching the soil surface"))
    for t in history.tracks:
        if t.label in ("radial", "intermediate") and np.isfinite(t.onset_mm):
            kind = "radial band" if t.label == "radial" else "intermediate band"
            events.append(Event(f"{t.name}.onset", f"{t.name} ({kind}) appears", "soil", t.onset_mm, sigma, t.name,
                                f"the appearance of {t.name} ({kind})"))
    return events


def tip_reach_events(history: TestBands, levels_mm: dict[str, float], tip_sigma_mm: float) -> list[Event]:
    """Push at which band-1's tip first reaches each level (height above the toe).

    The push is interpolated between photos. Its uncertainty combines the
    photo spacing with the tip-position uncertainty converted to push
    through the local rate of tip rise.
    """
    primary = history.primary
    push, tip = primary.delta_mm, primary.tip_z_mm
    ok = np.isfinite(tip)
    events = []
    if ok.sum() < 2:
        return events
    push, tip = push[ok], np.maximum.accumulate(tip[ok])      # the tip does not retreat
    frame_sigma = _frame_sigma(primary.delta_mm)
    for name, level in levels_mm.items():
        if tip[-1] < level:
            continue
        k = int(np.argmax(tip >= level))
        if k == 0:
            reach = push[0]
            rate = np.nan
        else:
            reach = float(np.interp(level, tip[k - 1:k + 1], push[k - 1:k + 1]))
            rate = (tip[k] - tip[k - 1]) / max(push[k] - push[k - 1], 1e-9)
        sigma = float(np.hypot(frame_sigma, tip_sigma_mm / rate)) if np.isfinite(rate) and rate > 0 else frame_sigma
        events.append(Event(f"band-1.reaches.{name}", f"Band-1 tip reaches the level of {name}", "soil",
                            float(reach), sigma, name, f"band-1's tip reaching the level of {name}"))
    return events
