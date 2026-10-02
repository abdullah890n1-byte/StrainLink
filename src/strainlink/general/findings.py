"""Findings about the object itself: how its zones form and develop (no logger needed).

The correlation engine links the object to sensor records. Even without a
logger, the object's own record carries evidence worth stating: the order
in which localisation zones form, and whether each zone keeps opening or
sliding. The same rules apply: every statement is computed from the data
by a fixed template, with its evidence and a grade.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..correlate.events import Event
from ..correlate.findings import Finding
from .events import MODE_NAMES

# Growth needs at least this many measured photos after the zone appears.
MIN_GROWTH_PHOTOS = 5
# Spearman rank correlation of the zone's opening/slip with the test axis:
# at least STRONG_TREND for "strong", MODERATE_TREND for "moderate".
STRONG_TREND = 0.9
MODERATE_TREND = 0.7
# Two zones "form together" within this many combined standard deviations.
TOGETHER_SIGMAS = 2.0
# Clearly ordered beyond this many combined standard deviations.
ORDERED_SIGMAS = 3.0


def _digits(scale: float) -> int:
    if not np.isfinite(scale) or scale <= 0:
        return 1
    return int(min(max(1 - math.floor(math.log10(scale)), 0), 4))


def formation_order(object_events: list[Event], axis: tuple[str, str]) -> list[Finding]:
    """Which zones form together and which form first."""
    name, unit = axis
    onsets = sorted((e for e in object_events if e.key.endswith(".onset")), key=lambda e: e.push_mm)
    findings = []
    for i, a in enumerate(onsets):
        for b in onsets[i + 1:]:
            sigma = math.hypot(a.sigma_mm, b.sigma_mm)
            delta = b.push_mm - a.push_mm
            d = _digits(sigma)
            evidence = {f"{a.subject} ({unit})": round(a.push_mm, d + 1), f"{b.subject} ({unit})": round(b.push_mm, d + 1),
                        f"difference ({unit})": round(delta, d + 1), f"combined σ ({unit})": round(sigma, d + 1)}
            if abs(delta) <= TOGETHER_SIGMAS * sigma:
                statement = f"{a.label.replace(' appears', '')} and {b.label.replace(' appears', '')} form at the same {name} ({a.push_mm:.{d}f} vs {b.push_mm:.{d}f} {unit})."
                strength = "moderate"
            else:
                statement = (f"{a.label.replace(' appears', '')} forms first; {b.label.replace(' appears', '')} follows "
                             f"{delta:.{d}f} ± {sigma:.{d}f} {unit} of {name} later.")
                strength = "strong" if delta > ORDERED_SIGMAS * sigma else "moderate"
            findings.append(Finding(f"order:{a.subject}:{b.subject}", f"{a.subject} ↔ {b.subject} (formation)", statement,
                                    strength, "Onset timing comparison with combined uncertainty (half the photo spacing each).",
                                    evidence, ["Forming together does not show that one zone caused the other."],
                                    [a.key, b.key]))
    return findings


def zone_growth(zones: list[dict], series: dict[str, np.ndarray], axis_values: np.ndarray,
                axis: tuple[str, str]) -> list[Finding]:
    """For each zone with a confirmed onset: does its opening (crack) or slip (shear band) keep growing?"""
    name, unit = axis
    findings = []
    for zone in zones:
        if zone.get("onset") is None or not np.isfinite(zone["onset"]):
            continue
        what = "slip" if zone["mode"] == "shear" else "opening"
        values = series.get(f"{zone['name']} {what} (mm)")
        if values is None:
            continue
        ok = np.isfinite(values) & np.isfinite(axis_values) & (axis_values >= zone["onset"])
        if ok.sum() < MIN_GROWTH_PHOTOS:
            continue
        a, v = axis_values[ok], np.abs(values[ok])
        rho = float(stats.spearmanr(a, v).statistic)
        slope = float(np.polyfit(a, v, 1)[0])
        kind = MODE_NAMES.get(zone["mode"], "zone")
        if rho >= MODERATE_TREND:
            statement = (f"{zone['name']} ({kind}) keeps {'sliding' if what == 'slip' else 'opening'} after it forms: "
                         f"its {what} grows from {v[0]:.3f} to {v[-1]:.3f} mm "
                         f"({slope:.3g} mm per {unit} of {name}).")
            strength = "strong" if rho >= STRONG_TREND else "moderate"
        elif rho <= -MODERATE_TREND:
            statement = f"{zone['name']} ({kind}) closes after it forms: its {what} falls from {v[0]:.3f} to {v[-1]:.3f} mm."
            strength = "strong" if rho <= -STRONG_TREND else "moderate"
        else:
            statement = (f"{zone['name']} ({kind}) does not keep growing steadily after it forms "
                         f"({what} between {v.min():.3f} and {v.max():.3f} mm).")
            strength = "weak"
        findings.append(Finding(f"growth:{zone['name']}", f"{zone['name']} after it forms", statement, strength,
                                f"Spearman rank correlation of the zone's {what} with {name} over the photos after its onset; "
                                "slope by least squares.",
                                {"photos": int(ok.sum()), "Spearman ρ": round(rho, 3), f"slope (mm per {unit})": round(slope, 5),
                                 f"first {what} (mm)": round(float(v[0]), 4), f"last {what} (mm)": round(float(v[-1]), 4)},
                                ["Describes the zone's own record; it says nothing about loads without a logger file."],
                                [f"{zone['name']}.onset"]))
    return findings
