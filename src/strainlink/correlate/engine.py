"""The correlation engine: finds, tests and grades links between soil and sensors.

Three layers, all evidence-based (no free-text generation):

1. Physical hypothesis: *pressure cells soften when band-1's tip reaches
   their level* (the reference paper argues this qualitatively, Sec. 5.2).
   Tested across all cells: regression of softening onset on tip arrival
   (slope ~1 expected), an exact permutation test, a comparison with the
   alternative "all cells soften together", and a check that cells the
   tip never reached did not soften.
2. Timing coincidences between every sensor event and every soil event,
   with uncertainty. A single coincidence cannot show cause, so these are
   never graded above "moderate", and a coincidence shared with several
   soil events is marked ambiguous.
3. Data-driven screening of continuous records: every sensor channel
   against every soil measure, using frame-to-frame *changes* (both rise
   with wall push, which would make any pair look correlated), with
   lead/lag search built into a permutation test and false-discovery-rate
   control across all pairs.

A predictive machine-learning layer was tried and removed (2026-09-28):
within one monotonic test of ~60 photos, cross-validated models cannot
extrapolate along wall push and gave meaningless scores. It belongs in
cross-test analysis (many tests), not here. See Decisions.md.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .events import Event
from .findings import Finding, ranked

# Screening settings.
MAX_LAG_FRAMES = 2
PERMUTATIONS = 2000
FDR_LEVEL = 0.05
MIN_FRAMES_FOR_SCREENING = 12
MIN_DISTINCT_CHANGES = 5
# Tip-arrival pushes must differ by at least this much (mm) to regress on them.
MIN_SPREAD_MM = 1e-6
# Coincidence: two events within this many combined standard deviations.
COINCIDENCE_SIGMAS = 2.0
SEED = 20260928


@dataclass
class CorrelationReport:
    events: list[Event]
    findings: list[Finding]
    warnings: list[str] = field(default_factory=list)


def analyse(sensor_events: list[Event], soil_events: list[Event], tip_events: list[Event],
            cell_levels_mm: dict[str, float], frames: pd.DataFrame, sensor_columns: list[str],
            soil_columns: list[str]) -> CorrelationReport:
    """Run all layers.

    frames: one row per photo, with the push (column "push_mm"), sensor
    values sampled at the photo times and soil measures from the band tracks.
    """
    findings: list[Finding] = []
    warnings: list[str] = []
    cell_finding = cell_softening_hypothesis(sensor_events, tip_events, cell_levels_mm)
    if cell_finding is not None:
        findings.append(cell_finding)
    findings.extend(coincidences(sensor_events, soil_events))
    if len(frames) >= MIN_FRAMES_FOR_SCREENING:
        findings.extend(screen(frames, sensor_columns, soil_columns))
    else:
        warnings.append(f"Only {len(frames)} photos: continuous screening needs at least {MIN_FRAMES_FOR_SCREENING}.")
    return CorrelationReport(events=sensor_events + soil_events + tip_events, findings=ranked(findings), warnings=warnings)


# --- layer 1: pressure cells vs band-1 tip ----------------------------------

def cell_softening_hypothesis(sensor_events: list[Event], tip_events: list[Event],
                              cell_levels_mm: dict[str, float]) -> Finding | None:
    onsets = {e.subject: e for e in sensor_events if e.key.endswith(".softening") and e.subject in cell_levels_mm}
    reaches = {e.subject: e for e in tip_events}
    if not cell_levels_mm:
        return None
    paired = sorted(set(onsets) & set(reaches), key=lambda c: cell_levels_mm[c])
    reached_not_softened = sorted(set(reaches) - set(onsets))
    softened_not_reached = sorted(set(onsets) - set(reaches))
    evidence: dict[str, float | int | str] = {
        "cells paired": len(paired),
        "reached, no detectable softening": ", ".join(reached_not_softened) or "none",
        "softened but tip never reached": ", ".join(softened_not_reached) or "none",
    }
    caveats: list[str] = []
    title = "Pressure-cell softening vs band-1 tip"
    method = ("Regression of each cell's softening onset on the push at which band-1's tip reaches the cell's level; "
              "exact permutation test over cell assignments; comparison with a common-onset model.")
    if len(paired) < 2:
        return Finding("cells-vs-tip", title,
                       "Too few pressure cells both softened and were reached by band-1 to test the link.",
                       "not supported", method, evidence, ["Needs at least 2, ideally 3+ cells."])

    onset = np.array([onsets[c].push_mm for c in paired])
    reach = np.array([reaches[c].push_mm for c in paired])
    lag = onset - reach
    lag_mean = float(np.mean(lag))
    lag_se = float(np.std(lag, ddof=1) / math.sqrt(len(lag))) if len(lag) > 1 else float("nan")
    evidence.update({"mean lag (mm of push)": round(lag_mean, 2), "lag standard error (mm)": round(lag_se, 2)})

    strength = "weak"
    if np.ptp(reach) < MIN_SPREAD_MM:
        caveats.append("The tip reached all paired cells between the same two photos: the photo spacing is too "
                       "coarse to test the timing. Take photos more often.")
        return Finding("cells-vs-tip", title,
                       f"Cannot be tested at this photo spacing: band-1's tip reached {', '.join(paired)} between "
                       f"the same two photos.", "not supported", method, evidence, caveats)
    if len(paired) >= 3:
        fit = stats.linregress(reach, onset)
        t = stats.t.ppf(0.975, len(paired) - 2)
        slope_ci = (fit.slope - t * fit.stderr, fit.slope + t * fit.stderr)
        r2 = fit.rvalue**2
        p_perm = _permutation_p(reach, onset)
        # Alternative: all cells soften at one common push (no dependence on the tip).
        rss_tip = float(np.sum((onset - (reach + lag_mean)) ** 2))
        rss_common = float(np.sum((onset - onset.mean()) ** 2))
        evidence.update({"slope": round(fit.slope, 3), "slope 95% CI": f"{slope_ci[0]:.2f} to {slope_ci[1]:.2f}",
                         "R²": round(r2, 3), "permutation p": round(p_perm, 4),
                         "residual (tip model) mm²": round(rss_tip, 3), "residual (common onset) mm²": round(rss_common, 3)})
        slope_ok = slope_ci[0] <= 1.0 <= slope_ci[1] or abs(fit.slope - 1.0) < 0.15
        if slope_ok and r2 >= 0.9 and p_perm < 0.05 and rss_tip < rss_common:
            strength = "strong"
        elif r2 >= 0.7 and p_perm < 0.1 and rss_tip < rss_common:
            strength = "moderate"
        elif r2 < 0.5 or rss_tip >= rss_common:
            strength = "not supported"
        if len(paired) <= 4:
            caveats.append(f"Only {len(paired)} cells: the permutation test cannot give p below "
                           f"{1 / math.factorial(len(paired)):.3f}.")
    else:
        caveats.append("Only 2 cells paired: consistency is shown, but it cannot be tested statistically.")
    if softened_not_reached:
        caveats.append(f"{', '.join(softened_not_reached)} softened although the tip never reached them.")
        if strength == "strong":
            strength = "moderate"
    if strength == "not supported":
        statement = ("Pressure-cell softening does NOT follow band-1's tip reaching each cell: the timing is better "
                     "explained otherwise (see evidence).")
    else:
        digits = 2 if lag_se < 0.1 else 1
        statement = (f"Each pressure cell starts to soften {lag_mean:.{digits}f} ± {lag_se:.{digits}f} mm of wall push after "
                     f"band-1's tip reaches its level ({len(paired)} cells: {', '.join(paired)}).")
        if reached_not_softened:
            statement += (f" {', '.join(reached_not_softened)}: reached, but no softening could be detected above "
                          f"its measurement noise.")
    events = [onsets[c].key for c in paired] + [reaches[c].key for c in paired]
    return Finding("cells-vs-tip", title, statement, strength, method, evidence, caveats, events)


def _permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    """Exact (small n) or Monte-Carlo permutation p-value for |correlation|."""
    observed = abs(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if n <= 7:
        perms = itertools.permutations(range(n))
        values = [abs(np.corrcoef(x, y[list(p)])[0, 1]) for p in perms]
    else:
        rng = np.random.default_rng(SEED)
        values = [abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(PERMUTATIONS)]
    return float(np.mean(np.array(values) >= observed - 1e-12))


# --- layer 2: timing coincidences --------------------------------------------

def coincidences(sensor_events: list[Event], soil_events: list[Event]) -> list[Finding]:
    findings = []
    for s in sensor_events:
        close = []
        for d in soil_events:
            sigma = math.hypot(s.sigma_mm, d.sigma_mm)
            delta = s.push_mm - d.push_mm
            if abs(delta) <= COINCIDENCE_SIGMAS * sigma:
                close.append((abs(delta) / max(sigma, 1e-9), delta, sigma, d))
        if not close:
            continue
        close.sort(key=lambda c: c[0])
        z, delta, sigma, d = close[0]
        ambiguous = len(close) > 1
        strength = "weak" if ambiguous or z > 1.0 else "moderate"
        when = ("at the same wall push as" if abs(delta) < 0.5 * sigma
                else f"{abs(delta):.1f} ± {sigma:.1f} mm of push {'after' if delta > 0 else 'before'}")
        statement = (f"{s.label} {when} {d.phrase or d.label} "
                     f"({s.push_mm:.1f} vs {d.push_mm:.1f} mm).")
        caveats = ["A single timing coincidence does not show cause."]
        if ambiguous:
            others = ", ".join(c[3].label for c in close[1:])
            caveats.append(f"Also coincides with: {others}.")
        findings.append(Finding(f"coincide:{s.key}:{d.key}", f"{s.label} ↔ {d.label}", statement, strength,
                                "Event timing comparison with combined uncertainty (2σ window).",
                                {"sensor event (mm)": round(s.push_mm, 2), "soil event (mm)": round(d.push_mm, 2),
                                 "difference (mm)": round(delta, 2), "combined σ (mm)": round(sigma, 2)},
                                caveats, [s.key, d.key]))
    return findings


# --- layer 3: continuous screening and prediction -----------------------------

def screen(frames: pd.DataFrame, sensor_columns: list[str], soil_columns: list[str]) -> list[Finding]:
    """Rank-correlate frame-to-frame changes, with lag search inside the permutation test."""
    rng = np.random.default_rng(SEED)
    rows = []
    for s_col, d_col in itertools.product(sensor_columns, soil_columns):
        both = frames[[s_col, d_col]].dropna()
        if len(both) < MIN_FRAMES_FOR_SCREENING:
            continue
        ds = np.diff(both[s_col].to_numpy())
        dd = np.diff(both[d_col].to_numpy())
        # Near-constant changes carry no information (and make rank statistics unstable).
        if len(np.unique(np.round(ds, 9))) < MIN_DISTINCT_CHANGES or len(np.unique(np.round(dd, 9))) < MIN_DISTINCT_CHANGES:
            continue
        rho, lag = _best_lag(ds, dd)
        # Null: circular shifts keep each series' own autocorrelation but break alignment.
        null = [abs(_best_lag(ds, np.roll(dd, int(rng.integers(MAX_LAG_FRAMES + 1, len(dd) - MAX_LAG_FRAMES))))[0])
                for _ in range(PERMUTATIONS // 4)]
        p = (1 + np.sum(np.array(null) >= abs(rho))) / (1 + len(null))
        rows.append((s_col, d_col, rho, lag, p, len(ds)))
    if not rows:
        return []
    q = _benjamini_hochberg(np.array([r[4] for r in rows]))
    findings = []
    for (s_col, d_col, rho, lag, p, n), q_value in zip(rows, q):
        if q_value > FDR_LEVEL:
            continue
        strength = "strong" if abs(rho) >= 0.7 else ("moderate" if abs(rho) >= 0.5 else "weak")
        direction = "rises" if rho > 0 else "falls"
        timing = "at the same time" if lag == 0 else (f"{abs(lag)} photo(s) later" if lag > 0 else f"{abs(lag)} photo(s) earlier")
        findings.append(Finding(
            f"screen:{s_col}:{d_col}", f"{s_col} ↔ {d_col} (changes)",
            f"When {d_col} changes faster, {s_col} {direction} faster, {timing} "
            f"(Spearman ρ = {rho:+.2f} on frame-to-frame changes).",
            strength,
            "Spearman correlation of frame-to-frame changes, lag search ±2 photos inside a circular-shift "
            "permutation test; Benjamini–Hochberg false-discovery control across all pairs.",
            {"ρ": round(rho, 3), "lag (photos)": lag, "p": round(p, 4), "q (FDR)": round(q_value, 4), "n changes": n},
            ["Correlation of changes, not proof of cause."], []))
    return findings


def _best_lag(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    best = (0.0, 0)
    for lag in range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1):
        x, y = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
            continue
        rho = stats.spearmanr(x, y).statistic
        if np.isfinite(rho) and abs(rho) > abs(best[0]):
            best = (float(rho), lag)
    return best


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked_q = p[order] * n / np.arange(1, n + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked_q, 0, 1)
    return q
