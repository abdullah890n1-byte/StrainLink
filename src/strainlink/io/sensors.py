"""Sensor log import: CSV/TSV/Excel files from data loggers and test controllers.

Lab exports vary (delimiters, header rows, units in column names), so the
reader detects the structure and proposes a *channel map*: which column is
time, actuator force, actuator displacement, LVDTs, uplift and each
pressure cell, and in which unit. The app shows this guess for the user to
confirm. Nothing is assumed silently: unrecognised columns are listed and
every guess carries a confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Role = Literal["time", "force", "displacement", "lvdt", "uplift", "pressure", "ignore"]
# Lines from the end of a text export used to detect its delimiter.
DELIMITER_SAMPLE_LINES = 30

# Unit conversions to StrainLink's base units (s, kN, mm, kPa).
_UNIT_FACTORS = {
    "s": 1.0, "sec": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0,
    "kn": 1.0, "n": 1e-3, "mn": 1e3, "lbf": 4.44822e-3, "kip": 4.44822,
    "mm": 1.0, "m": 1e3, "cm": 10.0, "um": 1e-3, "µm": 1e-3, "in": 25.4,
    "kpa": 1.0, "pa": 1e-3, "mpa": 1e3, "psi": 6.89476, "bar": 100.0,
}
_BASE_UNIT = {"time": "s", "force": "kN", "displacement": "mm", "lvdt": "mm", "uplift": "mm", "pressure": "kPa"}
_UNIT_KIND = {
    "s": "time", "sec": "time", "ms": "time", "min": "time", "h": "time",
    "kn": "force", "n": "force", "mn": "force", "lbf": "force", "kip": "force",
    "mm": "length", "m": "length", "cm": "length", "um": "length", "µm": "length", "in": "length",
    "kpa": "pressure", "pa": "pressure", "mpa": "pressure", "psi": "pressure", "bar": "pressure",
}
# Name patterns (lower case) per role; the first match wins, in this order.
# Word boundaries are written as (?<![a-z]) / (?![a-z]) because lab exports
# join words with underscores ("time_s"), which \b treats as word characters.
_ROLE_PATTERNS: list[tuple[Role, str]] = [
    ("time", r"(?<![a-z])(time|t|elapsed|seconds?)(?![a-z])"),
    ("uplift", r"(uplift|vertical|lvdt[_ ]?v|vert)"),
    ("pressure", r"(^|[^a-z])(pt|pc|epc|p)\s*_?\d+|pressure|earth\s*press"),
    ("lvdt", r"lvdt|ldvt|dial|transducer\s*\d"),
    ("force", r"force|load|actuator[_ ]?f|(?<![a-z])kn(?![a-z])"),
    ("displacement", r"disp|stroke|position|push|travel|actuator[_ ]?d"),
]


@dataclass
class Channel:
    column: str
    role: Role
    unit: str               # unit found in the file (as written)
    factor: float           # multiply file values by this to get base units
    name: str               # short display name, e.g. "PT3"
    confidence: float       # 0..1, how sure the automatic mapping is


@dataclass
class SensorTable:
    path: Path
    frame: pd.DataFrame
    channels: list[Channel]
    unmapped: list[str] = field(default_factory=list)

    def values(self, channel: Channel) -> np.ndarray:
        return pd.to_numeric(self.frame[channel.column], errors="coerce").to_numpy(dtype=float) * channel.factor

    def by_role(self, role: Role) -> list[Channel]:
        return [c for c in self.channels if c.role == role]

    @property
    def time_s(self) -> np.ndarray | None:
        time = self.by_role("time")
        return self.values(time[0]) if time else None


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV/TXT/XLSX export, detecting the delimiter and the header row."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        raw = pd.read_excel(path, header=None)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        delimiter = _detect_delimiter(text)
        raw = pd.read_csv(path, header=None, sep=delimiter, engine="python", skip_blank_lines=True,
                          on_bad_lines="skip", skiprows=_table_start(text, delimiter))
    header_row = _header_row(raw)
    frame = raw.iloc[header_row + 1:].reset_index(drop=True)
    frame.columns = _unique([str(c).strip() for c in raw.iloc[header_row]])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    return frame


def _detect_delimiter(text: str) -> str:
    """The delimiter giving the most, and most consistent, fields on the data lines.

    Judged on the last lines of the file: logger exports often start with
    free-text preamble lines that would mislead a sniffer.
    """
    lines = [line for line in text.splitlines() if line.strip()][-DELIMITER_SAMPLE_LINES:]
    best, best_score = r"\s+", 1.0
    for candidate in (",", ";", "\t", "|"):
        counts = np.array([line.count(candidate) + 1 for line in lines])
        if counts.min() < 2:
            continue
        consistency = np.mean(counts == np.median(counts))
        score = np.median(counts) * consistency
        if score > best_score:
            best, best_score = candidate, score
    return best


def _table_start(text: str, delimiter: str) -> int:
    """Index of the first line with the data's field count (skips preamble lines).

    pandas sizes the table from its first line, so a one-field preamble line
    would make it discard every real row.
    """
    lines = text.splitlines()
    split = (lambda line: re.split(delimiter, line.strip())) if delimiter == r"\s+" else (lambda line: line.split(delimiter))
    counts = [len(split(line)) if line.strip() else 0 for line in lines]
    data = [c for c in counts[-DELIMITER_SAMPLE_LINES:] if c]
    if not data:
        return 0
    target = int(np.median(data))
    return next((i for i, c in enumerate(counts) if c == target), 0)


def _header_row(raw: pd.DataFrame) -> int:
    """The last mostly-text row before the numeric data starts (loggers often add preamble lines)."""
    numeric = raw.apply(pd.to_numeric, errors="coerce").notna().mean(axis=1).to_numpy()
    first_data = int(np.argmax(numeric > 0.5)) if (numeric > 0.5).any() else 1
    return max(first_data - 1, 0)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _unit_of(column: str) -> str | None:
    match = re.search(r"[\[(]\s*([A-Za-zµ]+)\s*[\])]|[_ ]([A-Za-zµ]+)$", column)
    if not match:
        return None
    unit = (match.group(1) or match.group(2)).lower()
    return unit if unit in _UNIT_FACTORS else None


def _short_name(column: str, role: Role, index: int) -> str:
    number = re.search(r"(\d+)", column)
    if role == "pressure":
        return f"PT{number.group(1) if number else index}"
    if role == "lvdt":
        return f"LVDT{number.group(1) if number else index}"
    return {"time": "Time", "force": "Force", "displacement": "Displacement", "uplift": "Uplift"}.get(role, column)


def guess_channels(frame: pd.DataFrame) -> tuple[list[Channel], list[str]]:
    """Propose a role and unit for every column, with a confidence."""
    channels, unmapped = [], []
    counters: dict[Role, int] = {}
    for column in frame.columns:
        lower = column.lower()
        unit = _unit_of(column)
        role: Role | None = None
        for candidate, pattern in _ROLE_PATTERNS:
            if re.search(pattern, lower):
                role = candidate
                break
        confidence = 0.8 if role else 0.0
        kind = _UNIT_KIND.get(unit) if unit else None
        if role is None and kind:
            role = {"time": "time", "force": "force", "pressure": "pressure", "length": "displacement"}[kind]
            confidence = 0.5
        if role is None:
            unmapped.append(column)
            continue
        expected = {"time": "time", "force": "force", "pressure": "pressure"}.get(role, "length")
        if kind and kind == expected:
            confidence = min(1.0, confidence + 0.2)
        elif kind and kind != expected:
            confidence *= 0.5          # name and unit disagree: ask the user
        if unit is None or _UNIT_KIND.get(unit) != expected:
            unit = _BASE_UNIT[role]
        counters[role] = counters.get(role, 0) + 1
        channels.append(Channel(column=column, role=role, unit=unit, factor=_UNIT_FACTORS[unit.lower()],
                                name=_short_name(column, role, counters[role]), confidence=confidence))
    # Only one time, force and actuator-displacement channel: keep the most confident.
    for single in ("time", "force", "displacement"):
        same = [c for c in channels if c.role == single]
        for extra in sorted(same, key=lambda c: -c.confidence)[1:]:
            extra.role = "lvdt" if single == "displacement" else "ignore"
    return channels, unmapped


def load_sensors(path: str | Path) -> SensorTable:
    frame = read_table(path)
    channels, unmapped = guess_channels(frame)
    return SensorTable(path=Path(path), frame=frame, channels=channels, unmapped=unmapped)
