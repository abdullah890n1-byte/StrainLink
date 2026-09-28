"""Findings: what the correlation engine concluded, graded, with the evidence attached.

Statements are generated from the computed numbers by fixed templates;
nothing is free-written, so a finding can never claim more than its
evidence. "Not supported" is a first-class outcome: it tells the engineer
that a relationship one might expect is *not* in the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Strength = Literal["strong", "moderate", "weak", "not supported"]
STRENGTH_ORDER = {"strong": 0, "moderate": 1, "weak": 2, "not supported": 3}


@dataclass
class Finding:
    key: str
    title: str
    statement: str
    strength: Strength
    method: str
    evidence: dict[str, float | int | str] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "key": self.key, "title": self.title, "statement": self.statement, "strength": self.strength,
            "method": self.method, "evidence": self.evidence, "caveats": self.caveats, "events": self.events,
        }


def ranked(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: STRENGTH_ORDER[f.strength])
