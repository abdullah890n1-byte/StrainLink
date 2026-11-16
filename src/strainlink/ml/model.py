"""The failure-zone model: trained on measured synthetic scenes, used on any analysis.

For every measurement point it gives the probability that the point lies in
a failure zone (a crack or shear band, forming or formed). Points above the
decision threshold that hang together, and are long enough, are reported as
zones. The model is an add-on: analyses without it are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import features

MODEL_FILE = Path(__file__).with_name("zone_model.joblib")
# A point is called "in a zone" above this probability, unless the model carries its own threshold
# (scripts/ml_train.py chooses one on validation scenes for a low false-alarm rate).
THRESHOLD = 0.5
# A zone must span at least this many grid steps (a line, not a speck).
MIN_ZONE_SPAN_STEPS = 8


@dataclass
class ZoneModel:
    estimator: object
    info: dict

    @property
    def threshold(self) -> float:
        return float(self.info.get("threshold", THRESHOLD))

    def zones(self, probability: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        return zones(probability, self.threshold)

    def probability(self, u_px: np.ndarray, v_px: np.ndarray, valid: np.ndarray, step_px: float) -> np.ndarray:
        """Zone probability at every grid point (NaN where there is no measurement)."""
        X, have = features.compute(u_px, v_px, valid, step_px)
        out = np.full(u_px.shape, np.nan)
        if have.any():
            out[have] = self.estimator.predict_proba(X)[:, 1]
        return out

    def save(self, path: Path = MODEL_FILE) -> None:
        import joblib
        joblib.dump({"estimator": self.estimator, "info": self.info}, path, compress=3)


def load(path: Path = MODEL_FILE) -> ZoneModel:
    import joblib
    saved = joblib.load(path)
    if list(saved["info"].get("features", [])) != list(features.NAMES):
        raise ValueError("the saved failure-zone model was trained on different features: retrain it "
                         "(scripts/ml_train.py)")
    return ZoneModel(saved["estimator"], saved["info"])


def zones(probability: np.ndarray, threshold: float = THRESHOLD) -> tuple[np.ndarray, list[dict]]:
    """Group the points above the threshold into zones; drop specks too short to be a zone.

    Returns the zone mask (bool grid) and, per zone, its number of points, span (grid steps) and
    mean probability.
    """
    above = np.nan_to_num(probability, nan=0.0) >= threshold
    labels, n = ndimage.label(above, structure=np.ones((3, 3)))
    mask = np.zeros(above.shape, dtype=bool)
    found = []
    for index in range(1, n + 1):
        member = labels == index
        rows, cols = np.nonzero(member)
        span = float(np.hypot(np.ptp(rows), np.ptp(cols)))
        if span < MIN_ZONE_SPAN_STEPS:
            continue
        mask |= member
        found.append({"points": int(member.sum()), "span_steps": span,
                      "mean_probability": float(np.nanmean(probability[member]))})
    return mask, found


def describe(model: ZoneModel) -> str:
    return json.dumps(model.info, indent=1)
