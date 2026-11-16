"""Train the failure-zone model on the training scenes (SYNTHETIC DATA) and save it.

    .venv/bin/python scripts/ml_train.py

Two kinds of model are compared on validation scenes held back from training
(a gradient-boosted tree ensemble and a small neural network); the better one
is saved to src/strainlink/ml/zone_model.joblib with a record of how it was
trained. The decision threshold is chosen on the validation scenes too: the
lowest probability at which no more than MAX_FALSE_ALARM_SHARE of the scenes
without any zone would raise an alarm. The test scenes are not used here at
all (scripts/validate_ml.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from strainlink import __version__
from strainlink.ml import features
from strainlink.ml.model import THRESHOLD, ZoneModel, zones

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "ml" / "train"
VALIDATION_SCENES = 100          # the last ones of the training seeds, held back for choosing the model
POINTS_PER_SCENE = 2500          # sampled per scene (all zone points kept), so every scene counts
SEED = 20261116
# A false alarm in a sound specimen costs trust: at most this share of zone-free scenes may raise one.
MAX_FALSE_ALARM_SHARE = 0.05
THRESHOLD_CANDIDATES = np.round(np.arange(0.30, 0.991, 0.01), 2)


def choose_threshold(model: ZoneModel, paths: list[Path]) -> dict:
    """Lowest threshold meeting the false-alarm limit on validation scenes, with the zones it then finds."""
    clean, cracked = [], []
    for path in paths:
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        prob = np.full(d["have"].shape, np.nan)
        if len(d["X"]):                     # a scene can lose every point (e.g. a heavily compressed photo)
            prob[d["have"]] = model.estimator.predict_proba(d["X"])[:, 1]
        (cracked if meta["zones"] else clean).append((prob, d["truth_grid"]))
    for t in THRESHOLD_CANDIDATES:
        alarms = sum(bool(zones(p, t)[1]) for p, _ in clean)
        if alarms <= MAX_FALSE_ALARM_SHARE * len(clean):
            break
    found = [float(zones(p, t)[0][truth == 1].mean()) >= 0.5 for p, truth in cracked if (truth == 1).sum() >= 10]
    return {"threshold": float(t), "zone_free_scenes": len(clean), "false_alarm_scenes": int(alarms),
            "detectable_zone_scenes": len(found), "found": int(sum(found))}


def load(paths: list[Path], sample: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    xs, ys = [], []
    for path in paths:
        d = np.load(path)
        keep = d["label"] >= 0                                  # don't-care points are not used
        X, y = d["X"][keep], d["label"][keep]
        if sample and len(y) > POINTS_PER_SCENE:
            zone = np.flatnonzero(y == 1)
            intact = np.flatnonzero(y == 0)
            take = np.concatenate([zone, rng.choice(intact, size=min(len(intact), max(POINTS_PER_SCENE - len(zone), 0)), replace=False)])
            X, y = X[take], y[take]
        xs.append(X); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    paths = sorted(TRAIN.glob("scene_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    fit_paths, val_paths = paths[:-VALIDATION_SCENES], paths[-VALIDATION_SCENES:]
    X, y = load(fit_paths, sample=True)
    Xv, yv = load(val_paths, sample=False)
    print(f"training points {len(y)} ({y.mean():.1%} in zones) from {len(fit_paths)} scenes; "
          f"validation points {len(yv)} from {len(val_paths)} scenes")
    weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))       # balance the two classes
    candidates = {
        "gradient-boosted trees": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=48,
                                                                 l2_regularization=1.0, early_stopping=True,
                                                                 validation_fraction=0.1, random_state=SEED),
        "neural network (2 hidden layers)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                                         early_stopping=True, max_iter=200,
                                                                                         random_state=SEED)),
    }
    results = {}
    for name, estimator in candidates.items():
        started = time.perf_counter()
        if "neural" in name:
            # The network takes no sample weights: balance by resampling the zone points instead.
            rng = np.random.default_rng(SEED)
            zone, intact = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            take = np.concatenate([rng.choice(zone, size=len(intact), replace=True), intact])
            estimator.fit(X[take], y[take])
        else:
            estimator.fit(X, y, sample_weight=weight)
        p = estimator.predict_proba(Xv)[:, 1]
        results[name] = {"average_precision": float(average_precision_score(yv, p)),
                         "f1_at_threshold": float(f1_score(yv, p >= THRESHOLD)),
                         "seconds": time.perf_counter() - started}
        print(f"{name:34s} validation: average precision {results[name]['average_precision']:.3f}, "
              f"F1 {results[name]['f1_at_threshold']:.3f}, {results[name]['seconds']:.0f} s")
    best = max(results, key=lambda n: results[n]["average_precision"])
    model = ZoneModel(candidates[best], {})
    choice = choose_threshold(model, val_paths)
    print(f"threshold {choice['threshold']}: false alarms in {choice['false_alarm_scenes']} of {choice['zone_free_scenes']} "
          f"zone-free validation scenes; found {choice['found']} of {choice['detectable_zone_scenes']} zones")
    model.info = {"model": best, "features": list(features.NAMES), "trained_with": f"StrainLink {__version__}",
                  "training_scenes": len(fit_paths), "validation_scenes": len(val_paths), "training_points": int(len(y)),
                  "threshold": choice["threshold"], "threshold_choice": choice, "validation": results,
                  "data": "SYNTHETIC scenes (strainlink.ml.scenes.random_scene), measured with StrainLink's DIC"}
    model.save()
    print(f"saved: {best}")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
