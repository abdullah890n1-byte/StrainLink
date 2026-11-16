"""Build the failure-zone model's data sets: random scenes measured with StrainLink's DIC (SYNTHETIC DATA).

    .venv/bin/python scripts/ml_build_dataset.py train 0 600
    .venv/bin/python scripts/ml_build_dataset.py test 10000 200

Each scene is saved as data/ml/<set>/scene_<seed>.npz: the features of every
measured point, its true label (1 zone, 0 intact, -1 don't care), the true
jump size there, and a description of the scene. Seeds of the two sets never
overlap, so the test scenes are unseen by the model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from strainlink.ml.features import compute
from strainlink.ml.measure import measure
from strainlink.ml.scenes import random_scene

ROOT = Path(__file__).resolve().parents[1]


def describe(scene) -> dict:
    return {"hole": scene.hole is not None, "noise_grey": scene.noise_grey, "blur_px": scene.blur_px,
            "jpeg_quality": scene.jpeg_quality,
            "zones": [{"mode": z.mode, "jump_px": float(np.hypot(*z.jump)), "width_px": z.width,
                       "crosses": bool(z.length > 0.9 * np.hypot(*scene.shape))} for z in scene.zones]}


def main() -> None:
    name, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = ROOT / "data" / "ml" / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for seed in range(first, first + count):
        path = out / f"scene_{seed}.npz"
        if path.exists():
            continue
        scene = random_scene(seed)
        m = measure(scene)
        X, have = compute(m["u"], m["v"], m["valid"], m["step_px"])
        # The fields are kept too: validation scenes choose the decision threshold by whole-scene false
        # alarms, and test scenes score the rule-based detector on the same data.
        extra = {"u": m["u"], "v": m["v"], "valid": m["valid"], "have": have, "truth_grid": m["truth"],
                 "jump_grid": m["jump_px"], "step_px": m["step_px"], "noise": m["noise"]}
        np.savez_compressed(path, X=X, label=m["truth"][have], jump=m["jump_px"][have].astype(np.float32),
                            meta=json.dumps(describe(scene)), **extra)
        done = seed - first + 1
        if done % 25 == 0:
            print(f"{name}: {done}/{count} scenes, {time.perf_counter() - started:.0f} s", flush=True)
    print(f"{name}: finished, {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
