# StrainLink

Geotechnical digital image correlation (DIC) with sensor fusion and an
AI correlation engine. It links what the soil does (displacement, strain,
shear bands from images) to what the sensors record (force, displacement,
earth pressure) during the same model test.

Status (2026-09-28):
- **DIC engine v0.1:** validated on the DIC Challenge 1.0 benchmark.
- **Strain module:** validated analytically.
- **Automatic shear-band detection v0.1:** detects, labels and measures
  lift-off, shape, length, area and heave. Validated on synthetic
  retaining-wall experiments with known ground truth.
- **Next:** sensor sync and the AI correlation engine.

Runs natively on macOS (Apple Silicon), as well as Windows and Linux.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python scripts/validate_dic_challenge.py    # needs data/external/dic_challenge_1
.venv/bin/python scripts/validate_band_detection.py   # synthetic band-detection sweep
```

Documentation lives in the shared vault:
`/Users/neo/Documents/Documentation/StrainLink/`. Start with
`WorkingPrinciple.md` (how it works, vs Vic-2D/3D), `ProblemStatement.md`
and `Validation.md`.
