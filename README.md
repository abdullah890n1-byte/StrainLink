# StrainLink

Geotechnical digital image correlation (DIC) with sensor fusion and an
AI correlation engine. It links what the soil does (displacement, strain,
shear bands from images) to what the sensors record (force, displacement,
earth pressure) during the same model test.

Status (2026-09-29): first complete version.
- **Desktop app for macOS:** folder-first wizard, results viewer and
  shareable report.
- **Engine:** DIC → strain → shear bands (detection, labelling, tracking,
  measurements) → sensor import and clock sync → engineering results →
  AI correlation engine with graded, evidence-backed findings.
- **Validation:** end-to-end on synthetic tests with known answers (see
  Validation.md in the docs).

Runs natively on macOS (Apple Silicon), as well as Windows and Linux.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python scripts/validate_dic_challenge.py    # needs data/external/dic_challenge_1
.venv/bin/python scripts/validate_band_detection.py   # synthetic band-detection sweep
.venv/bin/python scripts/make_mac_app.py              # builds StrainLink.app (development build)
.venv/bin/strainlink app                              # or open the app from the command line
```

Documentation lives in the shared vault:
`/Users/neo/Documents/Documentation/StrainLink/`. Start with
`WorkingPrinciple.md` (how it works, vs Vic-2D/3D), `ProblemStatement.md`
and `Validation.md`.
