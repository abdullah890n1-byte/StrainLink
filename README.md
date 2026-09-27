# StrainLink

Geotechnical digital image correlation (DIC) with sensor fusion and an
AI correlation engine. It links what the soil does (displacement, strain,
shear bands from images) to what the sensors record (force, displacement,
earth pressure) during the same model test.

Status (2026-09-28): DIC engine v0.1 implemented and validated on the DIC
Challenge 1.0 benchmark. Band detection, sensor sync and the correlation
engine are next.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python scripts/validate_dic_challenge.py   # needs data/external/dic_challenge_1
```

Documentation lives in the shared vault:
`/Users/neo/Documents/Documentation/StrainLink/`. Start with
`ProblemStatement.md` and `Vision.md`.
