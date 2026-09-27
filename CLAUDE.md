# StrainLink — Engineering Constitution (Operating Rules)

StrainLink is a separate project from SnapSort and every other project on
this machine. It has its own git repo (this folder) and its own
documentation folder in the shared Obsidian vault at
`/Users/neo/Documents/Documentation/StrainLink/` (the vault is its own git
repo). Links below are relative to this repo.

Treat this file as binding for all work in this repo.

## Mission

Build research-grade software for geotechnical digital image correlation
(DIC) that goes beyond existing DIC packages (Vic-2D, GeoPIV-RG, Ncorr,
DICe). The main thing it adds is an AI-driven correlation engine that links
full-field image data (displacement, strain, shear bands) to the physical
sensor data (force, displacement, earth pressure) recorded during the same
test. Details: [Vision.md](../Documentation/StrainLink/Vision.md) and
[ProblemStatement.md](../Documentation/StrainLink/ProblemStatement.md).

## Scientific integrity rules (non-negotiable)

- Never fabricate data or results. Synthetic data must always be labelled
  synthetic, in file names, metadata and reports.
- Label every statement taken from literature as a reference, with the
  source. Never present a paper's findings as StrainLink's results.
- Every correlation the AI engine reports must carry its evidence
  (statistic, effect size, uncertainty, the data it came from). If the
  evidence is weak, say so. "No reliable correlation found" is a valid
  output.
- The DIC engine must be validated against ground truth (DIC Challenge
  images, synthetic images with known displacement fields) before its
  output is trusted, and the validation results are documented.
- Record the provenance of every external dataset (source URL, licence,
  download date) in [Datasets.md](../Documentation/StrainLink/Datasets.md).

## Definition of done

A task is incomplete unless ALL of the following hold:
- Code runs, and tests pass (`pytest`).
- No obvious bugs remain.
- Relevant docs in the vault are updated.
- `Progress.md` and `Changelog.md` are updated.
- `Architecture.md` is updated if the change affects architecture.
- Decisions with trade-offs are recorded in `Decisions.md`.

## Documentation (single source of truth: `../Documentation/StrainLink/`)

| File | Purpose |
|---|---|
| Vision.md | Product vision and principles |
| ProblemStatement.md | The research problem, from the reference paper |
| Roadmap.md | Milestones and planned direction |
| Features.md | Feature inventory and status |
| Architecture.md | System architecture and modules |
| DataModel.md | Data formats, units, coordinate conventions |
| API.md | Public Python API and CLI contracts |
| Datasets.md | External and synthetic datasets, with provenance |
| Validation.md | Accuracy results against ground truth |
| Progress.md | Dated log entry per completed feature |
| Changelog.md | Dated log of every meaningful change |
| Bugs.md | Bug, root cause, fix, prevention |
| Decisions.md | Why, benefits, drawbacks, alternatives |
| Tasks.md | Active/pending tasks |

Append history rather than overwriting it. Update docs in the same working
session as the code change.

## Development principles

- Plan before coding; explain non-trivial plans briefly first.
- Modular, reusable, readable; never duplicate logic.
- No placeholder code unless explicitly requested.
- Comments explain WHY, not WHAT. No magic numbers; name the constants and
  state their units.
- Units are explicit everywhere (mm, px, kN, kPa, s). Pixel↔mm conversion
  lives in one place only.
- Numerical code gets tests against analytical or known answers.

## Security and privacy

Never hardcode secrets or API keys (the optional LLM layer reads its key
from the environment). User experimental data stays local by default;
nothing is sent to an external service unless the user turns that on.

## Git practices

Clear commit messages, related changes grouped, no unrelated changes in
one commit. Raw datasets are not committed (see `.gitignore`).

## Workflow for every task

1. Understand the request. 2. Review relevant vault docs. 3. Plan.
4. Explain the plan briefly. 5. Implement. 6. Verify (tests, validation).
7. Update docs. 8. Update Progress.md and Changelog.md. 9. Summarize.
10. Suggest the next logical task.
