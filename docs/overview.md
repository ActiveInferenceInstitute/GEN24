# Project overview

## Purpose

GEN24 is an exploratory project that combines source material about entities, organizations, and domains with LLM-assisted generation. The active `GEN-24/` scripts generate a pro-introduction prompt, produce a written introduction letter, translate that letter, and generate audience-specific translations of a call to action.

The repository also includes two separate experimental artifacts:

- `ARCANGEL1_0_2.ipynb`, a Google Colab notebook for an ARC-style cognitive/problem-solving experiment.
- The NHIM module: `NHIM/nhim_inquiry_agent.gnn.md` specifies its inquiry loop in Generalized Notation Notation, `NHIM/nhim_model.py` implements that model in NumPy, and `NHIM/NHIM_API.py` is the original sketch with attention, memory, reinforcement-learning, distress-dynamics, Bayesian-update, and inquiry-processing components.

These artifacts are related by their exploratory cognitive/AI theme but are not invoked by the four-stage GEN-24 pipeline.

## Current contents

The source tree contains:

- Five entity worldview Python modules and entity context Markdown files.
- Two organization worldview Python modules and organization context Markdown files.
- Thirty-one synthetic domain Markdown files.
- Prompt and intermediate/final introduction-letter artifacts.
- 181 translated introduction-letter artifact directories/files.
- 29 call-to-action language directories, each with 11 audience-specific files.
- Eight historical scripts under `GEN-24/Archive/`.

Counts describe the checked-in tree at the time of this review; generated artifacts may change when the pipeline is run.

## Status and boundaries

This is not a packaged Python application: there is no `pyproject.toml` and no command-line interface. The scripts are file-oriented and use paths relative to their own directory in the active stages. They should be treated as exploratory automation, not as a stable API. The repository does have a test suite (`tests/`) and a CI workflow; the NHIM inquiry model additionally has a GNN specification pinned to an upstream parser revision.

The active LLM stages require a local `LLM_keys.key` file and a configured OpenAI-compatible model. Do not place credentials in tracked files. For details, see [the pipeline guide](pipeline.md) and [GNN integration](gnn.md).

## Education contract migration — blocked

The [Education workflow contract](https://github.com/ActiveInferenceInstitute/courses/blob/main/docs/workflows/education.md)
requires a bundle at `specs/education/gen24.json` for `GEN-24/` and `NHIM/`,
with learning objectives quoted verbatim from authored sources. Its schema
requires a nonempty `objectives` array for each unit.

The [pipeline guide](pipeline.md), [pipeline README](../GEN-24/README.md),
[NHIM notes](nhim.md), and [NHIM README](../NHIM/README.md) describe implementation
and operation, not learner objectives. No authored learning objectives were
found for these units. Implementation descriptions and model optimization
goals must not be relabeled as learning objectives.

The bundle and adoption pointer are therefore not emitted. A maintainer must
author learning objectives for both units before a faithful bundle can be
transcribed and validated; no topic mapping or prerequisite relation is inferred.
The independent pin-normalization part is implemented: `NHIM/gnn-pin.json`
uses the lead's `revision`/`tag` spelling, with its validator, tests and CI
consumer updated. The migration PR must remain open until the bundle blocker
is resolved.
