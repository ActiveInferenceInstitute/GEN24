# Project overview

## Purpose

GEN24 is an exploratory project that combines source material about entities, organizations, and domains with LLM-assisted generation. The active `GEN-24/` scripts generate a pro-introduction prompt, produce a written introduction letter, translate that letter, and generate audience-specific translations of a call to action.

The repository also includes two separate experimental artifacts:

- `ARCANGEL1_0_2.ipynb`, a Google Colab notebook for an ARC-style cognitive/problem-solving experiment.
- `NHIM/NHIM_API.py`, a large experimental Python module containing attention, memory, reinforcement-learning, distress-dynamics, Bayesian-update, and inquiry-processing components.

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

This is not a packaged Python application: there is no `pyproject.toml`, test suite, command-line interface, or CI configuration. The scripts are file-oriented and use paths relative to their own directory in the active stages. They should be treated as exploratory automation, not as a stable API.

The active LLM stages require a local `LLM_keys.key` file and a configured OpenAI-compatible model. Do not place credentials in tracked files. For details, see [the pipeline guide](pipeline.md).
