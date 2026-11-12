# GEN24 documentation

This directory is the documentation hub for the GEN24 repository. The flat
list of every page is [`documentation-index.md`](documentation-index.md).

## Project documentation

- [Project overview](overview.md) — purpose, scope, and current repository contents.
- [Pipeline guide](pipeline.md) — the four active GEN-24 stages, configuration, outputs, and limitations.
- [Data layout](data-layout.md) — how source material and generated artifacts are organized.
- [NHIM notes](nhim.md) — status and runtime considerations for the NHIM model and `NHIM/NHIM_API.py`.
- [GNN integration](gnn.md) — the GNN specification of the NHIM inquiry agent, the upstream parser pin, and the validator.
- [Notebook notes](notebook.md) — what `ARCANGEL1_0_2.ipynb` does and what environment it expects.

## Running the gate

- [Gate and CI run guide](RUN_GUIDE.md) — the `./run.sh` arms, the environment,
  the CI job mapping, and the measured cost of each arm.
- [Generated facts](_generated/README.md) — what is generated under
  `docs/_generated/` and which script produces it.
- [Gate metrics](_generated/GATE_METRICS.md) — arm list, collected test count,
  and per-arm wall times, refreshed by `scripts/gate_metrics.py`.

## Repository-level guidance

- [Full documentation index](documentation-index.md)
- [README](../README.md)
- [Agent notes](../AGENTS.md)
- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)
- [Citation](../CITATION.cff)
- [Review log](../REVIEW_LOG_2026-08-02.md)
- [Open/completed documentation work](../TO-DO.md)

The generated Markdown collections are documented in [GEN-24/Inputs_and_Outputs/README.md](../GEN-24/Inputs_and_Outputs/README.md). Historical scripts are documented in [GEN-24/Archive/README.md](../GEN-24/Archive/README.md).
