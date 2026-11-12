# Documentation index

Flat index of the repository's long-lived documentation. The narrative hub is
[`README.md`](README.md); this page exists so every page has one place that
links it, and `scripts/check_docs_links.py` fails when a page is missing from
it.

## Start here

- [`../README.md`](../README.md) — repository overview, layout, requirements.
- [`overview.md`](overview.md) — purpose, scope, and current repository contents.
- [`RUN_GUIDE.md`](RUN_GUIDE.md) — the gate entry point, its arms, the CI
  mapping, and the measured cost of each arm.

## Project documentation

- [`pipeline.md`](pipeline.md) — the four active `GEN-24/` stages, configuration,
  outputs, and limitations.
- [`data-layout.md`](data-layout.md) — how source material and generated
  artifacts are organized.
- [`nhim.md`](nhim.md) — status and runtime considerations for the NHIM model and
  `NHIM/NHIM_API.py`.
- [`gnn.md`](gnn.md) — the GNN specification of the NHIM inquiry agent, the
  upstream parser pin, and the validator.
- [`notebook.md`](notebook.md) — what `ARCANGEL1_0_2.ipynb` does and what
  environment it expects.

## Generated facts

- [`_generated/README.md`](_generated/README.md) — what lives in `_generated/`
  and which script produces it.
- [`_generated/GATE_METRICS.md`](_generated/GATE_METRICS.md) — arm list,
  collected test count, and per-arm wall times, produced by
  [`../scripts/gate_metrics.py`](../scripts/gate_metrics.py).

## Repository-level documents

- [`../AGENTS.md`](../AGENTS.md) — agent notes and the gate list.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — contribution rules.
- [`../SECURITY.md`](../SECURITY.md) — security policy.
- [`../CITATION.cff`](../CITATION.cff) — citation metadata.
- [`../TO-DO.md`](../TO-DO.md) — open and completed documentation work.
- [`../REVIEW_LOG_2026-08-02.md`](../REVIEW_LOG_2026-08-02.md) — point-in-time
  review record.

## Folder guides

- [`../GEN-24/Inputs_and_Outputs/README.md`](../GEN-24/Inputs_and_Outputs/README.md)
  — folder-by-folder reference for source material and generated artifacts.
- [`../GEN-24/Archive/README.md`](../GEN-24/Archive/README.md) — the historical,
  non-runnable scripts.
- [`../NHIM/README.md`](../NHIM/README.md) — the NHIM module.
- [`../scripts/README.md`](../scripts/README.md) — the validator and the gate
  helper scripts.
- [`../tests/README.md`](../tests/README.md) — the offline test suite.
- [`../.github/README.md`](../.github/README.md) — workflows and the GNN
  conformance pointer.
