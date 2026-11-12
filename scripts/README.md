# `scripts/` — validators and gate helpers

- `validate_gnn_spec.py` validates `NHIM/nhim_inquiry_agent.gnn.md` with the
  pinned upstream parser and checks it against `NHIM/nhim_model.py`. It
  resolves the GNN checkout from `--gnn-root`, `$GNN_ROOT`, or the sibling
  `../Generalized_Notation_Notation`. Without a checkout it prints `SKIPPED`
  unless `--require-gnn` is passed; without `--require-tag` a missing release
  tag is reported as unchecked rather than failing. This is the `gnn` arm.
- `check_docs_links.py` checks that every relative link in a tracked `*.md`
  file resolves, that `docs/README.md` links every page under `docs/`, and that
  `docs/documentation-index.md` links every page under `docs/` at any depth.
- `gate_metrics.py` generates `docs/_generated/GATE_METRICS.md` (arm list,
  collected test count, per-arm wall times) and, with `--check`, fails when the
  committed facts block is stale. Both are part of the `docs` arm.

Every arm is invoked through [`../run.sh`](../run.sh); see
[`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md). Parent guide:
[`../AGENTS.md`](../AGENTS.md).
