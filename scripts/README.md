# `scripts/` — validators

`validate_gnn_spec.py` validates `NHIM/nhim_inquiry_agent.gnn.md` with the
pinned upstream parser and checks it against `NHIM/nhim_model.py`.

It resolves the GNN checkout from `--gnn-root`, `$GNN_ROOT`, or the sibling
`../Generalized_Notation_Notation`. Without a checkout it prints `SKIPPED`
unless `--require-gnn` is passed.

Run: `python scripts/validate_gnn_spec.py --require-gnn`.
Parent guide: [`../AGENTS.md`](../AGENTS.md).
