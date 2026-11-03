# AGENTS.md — `scripts/`

- Keep the validator's resolution order: explicit `--gnn-root`, then
  `$GNN_ROOT`, then the sibling checkout. A missing checkout is `SKIPPED` unless
  `--require-gnn` is set, and `--require-gnn` must fail rather than skip.
- Read the pin and parser hashes from `NHIM/gnn-pin.json`; do not hardcode a
  second copy.
- Keep the comparison against `NHIM/nhim_model.py` real; a smoke import is not
  a conformance check.
- Parent rules: [../AGENTS.md](../AGENTS.md).
