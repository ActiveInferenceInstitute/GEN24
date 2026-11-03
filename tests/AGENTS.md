# AGENTS.md — `tests/`

- Keep the suite offline: no provider key, no live model call, no network.
  Stub the model client at its boundary.
- A skipped GNN test is not conformance evidence. The required check runs with
  `GNN_ROOT` set; locally use `scripts/validate_gnn_spec.py --require-gnn`.
- Add a test for each new deterministic path. Do not assert on generated
  artifact prose that a stage can rewrite.
- Parent rules: [../AGENTS.md](../AGENTS.md).
