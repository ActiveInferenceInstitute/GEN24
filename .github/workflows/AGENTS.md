# AGENTS.md — `.github/workflows/`

- Keep the `tests` and `gnn conformance (pinned)` names stable; they are
  the required checks.
- Do not add a job that needs `GEN-24/LLM_keys.key`, a provider budget or a live
  network model call.
- Read the GNN commit from `NHIM/gnn-pin.json`; a pin change must be accompanied
  by refreshed parser-file hashes and a local `--require-gnn` run.
- Parent rules: [../AGENTS.md](../AGENTS.md).
