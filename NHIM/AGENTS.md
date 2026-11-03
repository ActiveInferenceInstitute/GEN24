# AGENTS.md — `NHIM/`

- `nhim_model.py` and `nhim_inquiry_agent.gnn.md` are one contract. Change them
  together and rerun `scripts/validate_gnn_spec.py --require-gnn`.
- Read the parser pin from `gnn-pin.json`; never validate against a live GNN
  working tree. A re-pin updates the commit and every recorded parser-file hash.
- Keep `NHIM_API.py` imports lazy and its interactive loop behind
  `if __name__ == "__main__":`. The deterministic path must import with NumPy
  alone.
- Do not claim a live model result or a completed inquiry run from the tests.
- Parent rules: [../AGENTS.md](../AGENTS.md).
