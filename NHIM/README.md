# `NHIM/` — inquiry-model specification and implementation

- `nhim_inquiry_agent.gnn.md` — the GNN specification.
- `nhim_model.py` — the single NumPy implementation; must stay numerically
  identical to the specification.
- `NHIM_API.py` — experimental sketch with deliberately lazy spaCy,
  Transformers and Matplotlib imports; the deterministic path runs with NumPy
  alone.
- `gnn-pin.json` — the pinned parser revision (`v5.0.3`, commit
  `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) and the parser-file hashes.
- `requirements.txt` — the module's own dependency list.

Validate: `./run.sh gnn` from the repository root, which runs
`python scripts/validate_gnn_spec.py --require-gnn --require-tag` and needs a
GNN checkout ([`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md)).
Parent guide: [`../AGENTS.md`](../AGENTS.md). Notes:
[`../docs/nhim.md`](../docs/nhim.md).
