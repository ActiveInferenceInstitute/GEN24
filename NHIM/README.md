# `NHIM/` — inquiry-model specification and implementation

- `nhim_inquiry_agent.gnn.md` — the GNN specification.
- `nhim_model.py` — the single NumPy implementation; must stay numerically
  identical to the specification.
- `NHIM_API.py` — experimental sketch with deliberately lazy spaCy,
  Transformers and Matplotlib imports; the deterministic path runs with NumPy
  alone.
- `gnn-pin.json` — the pinned parser revision (`v5.1.0`, commit
  `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`) and the parser-file hashes.
- `requirements.txt` — the module's own dependency list.

Validate: `./run.sh gnn` from the repository root, which runs
`python scripts/validate_gnn_spec.py --require-gnn --require-tag` and needs a
GNN checkout ([`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md)).
Parent guide: [`../AGENTS.md`](../AGENTS.md). Notes:
[`../docs/nhim.md`](../docs/nhim.md).
