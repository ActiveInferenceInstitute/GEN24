# `tests/` — offline test suite

- `test_pipeline.py` — the four-stage file/prompt helpers.
- `test_nhim_model.py` — the NumPy inquiry model.
- `test_nhim_api.py` — deterministic paths and lazy optional imports.
- `test_gnn_spec.py` — specification/implementation conformance; skips without
  a GNN checkout.
- `conftest.py` — shared paths and fixtures.

Run: `python -m pytest tests/ -q`. The suite stubs the model client, so it needs
no API key and no network; it does not prove a live request works.

Parent guide: [`../AGENTS.md`](../AGENTS.md).
