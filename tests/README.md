# `tests/` — offline test suite

- `test_pipeline.py` — the four-stage file/prompt helpers.
- `test_nhim_model.py` — the NumPy inquiry model.
- `test_nhim_api.py` — deterministic paths and lazy optional imports.
- `test_gnn_spec.py` — the pin record, the specification's sections and its
  declared variables. Offline: it does not need a GNN checkout.
- `conftest.py` — shared paths and fixtures.

Run: `./run.sh test` (or `python -m pytest tests/ -q`). `pytest.ini` pins the
rootdir and `testpaths` so the collected node IDs do not depend on a parent
directory's configuration. The suite stubs the model client, so it needs no API
key and no network; it does not prove a live request works.

The parser-level conformance check is not a test. It is the `gnn` arm of
[`../run.sh`](../run.sh), which runs `scripts/validate_gnn_spec.py
--require-gnn --require-tag` once. Keeping it out of pytest means a gate runs
the validator once instead of twice.

Parent guide: [`../AGENTS.md`](../AGENTS.md). Gate reference:
[`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md).
