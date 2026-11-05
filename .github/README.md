# `.github/` — repository automation

`workflows/ci.yml` is the only workflow. It runs the compile-all and pytest
gate on Python 3.12, then the `gnn conformance (pinned)` job that
provisions the parser commit pinned in
[`../NHIM/gnn-pin.json`](../NHIM/gnn-pin.json) and runs
[`../scripts/validate_gnn_spec.py`](../scripts/validate_gnn_spec.py).

The test job needs no API key and no network. See
[`workflows/README.md`](workflows/README.md) and the root
[`../AGENTS.md`](../AGENTS.md) gate list.
