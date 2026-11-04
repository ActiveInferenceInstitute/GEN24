# `.github/workflows/` — CI jobs

`ci.yml` defines two jobs:

- `tests` — installs `requirements-dev.txt`, runs `python -m compileall -q .`
  and `python -m pytest tests/ -q` on Python 3.12.
- `gnn conformance (pinned)` — fetches the commit in
  [`../../NHIM/gnn-pin.json`](../../NHIM/gnn-pin.json), provisions a Python
  3.12 environment for the parser, and runs the spec validator.

Neither job makes a live LLM request. The test suite stubs the model client.
