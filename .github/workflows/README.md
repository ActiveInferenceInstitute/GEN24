# `.github/workflows/` — CI jobs

`ci.yml` defines two jobs, and each gate step is an arm of
[`../../run.sh`](../../run.sh):

- `tests` — installs `requirements-dev.txt` on Python 3.12, then runs
  `./run.sh lint`, `./run.sh test` and `./run.sh docs`. Offline.
- `gnn conformance (pinned)` (job key `gnn-conformance`) — fetches the commit in
  [`../../NHIM/gnn-pin.json`](../../NHIM/gnn-pin.json) without a worktree
  checkout, exports `GNN_ROOT`, and runs `./run.sh gnn`.

Neither job makes a live LLM request. The test suite stubs the model client.

Only one install runs per job, and both jobs share the uv dependency cache
keyed on `requirements-dev.txt`. The workflow does not inline a command that an
arm already defines; add a new step to `run.sh` first.
