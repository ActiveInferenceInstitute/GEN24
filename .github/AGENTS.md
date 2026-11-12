# AGENTS.md — `.github/`

- Every gate step in `workflows/ci.yml` is an arm of `../run.sh`. Add or change
  the arm first; never inline a second copy of a command in the workflow.
- Keep action references pinned to full commit SHAs with the release in a
  trailing comment. The organization composite action
  `ActiveInferenceInstitute/.github/.github/actions/setup-python-uv` is pinned
  at `7159e67315330aac8e414aebae0ad1e999a41448`; do not float it.
- Both jobs stay offline and deterministic: no provider key, no live model call,
  no network beyond the pinned GNN clone.
- The `gnn-conformance` job key and its `gnn conformance (pinned)` display name
  are the required-check identity. The commit and release come from
  `NHIM/gnn-pin.json`; do not duplicate the hash here.
- One install per job, and the uv cache is keyed on `requirements-dev.txt` so
  the two jobs do not resolve dependencies twice.
- Parent rules: [../AGENTS.md](../AGENTS.md).
