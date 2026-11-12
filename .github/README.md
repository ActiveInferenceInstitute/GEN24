# `.github/` — repository automation

[`workflows/ci.yml`](workflows/ci.yml) is the only workflow. Every gate step in
it is an arm of [`../run.sh`](../run.sh), so a local run and CI execute the same
commands:

| Job key | Display name | Arms |
|---|---|---|
| `tests` | `tests` | `./run.sh lint`, `./run.sh test`, `./run.sh docs` |
| `gnn-conformance` | `gnn conformance (pinned)` | `./run.sh gnn` |

The `tests` job needs no API key and no network. The `gnn-conformance` job
clones the parser commit pinned in [`../NHIM/gnn-pin.json`](../NHIM/gnn-pin.json)
and runs the validator once; see [`GNN.md`](GNN.md).

Both jobs set up Python through the organization composite action
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`,
cached on [`../requirements-dev.txt`](../requirements-dev.txt).

GitHub Actions is billing-locked for this organization, so a green local
`./run.sh` is the gate that actually runs. See
[`workflows/README.md`](workflows/README.md) and
[`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md).
