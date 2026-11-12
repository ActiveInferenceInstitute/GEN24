# GNN conformance

`gnn conformance (pinned)` is the display name of the `gnn-conformance` job in
[`workflows/ci.yml`](workflows/ci.yml). It reads
[`../NHIM/gnn-pin.json`](../NHIM/gnn-pin.json) — schema `gnn-importer-pin/v1` —
which pins `ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`).

The job clones the pinned repository without a worktree checkout, exports
`GNN_ROOT`, and runs the `gnn` arm of [`../run.sh`](../run.sh), which is
`python scripts/validate_gnn_spec.py --require-gnn --require-tag`. The arm
requires the pinned commit to be present, the `v5.0.3` tag to peel to it, and
every recorded `parser_files` hash to match. The script then extracts `src/gnn`
from the pinned commit, parses `NHIM/nhim_inquiry_agent.gnn.md` with that
parser, checks the declared variables and their dimensions, checks that
matrices `A`, `B`, `C` and `D` equal `NHIM/nhim_model.py`, and checks that the
`A` columns and each `B` action slice are stochastic.

[`../tests/test_gnn_spec.py`](../tests/test_gnn_spec.py) re-checks the pin
record, the spec's sections and the declared variables offline. It does not run
the validator, so the parser-level check executes exactly once per gate
instead of twice. Neither the arm nor the tests scan the rest of the
repository for `*.gnn.md`.
