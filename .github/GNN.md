# GNN conformance

`gnn conformance (pinned)` is defined in `.github/workflows/ci.yml` and reads
`NHIM/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`).

The job checks out that commit, requires the `v5.0.3` tag to peel to it, and
runs `python scripts/validate_gnn_spec.py --require-gnn`. The script verifies
the commit and every recorded `parser_files` hash, extracts `src/gnn` from the
pin, and parses `NHIM/nhim_inquiry_agent.gnn.md` with that parser. It then
checks the declared variables and their dimensions, that matrices `A`, `B`, `C`
and `D` equal `NHIM/nhim_model.py`, and that the `A` columns and each `B`
action slice are stochastic. `tests/test_gnn_spec.py` re-checks the pin record,
the spec's sections and variables, and runs the validator. Neither step scans
the rest of the repository for `*.gnn.md`.
