# Gate and CI run guide

`./run.sh` is the repository's only gate entry point. It defines one command
per **primitive** arm, and each **composition** runs those primitives once
each — never twice. [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
calls the arms verbatim, so a local run and CI cannot drift apart. GitHub
Actions is billing-locked for this organization; a local run is the gate that
actually executes.

## Arms

Primitive arms:

| Arm | Command | Needs | Proves |
|---|---|---|---|
| `lint` | `python -m compileall -q -x '(^\|/)(\.venv\|\.git)/' .` | nothing | every tracked Python source parses |
| `test-fast` | `python -m pytest tests/ -q` | nothing | the offline suite: model, spec and pin record, sketch, pipeline |
| `docs` | `python scripts/check_docs_links.py` and `python scripts/gate_metrics.py --check` | nothing | relative Markdown links resolve, every `docs/` page is indexed, the generated facts are current |
| `gnn` | `python scripts/validate_gnn_spec.py --require-gnn --require-tag` | a GNN checkout | the pin's commit and release tag are present, every recorded parser hash matches, and the specification parses with the pinned parser and matches `NHIM/nhim_model.py` |

Compositions:

| Arm | Parts | Notes |
|---|---|---|
| `fast` | `lint`, `test-fast`, `docs` | the default; offline, no checkout needed |
| `test-full` | `test-fast`, `gnn` | everything that tests the code, including the pinned-parser conformance run |
| `all` | `fast`, `gnn` | every primitive arm once |

`./run.sh` with no arguments runs `fast`. Any subset works, in the order given:
`./run.sh lint test-fast` or `./run.sh test-full`. `./run.sh --list` prints
every arm with its kind (`primitive` or `composition`); that output is also what
[`scripts/gate_metrics.py`](../scripts/gate_metrics.py) reads, so the arm list
has one source.

`test-full` is a composition rather than a second pytest selection because the
repository's only slow or conditional test coverage *is* the pinned-parser
conformance run. There is no marker or `-k` expression that would add tests to
the offline suite; the conformance check is a separate command, so the full arm
is the fast arm plus that command.

There is no `types` arm. The repository has no type checker, no annotations
contract, and no `pyproject.toml`; adding an arm without a checker behind it
would be a claim, not a gate.

The `lint` arm excludes `.venv` and `.git`. A local virtual environment holds
site-packages, not repository sources, and compiling it dominated the arm's
cost without checking anything this repository owns.

## Environment

One environment definition: [`requirements-dev.txt`](../requirements-dev.txt).
It is the only place the suite's dependencies are declared, and both CI jobs
install from it.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

`run.sh` resolves the interpreter with one rule: `$PYTHON` if set, then
`.venv/bin/python`, then `python3`. CI satisfies the same rule because the
setup step activates `.venv` at the workspace root.

There is no lockfile. `requirements-dev.txt` is unpinned on purpose — it holds
four packages, and pinning them would need a `pyproject.toml` and a resolver
lock this repository does not have. Do not describe it as a lock.

CI installs with the organization's composite action,
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`,
which installs a pinned `uv` and CPython. Its cache is keyed on
`requirements-dev.txt`, so the two jobs share one dependency cache instead of
resolving twice.

## CI mapping

| Job key | Display name | Arm |
|---|---|---|
| `tests` | `tests` | `./run.sh fast` |
| `gnn-conformance` | `gnn conformance (pinned)` | `./run.sh gnn` |

The two jobs together run every primitive arm exactly once, which is what
`./run.sh all` does locally. The `tests` job is offline. The
`gnn-conformance` job fetches the commit recorded in
[`../NHIM/gnn-pin.json`](../NHIM/gnn-pin.json), requires the release tag to
peel to it, and exports `GNN_ROOT` before running the arm.

`gnn-conformance` is the required-check identity; the job key and its display
name are both part of the contract described in
[`.github/GNN.md`](../.github/GNN.md).

## Measured cost

Wall times on this host (Darwin arm64), one run each, CPython 3.12.16,
2026-11-12. The `before` column is the command the repository actually ran
before this gate existed; `after` is the arm.

| Arm | Before | After | Change |
|---|---|---|---|
| `lint` | `python -m compileall -q .` — 0.60s | `./run.sh lint` — 0.15s | `.venv` is no longer compiled |
| `test-fast` | `python -m pytest tests/ -q` — 2.43s | `./run.sh test-fast` — 4.45s | same command; **no speedup claimed** (see below) |
| `docs` | no check existed | `./run.sh docs` — 2.57s | new arm |
| `gnn` | validator, then `pytest tests/test_gnn_spec.py` — 5.27s | `./run.sh gnn` — 3.85s | the validator runs once, not twice |

Compositions, same pass:

| Arm | Parts | Wall time |
|---|---|---|
| `fast` | `lint`, `test-fast`, `docs` | 6.41s |
| `test-full` | `test-fast`, `gnn` | 7.38s |
| `all` | `fast`, `gnn` | 8.81s |

Two caveats, because the honest reading matters more than the ratio:

- The `test-fast` arm runs the same command over the same suite minus the
  removed wrapper test. Repeated runs of that command on this host measured
  6.70s, 4.42s, 2.43s, 1.94s and 4.45s as caches warmed and other work ran
  alongside. The spread is far larger than the effect of deleting one test, so
  **no speedup is claimed for `test-fast`**, and the `after` value above is
  simply the last single pass, not a best case.
- The `gnn` saving is structural, not statistical: the old sequence ran
  `scripts/validate_gnn_spec.py --require-gnn` and then a pytest test that
  shelled out to the same script with the same flag. The arm now invokes it
  once.

These numbers are a dated observation, not a contract. The current arm list and
the latest measurement live in
[`_generated/GATE_METRICS.md`](_generated/GATE_METRICS.md), which
[`../scripts/gate_metrics.py`](../scripts/gate_metrics.py) produces; that file
measures the primitive arms only, because a composition is the sum of its
parts.

## What the gate does not cover

- The live LLM stages in `GEN-24/` need `GEN-24/LLM_keys.key` and provider
  budget. The suite stubs the client, so a green gate is not evidence that a
  live request works.
- `ARCANGEL1_0_2.ipynb` needs Colab and Google Drive data.
- The NHIM sketch's language-model and attention paths need model downloads
  that are not vendored.
- The GNN checkout is not fetched by `run.sh`. The `gnn` arm resolves it from
  `--gnn-root`, `$GNN_ROOT`, or the sibling directory
  `../Generalized_Notation_Notation`, and fails when none is present, because
  `--require-gnn` turns an absent checkout into a failure rather than a skip.
