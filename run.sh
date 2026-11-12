#!/usr/bin/env bash
#
# GEN24 gate. One entry point, one command per primitive arm; compositions run
# those primitives once each and never repeat one. CI runs these arms verbatim
# (see .github/workflows/ci.yml), so there is no second copy of a command.
# Reference: docs/RUN_GUIDE.md.
#
#   ./run.sh                 # `fast`: lint, test-fast, docs (the default)
#   ./run.sh fast            # offline gate; no GNN checkout needed
#   ./run.sh all             # `fast` plus `gnn`; every command once
#
#   ./run.sh lint            # compile every Python source
#   ./run.sh test-fast       # offline test suite (no key, no network)
#   ./run.sh test-full       # test-fast plus gnn
#   ./run.sh docs            # relative links + documentation index coverage
#   ./run.sh gnn             # GNN conformance against the pinned parser
#
#   ./run.sh lint test-fast  # any subset, in the order given
#   ./run.sh --list          # arm name and kind (primitive or composition)
#
# There is no `types` arm: the repository has no type checker and no
# annotations contract to enforce. Do not add one without a checker.
set -euo pipefail

cd "$(dirname "$0")"

# One interpreter resolution rule for every arm and for CI: an activated
# environment wins, then the repository-local .venv, then python3.
if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

# name:kind. Compositions must not re-run a primitive that another composition
# in the same invocation already ran; each arm below lists its parts once.
ARMS="lint:primitive
test-fast:primitive
docs:primitive
gnn:primitive
test-full:composition
fast:composition
all:composition"

usage() {
  cat <<'EOF'
GEN24 gate. One entry point; CI runs these arms verbatim.

  ./run.sh                 `fast`: lint, test-fast, docs (the default)
  ./run.sh fast            offline gate; no GNN checkout needed
  ./run.sh all             `fast` plus `gnn`; every command once

  ./run.sh lint            compile every Python source
  ./run.sh test-fast       offline test suite (no key, no network)
  ./run.sh test-full       test-fast plus gnn
  ./run.sh docs            relative links + documentation index coverage
  ./run.sh gnn             GNN conformance against the pinned parser

  ./run.sh lint test-fast  any subset, in the order given
  ./run.sh --list          arm name and kind, one per line

There is no `types` arm: the repository has no type checker.
EOF
}

arm_lint() {
  # .venv holds site-packages, not repository sources; compiling it is noise.
  "$PYTHON" -m compileall -q -x '(^|/)(\.venv|\.git)/' .
}

arm_test_fast() {
  "$PYTHON" -m pytest tests/ -q
}

arm_docs() {
  "$PYTHON" scripts/check_docs_links.py
  "$PYTHON" scripts/gate_metrics.py --check
}

arm_gnn() {
  # --require-gnn makes an absent checkout a failure instead of a skip, and
  # --require-tag makes a missing release tag a failure instead of "not checked".
  "$PYTHON" scripts/validate_gnn_spec.py --require-gnn --require-tag
}

run_arm() {
  echo "==> $1"
  case "$1" in
    lint) arm_lint ;;
    test-fast) arm_test_fast ;;
    docs) arm_docs ;;
    gnn) arm_gnn ;;
    test-full)
      run_arm test-fast
      run_arm gnn
      ;;
    fast)
      run_arm lint
      run_arm test-fast
      run_arm docs
      ;;
    all)
      run_arm fast
      run_arm gnn
      ;;
    *)
      echo "run.sh: unknown arm '$1' (run ./run.sh --list)" >&2
      usage >&2
      exit 2
      ;;
  esac
}

case "${1:-}" in
  -h | --help | help)
    usage
    exit 0
    ;;
  --list)
    printf '%s\n' "$ARMS"
    exit 0
    ;;
esac

if [ "$#" -eq 0 ]; then
  set -- fast
fi

for arm in "$@"; do
  run_arm "$arm"
done
