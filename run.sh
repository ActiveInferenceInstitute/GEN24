#!/usr/bin/env bash
#
# GEN24 gate. One entry point, one command per arm, and CI runs these arms
# verbatim (see .github/workflows/ci.yml) so there is no second copy of a
# command. Reference: docs/RUN_GUIDE.md.
#
#   ./run.sh                 # all arms, in order
#   ./run.sh lint            # compile every Python source
#   ./run.sh test            # offline test suite (no key, no network)
#   ./run.sh docs            # relative links + documentation index coverage
#   ./run.sh gnn             # GNN conformance against the pinned parser
#   ./run.sh lint test docs  # any subset, in the order given
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

ARMS="lint test docs gnn"

usage() {
  cat <<'EOF'
GEN24 gate. One entry point, one command per arm; CI runs these arms verbatim.

  ./run.sh                 all arms, in order
  ./run.sh lint            compile every Python source
  ./run.sh test            offline test suite (no key, no network)
  ./run.sh docs            relative links + documentation index coverage
  ./run.sh gnn             GNN conformance against the pinned parser
  ./run.sh lint test docs  any subset, in the order given
  ./run.sh --list          arm names, one per line

There is no `types` arm: the repository has no type checker.
EOF
}

arm_lint() {
  # .venv holds site-packages, not repository sources; compiling it is noise.
  "$PYTHON" -m compileall -q -x '(^|/)(\.venv|\.git)/' .
}

arm_test() {
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
  case "$1" in
    lint) arm_lint ;;
    test) arm_test ;;
    docs) arm_docs ;;
    gnn) arm_gnn ;;
    *)
      echo "run.sh: unknown arm '$1' (known: $ARMS)" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- $ARMS
fi

case "$1" in
  -h | --help | help)
    usage
    exit 0
    ;;
  --list)
    for arm in $ARMS; do echo "$arm"; done
    exit 0
    ;;
esac

for arm in "$@"; do
  echo "==> $arm"
  run_arm "$arm"
done
