#!/usr/bin/env python3
"""Validate the NHIM GNN specification against the pinned upstream parser.

Checks performed (all against the pinned commit, never a live working tree):

1. The pin record ``NHIM/gnn-pin.json`` is readable and complete.
2. The GNN checkout contains the pinned commit, and the recorded release tag
   (when present) points at it.
3. Every recorded parser file matches its SHA-256 from the pinned commit.
4. ``src/gnn`` is extracted from the pinned commit into a temporary directory
   and the parser is imported from there.
5. The specification parses without errors, declares the expected variables and
   dimensions, and its ``A``/``B``/``C``/``D`` matrices equal the reference
   implementation in ``NHIM/nhim_model.py``.

Usage:

    python scripts/validate_gnn_spec.py [--gnn-root PATH] [--require-gnn] [--require-tag]

The GNN checkout is resolved from ``--gnn-root``, then ``$GNN_ROOT``, then the
sibling directory ``../Generalized_Notation_Notation``. When it is absent the
script prints ``SKIPPED`` and exits 0 unless ``--require-gnn`` is passed
(exit 2). A failed check exits 1. The upstream parser needs ``defusedxml``
(see ``requirements-dev.txt``).
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = REPO_ROOT / "NHIM" / "gnn-pin.json"

EXPECTED_DIMENSIONS = {
    "A": [3, 3],
    "B": [3, 3, 3],
    "C": [3],
    "D": [3, 1],
    "s": [3, 1],
    "s_prime": [3, 1],
    "F": [1],
    "o": [3, 1],
    "π": [3],
    "G": [3],
    "u": [1],
    "distress": [1],
    "t": [1],
}
EXPECTED_PARAMETERS = ("A", "B", "C", "D")
TOLERANCE = 1e-5


def _git(gnn_root, *args):
    """Run a read-only git command in the GNN checkout."""
    result = subprocess.run(
        ["git", "-C", str(gnn_root), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    return result


def resolve_gnn_root(explicit):
    """Return the GNN checkout path, or None when it is not present."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(REPO_ROOT.parent / "Generalized_Notation_Notation")
    import os

    if os.environ.get("GNN_ROOT"):
        candidates.insert(0, Path(os.environ["GNN_ROOT"]))
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate.resolve()
    return None


def verify_pin(gnn_root, pin, fail, require_tag=False):
    """Verify commit identity, the release tag, and recorded parser-file hashes."""
    commit = pin["commit"]
    result = _git(gnn_root, "cat-file", "-t", commit)
    if result.returncode != 0 or result.stdout.decode().strip() != "commit":
        fail(f"pinned commit {commit} is not present in {gnn_root}")
    resolved = _git(gnn_root, "rev-parse", f"{commit}^{{commit}}")
    if resolved.returncode != 0 or resolved.stdout.decode().strip() != commit:
        fail(f"pinned commit {commit} does not resolve to itself")
    print(f"pin commit present: {commit}")

    release = pin.get("release")
    if release:
        tagged = _git(gnn_root, "rev-parse", f"{release}^{{commit}}")
        if tagged.returncode != 0:
            if require_tag:
                fail(f"release tag {release} is not present in {gnn_root}")
            else:
                print(f"release tag {release}: not present locally (not checked)")
        elif tagged.stdout.decode().strip() == commit:
            print(f"release tag {release}: points at the pinned commit")
        else:
            fail(f"release tag {release} does not point at {commit}")

    for path, expected in sorted(pin["parser_files"].items()):
        result = _git(gnn_root, "show", f"{commit}:{path}")
        if result.returncode != 0:
            fail(f"pinned file missing: {path}")
        actual = hashlib.sha256(result.stdout).hexdigest()
        if actual != expected:
            fail(f"hash mismatch for {path}: {actual} != {expected}")
        print(f"hash ok: {path}")


def import_pinned_parser(gnn_root, commit):
    """Extract src/gnn from the pinned commit and import the parser from it."""
    result = subprocess.run(
        ["git", "-C", str(gnn_root), "archive", commit, "src/gnn"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode().strip() or "git archive failed")

    tmp = tempfile.TemporaryDirectory(prefix="gnn-pin-")
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        archive.extractall(tmp.name, filter="data")
    tmp_path = Path(tmp.name).resolve()
    sys.path.insert(0, str(tmp_path / "src"))

    from gnn.parsers.markdown_parser import MarkdownGNNParser  # noqa: E402

    module_path = Path(sys.modules[MarkdownGNNParser.__module__].__file__).resolve()
    # Resolve both sides: TMPDIR can sit behind a symlink (macOS /var ->
    # /private/var), so a textual prefix comparison is not reliable.
    if not module_path.is_relative_to(tmp_path):
        raise RuntimeError(f"parser was not imported from the pinned tree: {module_path}")
    return tmp, MarkdownGNNParser, module_path


def check_spec(spec_path, parser, model_module, fail):
    """Parse the spec and compare it with the reference implementation."""
    result = parser().parse_file(str(spec_path))
    if not result.success or result.errors:
        fail(f"spec did not parse cleanly: success={result.success} errors={result.errors}")

    model = result.model
    issues = model.validate_structure()
    if issues:
        fail(f"spec structure issues: {issues}")

    for name, dimensions in EXPECTED_DIMENSIONS.items():
        variable = model.get_variable_by_name(name)
        if variable is None:
            fail(f"spec is missing variable {name}")
        if list(variable.dimensions) != dimensions:
            fail(f"{name} dimensions {variable.dimensions} != {dimensions}")
    print(f"variables ok: {len(EXPECTED_DIMENSIONS)} declared with expected dimensions")

    import numpy as np

    for name in EXPECTED_PARAMETERS:
        parameter = model.get_parameter_by_name(name)
        if parameter is None:
            fail(f"spec is missing parameter {name}")

    spec_A = np.array(model.get_parameter_by_name("A").value, dtype=float)
    if not np.allclose(spec_A, model_module.A, atol=TOLERANCE):
        fail(f"A mismatch: spec={spec_A.tolist()} model={model_module.A.tolist()}")

    # The spec stores B per action as [action][next_state][previous_state].
    spec_B = np.transpose(
        np.array(model.get_parameter_by_name("B").value, dtype=float), (1, 2, 0)
    )
    if not np.allclose(spec_B, model_module.B, atol=TOLERANCE):
        fail("B mismatch between spec and NHIM/nhim_model.py")

    spec_C = np.array(model.get_parameter_by_name("C").value, dtype=float).ravel()
    if not np.allclose(spec_C, model_module.C, atol=TOLERANCE):
        fail(f"C mismatch: spec={spec_C.tolist()} model={model_module.C.tolist()}")

    spec_D = np.array(model.get_parameter_by_name("D").value, dtype=float).ravel()
    if not np.allclose(spec_D, model_module.D, atol=TOLERANCE):
        fail(f"D mismatch: spec={spec_D.tolist()} model={model_module.D.tolist()}")
    print("matrices ok: A, B, C, D equal NHIM/nhim_model.py")

    if not np.allclose(spec_A.sum(axis=0), np.ones(3), atol=1e-6):
        fail("A columns are not stochastic")
    for action in range(spec_B.shape[2]):
        if not np.allclose(spec_B[:, :, action].sum(axis=0), np.ones(3), atol=1e-6):
            fail(f"B slice {action} is not column-stochastic")
    print("stochasticity ok: A columns and every B action slice sum to 1")

    for name in ("Equations", "Time", "ActInfOntologyAnnotation", "ModelParameters"):
        if name not in model.raw_sections:
            fail(f"spec is missing optional section {name}")
    print("sections ok: required and optional sections present")


def main(argv=None):
    parser_args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser_args.add_argument("--gnn-root", help="path to the GNN checkout")
    parser_args.add_argument(
        "--require-gnn",
        action="store_true",
        help="fail (exit 2) instead of skipping when the GNN checkout is absent",
    )
    parser_args.add_argument(
        "--require-tag",
        action="store_true",
        help="fail when the release tag is missing instead of reporting it unchecked",
    )
    args = parser_args.parse_args(argv)

    failures = []

    def fail(message):
        failures.append(message)
        print(f"FAIL: {message}")

    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    spec_path = REPO_ROOT / pin["spec"]
    print(f"spec: {spec_path.relative_to(REPO_ROOT)}")
    print(f"spec sha256: {hashlib.sha256(spec_path.read_bytes()).hexdigest()}")
    print(f"pin: {pin['repository']} {pin['release']} {pin['commit']}")

    gnn_root = resolve_gnn_root(args.gnn_root)
    if gnn_root is None:
        if args.require_gnn:
            print("SKIPPED with --require-gnn: GNN checkout not found", file=sys.stderr)
            return 2
        print(
            "SKIPPED: GNN checkout not found (set --gnn-root or $GNN_ROOT to validate)"
        )
        return 0
    print(f"GNN checkout: {gnn_root}")

    verify_pin(gnn_root, pin, fail, require_tag=args.require_tag)
    if failures:
        return 1

    sys.path.insert(0, str(REPO_ROOT / "NHIM"))
    import nhim_model  # noqa: E402

    try:
        tmp, parser_class, module_path = import_pinned_parser(gnn_root, pin["commit"])
    except Exception as exc:  # dependency or extraction problem
        print(f"FAIL: could not import the pinned parser: {exc}", file=sys.stderr)
        return 1

    with tmp:
        print(f"pinned parser module: {module_path}")
        check_spec(spec_path, parser_class, nhim_model, fail)

    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("VALID: spec parses against the pinned parser and matches NHIM/nhim_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
