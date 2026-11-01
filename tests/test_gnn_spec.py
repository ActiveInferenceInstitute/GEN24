"""Checks for the GNN specification, its pin record, and the validator.

The parser-level validation runs only when the GNN checkout is present; the
pin-record and specification-structure checks always run.
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIN = json.loads((ROOT / "NHIM" / "gnn-pin.json").read_text(encoding="utf-8"))
SPEC = ROOT / PIN["spec"]

sys.path.insert(0, str(ROOT / "scripts"))
import validate_gnn_spec  # noqa: E402

GNN_ROOT = validate_gnn_spec.resolve_gnn_root(None)

REQUIRED_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)

DECLARED_VARIABLES = (
    "A[3,3,type=float]",
    "B[3,3,3,type=float]",
    "C[3,type=float]",
    "D[3,1,type=float]",
    "s[3,1,type=float]",
    "s_prime[3,1,type=float]",
    "F[1,type=float]",
    "o[3,1,type=int]",
    "π[3,type=float]",
    "G[3,type=float]",
    "u[1,type=int]",
    "distress[1,type=float]",
    "t[1,type=int]",
)


def test_pin_record_is_complete():
    assert PIN["repository"] == "ActiveInferenceInstitute/Generalized_Notation_Notation"
    assert PIN["release"] == "v5.0.3"
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["commit"])
    assert PIN["parser_entrypoint"] == "gnn.parsers.markdown_parser.MarkdownGNNParser"
    assert PIN["parser_files"], "the pin must record the parser files it depends on"
    for path, digest in PIN["parser_files"].items():
        assert path.startswith("src/gnn/") or path.startswith("docs/"), path
        assert re.fullmatch(r"[0-9a-f]{64}", digest), path


def test_spec_file_exists_and_declares_its_sections():
    text = SPEC.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert f"## {section}" in text, section


def test_spec_declares_the_model_variables():
    text = SPEC.read_text(encoding="utf-8")
    for declaration in DECLARED_VARIABLES:
        assert declaration in text, declaration


def test_spec_points_at_its_implementation():
    text = SPEC.read_text(encoding="utf-8")
    assert PIN["implementation"] in text
    assert (ROOT / PIN["implementation"]).exists()


@pytest.mark.skipif(GNN_ROOT is None, reason="GNN checkout not present")
def test_validator_passes_against_the_pinned_parser():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_gnn_spec.py"), "--require-gnn"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALID" in result.stdout
    assert "hash ok" in result.stdout
