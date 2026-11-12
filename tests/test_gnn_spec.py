"""Checks for the GNN specification and its pin record.

These are the offline checks: they run without a GNN checkout. The parser-level
conformance run is the `gnn` arm of `./run.sh`, which invokes
``scripts/validate_gnn_spec.py --require-gnn`` once; it is deliberately not
wrapped in a test, so the validator never runs twice in one gate.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIN = json.loads((ROOT / "NHIM" / "gnn-pin.json").read_text(encoding="utf-8"))
SPEC = ROOT / PIN["spec"]

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
    assert PIN["tag"] == "v5.0.3"
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["revision"])
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
