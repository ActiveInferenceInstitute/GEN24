"""Tests for the refactored NHIM sketch (NHIM/NHIM_API.py).

Only numpy is required: spaCy, Transformers, and Matplotlib are optional and
load lazily, so the deterministic paths are what these tests exercise.
"""

import ast
import pathlib

import numpy as np
import pytest

import NHIM_API as nhim

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "NHIM" / "NHIM_API.py").read_text(encoding="utf-8")


def test_no_duplicate_definitions():
    tree = ast.parse(SOURCE)
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(classes) == len(set(classes)), classes
    assert len(functions) == len(set(functions)), functions


def test_import_has_no_top_level_side_effects():
    # The old file loaded models and started the interactive loop at import
    # time. Logging configuration is the only call allowed at module level.
    tree = ast.parse(SOURCE)
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    for node in top_level_calls:
        assert ast.unparse(node).startswith("logging."), ast.unparse(node)


def test_inquiry_loop_updates_beliefs_and_distress():
    psi = nhim.ProblemSolvingIntelligence()
    for inquiry in ("What is active inference?", "Describe the garden.", "Is the model stable?"):
        response, beliefs, distress = psi.process_inquiry(inquiry)
        assert isinstance(response, str) and response
        assert beliefs.shape == (3,)
        assert beliefs.sum() == pytest.approx(1.0)
        assert 0.0 <= distress <= nhim.nhim_model.DISTRESS_MAX
    assert len(psi.memory_system.memory_storage) == 3


def test_observation_mapping_is_deterministic():
    entities = [("garden", "NOUN")]
    first = nhim.ProblemSolvingIntelligence.observe("text", entities)
    second = nhim.ProblemSolvingIntelligence.observe("text", entities)
    assert first == second
    assert 0 <= first < nhim.nhim_model.NUM_OBSERVATIONS
    assert 0 <= nhim.ProblemSolvingIntelligence.observe("text", []) < 3


def test_attention_gating_and_memory_system():
    saliency = nhim.AttentionalSelection.bottom_up_attention(
        np.array([[1.0], [2.0]]), np.ones(1)
    )
    assert saliency.shape == (2, 1)
    modulated = nhim.AttentionalSelection.top_down_attention(saliency, np.ones(2))
    assert np.allclose(modulated, saliency)

    memory = nhim.MemorySystem(capacity=2, hopfield_size=3)
    memory.memory_encoding([1.0, 0.0, 0.0])
    memory.memory_encoding([0.0, 1.0, 0.0])
    memory.memory_encoding([0.0, 0.0, 1.0])
    assert len(memory.memory_storage) == 2  # capacity is respected
    assert memory.memory_retrieval([0.0, 0.0, 1.0]) is not None


def test_validate_model_returns_results_without_matplotlib():
    results = nhim.validate_model(nhim.ProblemSolvingIntelligence(), ["first", "second"])
    assert len(results) == 2
    assert set(results[0]) == {
        "inquiry",
        "initial_beliefs",
        "final_beliefs",
        "initial_distress",
        "final_distress",
        "response",
        "elapsed_seconds",
    }
