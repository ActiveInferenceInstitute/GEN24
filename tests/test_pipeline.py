"""Import-safety and contract tests for the GEN-24 letter pipeline.

These cover the modules that call the model API without performing a live call:
the shared client in ``utils.py`` is replaced with a stub.
"""

import importlib.util
import pathlib
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN24 = ROOT / "GEN-24"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, GEN24 / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config():
    return _load("gen24_config", "config.py")


@pytest.fixture(scope="module")
def utils():
    return _load("gen24_utils", "utils.py")


def test_config_targets_a_current_model_and_known_directories(config):
    assert config.USER_CONFIG["llm_params"]["model"] == "gpt-4o-mini"
    assert config.DEFAULT_CONFIG["llm_params"]["model"] == "gpt-4o-mini"
    for directory in ("Pro-Introduction_Letter", "Written_Introduction_Letter"):
        assert directory in config.USER_CONFIG["output_dirs"]


def test_utils_generates_a_response_through_the_current_sdk(utils, monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = types.SimpleNamespace(content="  hello  ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(utils, "client", fake_client)

    text, elapsed = utils.generate_llm_response("prompt")
    assert text == "hello"
    assert elapsed >= 0
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]


def test_api_key_lookup_reports_the_missing_file(utils):
    if (GEN24 / "LLM_keys.key").exists():
        pytest.skip("a local LLM_keys.key exists; the missing-file path cannot be tested")
    with pytest.raises(ValueError):
        utils.get_api_key()


def test_pipeline_scripts_expose_their_entry_points():
    expected = {
        "1_generate_pro-intro.py": ("generate_introduction_letter", "main"),
        "2_write_intro_letter.py": ("process_introduction_letter", "process_introduction_letters"),
        "3_translate_intro_letter.py": ("process_introduction_letter", "process_introduction_letters"),
        "4_translate_call_to_action.py": ("process_call_to_action", "process_call_to_actions"),
    }
    for filename, functions in expected.items():
        module = _load(f"gen24_{filename.replace('-', '_').replace('.py', '')}", filename)
        for function in functions:
            assert callable(getattr(module, function)), f"{filename}:{function}"
