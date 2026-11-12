# GEN24 — agent notes

Exploratory Active Inference Institute repository: a four-stage letter
generation/translation pipeline, a Colab notebook, and the NHIM research
module with its GNN-specified inquiry model. Stewarded by the Active Inference
Institute; MIT licensed. See [README.md](README.md), [docs/README.md](docs/README.md),
and [docs/gnn.md](docs/gnn.md).

## Layout and contracts

- `GEN-24/` is the pipeline: `1_generate_pro-intro.py`,
  `2_write_intro_letter.py`, `3_translate_intro_letter.py`,
  `4_translate_call_to_action.py`, plus `config.py` and `utils.py`.
  `GEN-24/Archive/` is historical and not runnable.
- `GEN-24/utils.py` is the single home for model calls
  (`generate_llm_response`, `get_api_key`); scripts must not construct their
  own clients. Credentials come only from the untracked file
  `GEN-24/LLM_keys.key` — never commit a key.
- `GEN-24/Inputs_and_Outputs/` contains both source material and generated
  artifacts, and the generated ones are part of the project record: do not
  delete, move, or regenerate them as cleanup. `Translated_Letters/` holds 181
  language directories while the script lists 19 languages; that gap is
  documented, not a bug to "fix" by deleting artifacts.
- `NHIM/nhim_model.py` is the single implementation of the NHIM inquiry model.
  `NHIM/nhim_inquiry_agent.gnn.md` is its specification, and the two must stay
  numerically identical — `scripts/validate_gnn_spec.py` enforces that.
- `NHIM/NHIM_API.py` is the sketch. Its spaCy/Transformers/Matplotlib imports
  are deliberately lazy; keep the deterministic paths runnable with NumPy
  alone and keep the interactive loop behind `if __name__ == "__main__":`.

## Gates (run before publishing)

```bash
pip install -r requirements-dev.txt
./run.sh lint test docs   # offline: compile, suite, links + generated facts
./run.sh gnn              # needs a GNN checkout
```

`./run.sh` is the single gate entry point and CI runs the same arms; see
[docs/RUN_GUIDE.md](docs/RUN_GUIDE.md). There is no `types` arm because the
repository has no type checker. Do not add a command to the workflow that is
not an arm.

The test suite stubs the model client: it needs no API key and no network, and
it does not prove that a live model request works. The GNN validator resolves
the checkout from `--gnn-root`, `$GNN_ROOT`, or the sibling directory
`../Generalized_Notation_Notation`; the `gnn` arm passes `--require-gnn` and
`--require-tag`, so an absent checkout or release tag fails instead of
skipping.

## Boundaries

- Do not claim live model results. The LLM-backed stages need
  `GEN-24/LLM_keys.key` and provider budget, neither of which is in the
  repository.
- The notebook needs Colab and Google Drive data; its dependencies are not
  installed by `requirements-dev.txt`.
- Never describe `GEN-24/` as a generative model. It is file-and-prompt
  automation; the repository's only model is the NHIM inquiry agent.

## Generalized Notation Notation (GNN)

The NHIM inquiry model is specified in GNN and implemented once in NumPy
(`NHIM/nhim_model.py`). Validate changes against the pinned upstream parser
revision recorded in [NHIM/gnn-pin.json](NHIM/gnn-pin.json) — never against a
live working tree — and re-pin by updating the commit plus every recorded
parser-file hash:

- GNN repository: https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation
- Integration guide: https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/framework_integration_guide.md
- Implementation guide: https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/gnn_implementation.md
- Related fleet repositories: [active_inference_language_model](https://github.com/ActiveInferenceInstitute/active_inference_language_model)
  (GNN-specified language stack), [Active_Inference_Ontology](https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology)
  (terms used in the spec's ontology annotation), [Garden-of-Iris](https://github.com/ActiveInferenceInstitute/Garden-of-Iris)
  (sibling exploratory repo; no model, so no specification).
