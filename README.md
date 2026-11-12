# GEN24

GEN24 is an exploratory Active Inference Institute repository for generating and transforming synthetic domain material, introduction letters, translations, and audience-specific calls to action. It contains Python scripts, source/context material, generated Markdown artifacts, an ARCANGEL Colab notebook, and the NHIM research module.

This repository is a working/exploring project rather than a packaged library. The generated artifacts currently checked in are part of the project record.

## Start here

- [Documentation index](docs/README.md)
- [Project overview](docs/overview.md)
- [Pipeline guide](docs/pipeline.md)
- [Input/output layout](GEN-24/Inputs_and_Outputs/README.md)
- [Notebook notes](docs/notebook.md)
- [NHIM notes](docs/nhim.md)
- [GNN integration](docs/gnn.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Citation](CITATION.cff)

## Repository layout

```text
GEN-24/
├── 1_generate_pro-intro.py       Build a pro-introduction prompt from source files
├── 2_write_intro_letter.py       Send the generated prompt to an OpenAI-compatible model
├── 3_translate_intro_letter.py   Translate the written letter into configured languages
├── 4_translate_call_to_action.py Generate audience-specific CTA translations
├── config.py                     Local model/output configuration
├── utils.py                      File, API-client, and LLM helpers
├── Inputs_and_Outputs/            Source material and generated Markdown artifacts
└── Archive/                      Historical scripts, not the active pipeline
NHIM/
├── nhim_inquiry_agent.gnn.md     GNN specification of the inquiry model
├── nhim_model.py                 NumPy implementation of that model
├── NHIM_API.py                   Experimental cognitive/ML module
└── gnn-pin.json                  Pinned upstream GNN parser revision
scripts/validate_gnn_spec.py      Validate the spec against the pinned parser
scripts/check_docs_links.py       Check relative Markdown links and the docs index
scripts/gate_metrics.py           Generate docs/_generated/GATE_METRICS.md
tests/                            Test suite (no API key or network required)
run.sh                            Gate entry point: lint, test, docs, gnn arms
pytest.ini                        Pinned pytest rootdir and test paths
ARCANGEL1_0_2.ipynb               Google Colab notebook
```

## Requirements

The GEN-24 pipeline uses Python and the `openai` package. Install its minimal dependency manifest from the `GEN-24` directory:

```bash
python -m pip install -r GEN-24/requirements.txt
```

The NHIM module has a separate, substantially heavier dependency set for its language-model and attention paths; see [docs/nhim.md](docs/nhim.md) and [NHIM/requirements.txt](NHIM/requirements.txt). Its canonical model, `NHIM/nhim_model.py`, needs NumPy only.

For the test suite and the GNN validator, install the repository-level development manifest:

```bash
python -m pip install -r requirements-dev.txt
```

## Generative model and GNN

The repository's only generative model is the **NHIM inquiry agent**, specified in Generalized Notation Notation and implemented once in NumPy:

- Specification: `NHIM/nhim_inquiry_agent.gnn.md`
- Implementation: `NHIM/nhim_model.py`
- Validator: `python scripts/validate_gnn_spec.py` (add `--require-gnn` to fail instead of skipping when no GNN checkout is present)

The specification is validated against the pinned upstream parser revision recorded in `NHIM/gnn-pin.json` (GNN `v5.0.3`, commit `8f293d9b`). The `GEN-24/` pipeline defines no generative model, so it carries no specification. See [GNN integration](docs/gnn.md) for the model, the pin, and the re-pin procedure.

GNN maintains a [downstream adoption guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/downstream_adoption.md) for repositories that consume the notation.

## Tests and CI

`./run.sh` is the only gate entry point. It defines one command per arm, and CI runs those arms verbatim, so a local run and the workflow cannot drift apart.

```bash
./run.sh                  # all arms: lint, test, docs, gnn
./run.sh lint test docs   # the offline gate
./run.sh gnn              # GNN conformance; needs a GNN checkout
```

The suite covers the inquiry model, its GNN specification and pin record, the NHIM sketch's deterministic paths, and the pipeline modules with a stubbed model client. It needs no API key and no network. The parser-level GNN conformance check is the `gnn` arm, not a pytest test: it runs `scripts/validate_gnn_spec.py --require-gnn --require-tag` against the checkout pinned in `NHIM/gnn-pin.json`, exactly once per gate.

GitHub Actions is billing-locked for this organization, so a green local `./run.sh` is the gate that actually executes. See the [gate and CI run guide](docs/RUN_GUIDE.md) for the arms, the environment, the CI job mapping, and the measured cost of each arm.

## API key setup

The live LLM scripts read `OPENAI_API_KEY` from a local file named `GEN-24/LLM_keys.key`. The file is intentionally not tracked and is ignored by Git. Create it locally with one line:

```text
OPENAI_API_KEY=your-key-here
```

Do not commit this file or any real credential. The first generation script can be imported/run without an API key after the lazy configuration fix, but the LLM-backed stages require one.

## Running the pipeline

Run commands from the repository root, or use the script paths shown below:

```bash
python GEN-24/1_generate_pro-intro.py
python GEN-24/2_write_intro_letter.py
python GEN-24/3_translate_intro_letter.py
python GEN-24/4_translate_call_to_action.py
```

Stages write into `GEN-24/Inputs_and_Outputs/`. The translation and call-to-action stages can make many model requests and may take time and incur provider charges. Existing artifact directories are created automatically by the scripts.

The checked-in translation artifacts are larger than the current language lists in the scripts: `Translated_Letters/` contains 181 language directories, while the current introduction-letter script lists 19 target languages; `Call_to_Action/` contains 29 language directories with 11 audience files each. See the [pipeline guide](docs/pipeline.md) for this distinction.

## Notebook

`ARCANGEL1_0_2.ipynb` is designed for Google Colab and begins by mounting Google Drive. It imports a broad scientific/ML stack and expects ARC-style challenge data in the notebook environment. It is not the same pipeline as `GEN-24/`. See [docs/notebook.md](docs/notebook.md).

## License

The repository is distributed under the MIT License. See [LICENSE](LICENSE).

## InstituteOS sidecar

This repository carries a committed [`.aii/`](.aii/README.md) sidecar with curated
InstituteOS metadata (`aii-sidecar/v1`): category, maturity, ecosystem relations,
key artifacts, provenance, and portable tasks. It is read by
[InstituteOS](https://github.com/ActiveInferenceInstitute/InstituteOS) tooling and
does not affect the project's build or runtime.
