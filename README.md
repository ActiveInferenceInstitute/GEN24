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
NHIM/NHIM_API.py                  Experimental cognitive/ML module
ARCANGEL1_0_2.ipynb               Google Colab notebook
```

## Requirements

The GEN-24 pipeline uses Python and the `openai` package. Install its minimal dependency manifest from the `GEN-24` directory:

```bash
python -m pip install -r GEN-24/requirements.txt
```

The NHIM module has a separate, substantially heavier dependency set; see [docs/nhim.md](docs/nhim.md) and [NHIM/requirements.txt](NHIM/requirements.txt).

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
