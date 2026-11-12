# GEN-24 pipeline

The active pipeline is a sequence of four scripts under `GEN-24/`. Each script uses paths derived from its own file location after the portability fixes in this pass.

## Stages

1. `1_generate_pro-intro.py` loads five entity/organization Python source files and the introduction-letter prompt, then writes `Inputs_and_Outputs/Pro-Introduction_Letter/introduction_letter_Hipolito_and_Friston.md`.
2. `2_write_intro_letter.py` reads that generated prompt, sends it through `utils.generate_llm_response()`, and writes `Inputs_and_Outputs/Written_Introduction_Letter/written_introduction_letter_Hipolito_and_Friston.md`.
3. `3_translate_intro_letter.py` reads Markdown files from `Written_Introduction_Letter/` and writes one language directory/file per configured target language under `Translated_Letters/`. Its current `TARGET_LANGUAGES` list contains 19 languages.
4. `4_translate_call_to_action.py` embeds the call-to-action source text in its prompt, adapts it for each configured audience and language, and writes 11 audience files under each language directory in `Call_to_Action/`. Its current lists contain 29 languages and 11 audiences.

Run from the repository root:

```bash
python GEN-24/1_generate_pro-intro.py
python GEN-24/2_write_intro_letter.py
python GEN-24/3_translate_intro_letter.py
python GEN-24/4_translate_call_to_action.py
```

## Configuration and credentials

`GEN-24/config.py` contains output directories and model parameters, including the model name, token limit, sampling settings, and API-key file location. API-key loading is intentionally deferred until an LLM request is made, so importing the configuration does not require a local secret file.

Create `GEN-24/LLM_keys.key` locally:

```text
OPENAI_API_KEY=your-key-here
```

The file is ignored by Git. Never commit a real key. The pipeline uses the `openai` client package; install [GEN-24/requirements.txt](../GEN-24/requirements.txt).

## Outputs and scale

The scripts create output directories as needed. Translation stages can issue many external model requests: rerunning them may overwrite existing generated files and may incur provider charges. The current checked-in artifacts are not a direct reproduction of the current script lists: `Translated_Letters/` has 181 language directories/files, while the script currently lists 19 languages. `Call_to_Action/` has 29 language directories and 319 files (11 audiences per language).

## Known limitations

- The call-to-action text is embedded in `4_translate_call_to_action.py`; a related source document is also present at `Inputs_and_Outputs/Entity/Shyaka/ReclAIMEd_Call_To_Action.md`. They are not automatically synchronized.
- `Archive/` contains historical scripts with old absolute paths and should not be treated as runnable pipeline stages.
- The repository has a test suite (`./run.sh test`) and a gate with four arms; see the [gate and CI run guide](RUN_GUIDE.md). The suite stubs the model client, so it does not prove that a live model request works.
