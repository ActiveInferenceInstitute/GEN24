# `GEN-24/` — the four-stage letter pipeline

- `1_generate_pro-intro.py` assembles the stage-1 prompt from source material.
- `2_write_intro_letter.py` sends that prompt to an OpenAI-compatible model.
- `3_translate_intro_letter.py` translates the written letter into configured
  languages.
- `4_translate_call_to_action.py` generates audience-specific call-to-action
  translations.
- `config.py` holds local model/output configuration; `utils.py` is the single
  home for model calls and key lookup.

[`Archive/`](Archive/) is historical and not runnable.
[`Inputs_and_Outputs/`](Inputs_and_Outputs/) holds both source material and the
checked-in generated record. Stage details:
[`../docs/pipeline.md`](../docs/pipeline.md).

Stages 2-4 need an untracked `LLM_keys.key`; the tests do not.
