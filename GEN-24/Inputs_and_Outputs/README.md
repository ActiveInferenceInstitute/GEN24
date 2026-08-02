# Inputs and outputs

This directory contains the source material, prompts, intermediate files, and generated Markdown artifacts used by the GEN-24 scripts.

- `Entity/` — entity worldview Python files and Markdown context documents.
- `Organization/` — organization worldview Python files and Markdown context documents.
- `Domain/` — synthetic domain descriptions.
- `Prompts/` — prompt templates.
- `Pro-Introduction_Letter/` — stage-1 assembled prompt.
- `Written_Introduction_Letter/` — stage-2 LLM-written letter.
- `Translated_Letters/` — translated introduction-letter outputs, organized by language.
- `Call_to_Action/` — translated/adapted call-to-action outputs, organized by language and audience.

The four-stage workflow and its current language/audience lists are documented in [../../docs/pipeline.md](../../docs/pipeline.md). Generated artifact counts can differ from the current script lists because the checked-in tree includes earlier runs.
