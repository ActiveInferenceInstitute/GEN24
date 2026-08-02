# Data and output layout

All active project material is under `GEN-24/Inputs_and_Outputs/`.

| Directory | Contents |
|---|---|
| `Entity/` | Entity worldview Python modules plus Markdown context/source documents. |
| `Organization/` | Organization worldview Python modules plus Markdown context/source documents. |
| `Domain/` | `Synthetic_*.md` synthetic domain descriptions. |
| `Prompts/` | Prompt templates, including `Introduction_Letter_Prompt.md`. |
| `Pro-Introduction_Letter/` | Prompt assembled from selected entity and organization files by stage 1. |
| `Written_Introduction_Letter/` | LLM-produced introduction letter from stage 2. |
| `Translated_Letters/` | Language-specific versions of the written introduction letter. |
| `Call_to_Action/` | Language- and audience-specific ReclAIMEd call-to-action artifacts from stage 4. |

## Naming patterns

The active scripts use these patterns:

- `introduction_letter_<Entity1>_and_<Entity2>.md` for the stage-1 assembled prompt.
- `written_introduction_letter_<Entity1>_and_<Entity2>.md` for the stage-2 letter.
- `translated_<Language>_written_introduction_letter_<Entity1>_and_<Entity2>.md` for translated letters.
- `translated_<Language>_<Audience>_ReclAIMEd_Call_To_Action.md` for call-to-action outputs.

Directory and filename inventories are generated artifacts, not a normative promise that every language or audience will always be present. Check the script constants when planning a new run.

## Source/code relationship

The entity and organization `.py` files are read as text by stage 1; they are source material, not imported as worldview objects by that stage. The Markdown context documents provide additional project material but are not automatically consumed by every active script. `utils.py` provides file I/O and OpenAI client helpers.

The archive has its own explanation in [GEN-24/Archive/README.md](../GEN-24/Archive/README.md).
