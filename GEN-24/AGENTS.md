# AGENTS.md — `GEN-24/`

- Construct model clients only through `utils.generate_llm_response` and
  `utils.get_api_key`. Do not add a second client or read a key elsewhere.
- `LLM_keys.key` is untracked and must stay untracked. Never print, commit or
  copy a key into an artifact.
- Do not describe this directory as a generative model. It is file-and-prompt
  automation; the repository's model is the NHIM inquiry agent.
- [`Inputs_and_Outputs/`](Inputs_and_Outputs/) is a project record. Do not
  delete, move or bulk-regenerate artifacts as cleanup.
- [`Archive/`](Archive/) is history, not setup. Do not wire it back into the
  active pipeline.
- Parent rules: [../AGENTS.md](../AGENTS.md).
