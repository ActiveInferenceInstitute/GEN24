# TO-DO — GEN24 documentation pass

Last reviewed: 2026-11-11 (sidecar and test-count accuracy pass) after 2026-11-01

Severity definitions:
- **Minor** — typo, broken link, formatting, small doc/code drift fix.
- **Medium** — stale section rewrite, doc restructure, added missing guide, small hardening fix.
- **Major** — large doc system overhaul, cross-cutting refactors.

Legend: `[ ]` open · `[x]` completed (✓ with commit) · `[~]` deferred (reason in notes)

## Current-API and GNN-alignment pass (2026-11-01)

- [x] ✓ Added the canonical discrete model of the NHIM inquiry loop, `NHIM/nhim_model.py` (NumPy only).
- [x] ✓ Specified that model in Generalized Notation Notation: `NHIM/nhim_inquiry_agent.gnn.md`.
- [x] ✓ Pinned the upstream parser revision and its file hashes in `NHIM/gnn-pin.json` (GNN `v5.0.3`, commit `8f293d9b`).
- [x] ✓ Added `scripts/validate_gnn_spec.py`, which validates the spec against the pinned parser (never a live tree) and compares `A`/`B`/`C`/`D` with the implementation.
- [x] ✓ Deduplicated `NHIM/NHIM_API.py` (every class was defined twice), removed undefined-name bugs, made heavy imports lazy, and put the interactive loop behind `if __name__ == "__main__":`.
- [x] ✓ Added a test suite (`tests/`, 27 tests) covering the model, the spec and pin record, the NHIM sketch's deterministic paths, and the pipeline modules with a stubbed client.
- [x] ✓ Added `requirements-dev.txt` and a CI workflow (`.github/workflows/ci.yml`) that runs compileall, pytest, and the pinned-parser validation.
- [x] ✓ Made the model name current: `gpt-4o-mini` instead of the dated snapshot `gpt-4o-mini-2024-07-18`.
- [x] ✓ Removed unused helpers: `config.read_api_key()`, `utils.setup_logging()`, `utils.setup_llm()`, `utils.batch_process_llm()`, `utils.extract_entity_files()`, `utils.extract_organization_files()`, and the unused `openai` imports in stages 2-4.
- [x] ✓ Added `AGENTS.md` and `docs/gnn.md` with GNN signposting to the upstream integration and implementation guides and related fleet repositories.

Still not done in this pass: live model calls (no key, no personal archives) and any training or fitting of the sketch's neural components.

## Minor

- [x] ✓ Rewrite 2-line README stub into a real repository README (structure, setup, usage, links). — `README.md` (commit: 4a63fd8)
- [x] ✓ Add title/overview cell to `ARCANGEL1_0_2.ipynb` so the notebook is self-describing on GitHub. (commit: 3e628a5)
- [x] ✓ Fix error messages in `2_write_intro_letter.py`, `3_translate_intro_letter.py`, `4_translate_call_to_action.py` — they claim ".env or LLM_keys.key" but the code only reads `LLM_keys.key`. (commit: 00a1581)
- [x] ✓ Remove stale "space after GEN24 space" workaround comment in `4_translate_call_to_action.py` (root cause fixed). (commit: 00a1581)
- [x] ✓ Rename `GEN-24/Archive/2_write_intro_letters copy.py` -> `2_write_intro_letters_copy.py` (space in filename). (commit: 00a1581)
- [x] ✓ Add explicit `LLM_keys.key` to `.gitignore` (already covered by `*.key`; explicit entry documents intent). (commit: 00a1581)
- [x] ✓ Document the 19-language script list vs 181-language artifact set, and the missing `English/` dir, in docs. (commit: 4a63fd8)
- [~] Historical commit-message inaccuracy: `77b6cf7` says "CC-BY-4.0 LICENSE"; LICENSE is MIT since initial commit and `.aii/config.yaml` correctly says MIT. No file change needed — noted in REVIEW_LOG.
- [~] Dead local `output_file_name`/`output_file_path` in `2_write_intro_letter.py::process_introduction_letters()`. Re-checked in the 2026-11-01 pass: both are used by the existing-output guard, so this was a stale finding and no change is needed.
- [x] ✓ `utils.setup_logging()` removed in the 2026-11-01 pass together with the other unused helpers (`setup_llm`, `batch_process_llm`, `extract_entity_files`, `extract_organization_files`, `config.read_api_key`).

## Medium

- [x] ✓ Remove hardcoded private local paths from live scripts; derive `base_dir` from `__file__` like the other scripts. — `3_translate_intro_letter.py`, `4_translate_call_to_action.py` (commit: 00a1581)
- [x] ✓ Stop importing-time API-key read in `config.py` (module import must not require `LLM_keys.key`; keys are resolved lazily via `utils.get_api_key()`). (commit: 00a1581)
- [x] ✓ Add dependency manifests: `GEN-24/requirements.txt` (`openai>=1.0`) and `NHIM/requirements.txt` (grounded in actual imports). (commit: 00a1581)
- [x] ✓ Add `docs/` documentation set with index: project overview, pipeline guide, data layout, NHIM notes, notebook notes. (commit: 4a63fd8)
- [x] ✓ Add CONTRIBUTING.md, SECURITY.md, CITATION.cff (code-adjacent docs; all content grounded in the repo). (commit: 4a63fd8)
- [x] ✓ Add `GEN-24/Inputs_and_Outputs/README.md` (folder-by-folder reference). (commit: 4a63fd8)
- [x] ✓ Add `GEN-24/Archive/README.md` marking the folder historical/non-runnable and flagging outdated local paths. (commit: 4a63fd8)
- [x] ✓ Cross-link the duplicated source call-to-action text (`Entity/Shyaka/ReclAIMEd_Call_To_Action.md` vs embedded copy in `4_translate_call_to_action.py`) in docs instead of consolidating. (commit: 4a63fd8)
- [x] ✓ Repair truncated worldview data modules so every tracked `.py` compiles: `Entity/Bancroft/Bancroft.py` (unclosed dict), `Entity/Hipolito/Hipolito.py` (unclosed list comprehension), `Organization/Active_Inference_Institute/Active_Inference_Institute.py` (orphaned fragment wrapped into a method). Duplicate method definitions in Hipolito/AII left intact (files are read as raw prompt text by stage 1; see docs/data-layout.md). (commit: 8f1788f)
- [~] Consolidate the duplicated CTA source text into one file. Deferred: changes script behavior/inputs; requires a pipeline owner decision.
- [~] Scrub hardcoded private paths in `Archive/*.py` (8 files, `/home/trim`, `/home/tet`, `Active_InferAnts`). Deferred: archive is historical/non-runnable; flagged in `Archive/README.md`.

## Major

- [x] ✓ Establish a documentation system where none existed: `docs/README.md` index + overview + pipeline + data-layout + NHIM + notebook pages, cross-linked with the README and folder READMEs. (commit: 4a63fd8)
- [x] ✓ Deduplicate `NHIM/NHIM_API.py` (every class defined twice, ~lines 30-570 vs ~677-1160). Completed 2026-11-01 without the heavy runtime: the surviving definitions are the ones Python actually resolved, the deterministic paths are covered by `tests/test_nhim_api.py`, and the language-model paths stay untouched. See `docs/nhim.md`.
- [x] ✓ Add CI and a test suite. Completed 2026-11-01: `tests/` (27 tests, no key/network), `requirements-dev.txt`, `.github/workflows/ci.yml` (compileall, pytest, pinned-parser GNN validation).

## Open / deferred list (not implemented this pass)

1. CTA source consolidation — behavior change, owner decision.
2. Archive path scrubbing — historical files, flagged instead.
3. Duplicate method definitions in Hipolito/AII data modules — left intact because stage 1 reads the files as raw prompt text.
4. Live end-to-end model runs — need a real `LLM_keys.key`; the test suite stubs the client by design.
5. Training/fitting the sketch's neural components (Hopfield memory, language-model responses) — no data or runtime in the repository.
