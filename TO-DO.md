# TO-DO — GEN24 documentation pass

Last reviewed: 2026-11-12 (GNN pin, gate-inventory and Education-blocker accuracy
pass) after 2026-11-01

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

## Gate and documentation pass (2026-11-12)

- [x] ✓ Added `run.sh` as the single gate entry point with separately runnable
  primitive arms (`lint`, `test-fast`, `docs`, `gnn`) and compositions
  (`fast` = lint + test-fast + docs, the default; `test-full` = test-fast + gnn;
  `all` = fast + gnn). Each primitive runs once per invocation. There is no
  `types` arm: the repository has no type checker.
- [x] ✓ Rewrote `.github/workflows/ci.yml` so every gate step is a `run.sh` arm,
  adopted the organization composite action
  `ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`,
  and removed the duplicated work: the second dependency install now shares one
  uv cache, and the GNN validator runs once instead of twice.
- [x] ✓ Removed the `test_validator_passes_against_the_pinned_parser` pytest
  wrapper. It shelled out to the same validator command the `gnn` arm runs, so a
  gate executed the validator twice. The suite was 26 tests and fully offline at
  that point; `#17` then added four checkout-resolution tests, so the current
  count is 30 (`docs/_generated/GATE_METRICS.md`, re-checked by the `docs` arm).
- [x] ✓ Added `--require-tag` to `scripts/validate_gnn_spec.py` so the local arm
  enforces the release-tag check that previously existed only as a CI shell
  step; the workflow no longer carries a second copy of it.
- [x] ✓ Added `pytest.ini` to pin the rootdir and `testpaths`, so collected node
  IDs do not depend on a parent directory's pytest configuration.
- [x] ✓ Added `scripts/check_docs_links.py` (relative links, hub coverage, flat
  index coverage) and `scripts/gate_metrics.py` (generated
  `docs/_generated/GATE_METRICS.md` with `--check`).
- [x] ✓ Added `docs/RUN_GUIDE.md`, `docs/documentation-index.md` and
  `docs/_generated/`, and updated the hub, the GNN page, the pipeline page, the
  folder guides and the root README/AGENTS/CONTRIBUTING gate instructions.
- [x] ✓ Measured every arm before and after on this host; the numbers are
  recorded in `docs/RUN_GUIDE.md` and refreshed by `scripts/gate_metrics.py`.
- [x] ✓ Fixed GNN checkout precedence in `scripts/validate_gnn_spec.py`: an
  explicit `--gnn-root` wins, then `GNN_ROOT`, then the sibling
  `Generalized_Notation_Notation` checkout. The environment variable used to be
  inserted ahead of the explicit argument, so `GNN_ROOT` silently overrode it.
  The offline pin test now asserts the pin `schema` and `release`, and four
  parametrized tests cover the resolution order — the suite went from 26 to 30
  tests. The pin's re-pin note points at `./run.sh gnn` instead of a bare
  validator call. Merged as `#17` (`e87025e`).
- [x] ✓ Re-read the gate against the tree on 2026-11-12: `./run.sh docs` reports
  `OK: 625 Markdown files, relative links resolve, docs index complete` and
  `OK: gate metrics current (30 tests, 4 primitive arms, 3 compositions)`.

## Open / deferred list (not implemented this pass)

1. CTA source consolidation — behavior change, owner decision.
2. Archive path scrubbing — historical files, flagged instead.
3. Duplicate method definitions in Hipolito/AII data modules — left intact because stage 1 reads the files as raw prompt text.
4. Live end-to-end model runs — need a real `LLM_keys.key`; the test suite stubs the client by design.
5. Training/fitting the sketch's neural components (Hopfield memory, language-model responses) — no data or runtime in the repository.
6. Education bundle `specs/education/gen24.json` — **blocked on authored
   learning objectives that do not exist in this repository.** The contract row
   (https://github.com/ActiveInferenceInstitute/courses/blob/main/docs/workflows/education.md,
   line 189) requires one unit for `GEN-24/` and one for `NHIM/`; the schema
   (`specs/education/schema.json`, `$defs.unit.objectives`) requires
   `minItems: 1`, and `specs/education/contract.json` names GEN24 with the
   adoption pointer `education-contract.json`. Neither that pointer nor any
   `specs/` directory exists here, so no bundle can be emitted and the
   membership stays unadopted. A search of every tracked Markdown file for an
   objectives/outcomes/goals heading returns nothing, and `GEN-24/README.md`,
   `docs/pipeline.md`, `docs/nhim.md`, `NHIM/README.md` and `docs/overview.md`
   describe implementation and operation only. The missing artefacts are
   authored learning objectives for the `GEN-24/` pipeline unit (in
   `GEN-24/README.md` or `docs/pipeline.md`) and for the `NHIM/` inquiry-model
   unit (in `NHIM/README.md` or `docs/nhim.md`). Unblocked by: the GEN24
   maintainer who authors those guides. Implementation descriptions must not be
   relabelled as objectives to satisfy the schema.
7. PR https://github.com/ActiveInferenceInstitute/GEN24/pull/16 (`workflows/migration`) —
   holds the finished pin-spelling normalization (`revision`/`tag`, matching
   `education.md:149`) plus the blocked-migration record, and is deliberately
   unmerged. It now conflicts with `main` on `NHIM/gnn-pin.json`, because `#17`
   rewrote the pin's re-pin note; `git merge-tree --write-tree origin/main
   workflows/migration` reports the content conflict. Unblocked by: whoever
   resumes the migration — rebase it, or split the pin change into its own PR.
8. Re-pin from GNN `v5.0.3` (`8f293d9b`) to `v5.1.0`
   (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, the tag present in the upstream
   checkout), which the rest of the importer fleet already carries. Nothing
   external blocks it: the recorded procedure is to update `commit`/`release`
   and every `parser_files` hash from the new commit, then run `./run.sh gnn`
   (which needs the checkout and its tag). Unblocked by: a GEN24 maintainer with
   an upstream checkout reachable via `GNN_ROOT` or as a sibling directory.

Not applicable here: GEN24 has no `inferactively-pymdp` dependency at all, so
the fleet's `1.0.5` floor has nothing to pin — `NHIM/nhim_model.py` imports NumPy
only and no requirements file names pymdp. Recorded so a later pass does not read
the absence as a missing dependency.
