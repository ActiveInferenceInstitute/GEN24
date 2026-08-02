# REVIEW LOG — 2026-08-02

Docs-deep review pass for the GEN24 repository (ActiveInferenceInstitute/GEN24).
Scope: documentation review + implementation of scoped improvements. This log records the audit trail.

## Phase 0 — Preflight

- Branch: `main` (default, `refs/remotes/origin/HEAD` -> `origin/main`), not detached.
- Head commit at start: `77b6cf7` "add/update .aii sidecar to 100% + CC-BY-4.0 LICENSE (InstituteOS metadata)".
- `git fetch origin` + `git pull --ff-only origin main` -> already up to date. Working tree was clean.
- Remote: `https://github.com/ActiveInferenceInstitute/GEN24.git`.

### Inventory (tracked files: 595)

| Location | Contents |
|---|---|
| `README.md` | 2-line stub (title + Coda link only) |
| `LICENSE` | MIT License, Copyright (c) 2024 Active Inference Institute |
| `ARCANGEL1_0_2.ipynb` | Colab notebook (~14.5 MB), 2 cells: one Colab badge cell, one ~77k-char code cell; no title/overview |
| `GEN-24/` | Pipeline: `1_generate_pro-intro.py`, `2_write_intro_letter.py`, `3_translate_intro_letter.py`, `4_translate_call_to_action.py`, `config.py`, `utils.py`; `Archive/` (8 legacy scripts); `Inputs_and_Outputs/` (see below) |
| `GEN-24/Inputs_and_Outputs/` | `Entity/` (5 worldview `.py` + 25 context `.md`), `Organization/` (2 `.py` + 9 `.md`), `Domain/` (31 `Synthetic_*.md`), `Prompts/` (1), `Pro-Introduction_Letter/` (1), `Written_Introduction_Letter/` (1), `Translated_Letters/` (181 languages, 181 files), `Call_to_Action/` (29 languages x 11 audiences = 319 files) |
| `NHIM/NHIM_API.py` | 1,316-line research module; heavy deps (spacy, transformers, torch); interactive loop |
| `.aii/config.yaml` | InstituteOS sidecar metadata |
| `.gitignore` | Standard Python ignore rules (already covers `*.key`, `.env`) |

- No `docs/` folder, no AGENTS.md/CLAUDE.md, no CONTRIBUTING/SECURITY/CITATION files, no CI (`.github/` absent), no TODO/ROADMAP file, no requirements.txt.

## Phase 1 — Findings by severity

### Minor

1. README is a 2-line stub with no structure, setup, or usage information.
2. `ARCANGEL1_0_2.ipynb` has no title/overview cell — first visible content is only a Colab badge.
3. Doc/code drift: scripts `2_write_intro_letter.py`, `3_translate_intro_letter.py`, `4_translate_call_to_action.py` raise
   "API key not found in .env or LLM_keys.key file", but `utils.get_api_key()` only reads `LLM_keys.key` (no .env support).
4. Stale comment in `4_translate_call_to_action.py` about a "space after GEN24 space" workaround.
5. Historical note: commit `77b6cf7` message says "CC-BY-4.0 LICENSE" but only touched `.aii/config.yaml`; the LICENSE file
   has been MIT since the initial commit and `.aii/config.yaml` correctly records `license: MIT`. Files are consistent;
   only the commit message is misleading (noted, no action possible on history).
6. `3_translate_intro_letter.py` lists 19 target languages and includes 'English', yet `Translated_Letters/` contains 181
   language folders and no `English/` folder — the repo artifacts were produced with an extended language list; the
   script list is a working subset. Documented, not changed.
7. `2_write_intro_letter.py` `process_introduction_letters()` computes a dead local `output_file_name`/`output_file_path`
   that is never used. Left as-is (no functional impact).
8. `utils.setup_logging()` defaults to log file `fieldshift2.log` — a legacy name from the FieldSHIFT-2 project.
   Function is unused by the live pipeline. Documented, not changed.
9. Filename `GEN-24/Archive/2_write_intro_letters copy.py` contains a space and "copy" suffix.
10. No `LLM_keys.key` entry visible in `.gitignore` (it is covered by the generic `*.key` rule, but an explicit entry documents intent).

### Medium

1. Live scripts `3_translate_intro_letter.py` (lines 12, 98) and `4_translate_call_to_action.py` (line 141) hardcode
   private local paths (`/home/trim/Documents/GitHub/GEN24 /GEN-24/` — including a typo'd space). Non-portable and
   leaks a contributor's local layout into a public repo. Other scripts derive paths from `__file__`.
2. `config.py` calls `read_api_key()` at module import time — importing `config` (directly or via `utils`) raises
   `FileNotFoundError` when `LLM_keys.key` is absent, which also breaks `1_generate_pro-intro.py` even though that
   script never calls the LLM.
3. No dependency manifests anywhere (`openai` required by the pipeline; spacy/transformers/torch/numpy/pandas/matplotlib
   by `NHIM/NHIM_API.py`).
4. No README inside `Inputs_and_Outputs/` or `Archive/`; no guidance on what each folder holds.
5. No CONTRIBUTING.md, SECURITY.md, or CITATION.cff.
6. Source call-to-action text is duplicated: embedded in `4_translate_call_to_action.py` AND present as
   `Entity/Shyaka/ReclAIMEd_Call_To_Action.md` (same text, differing only in apostrophe style). Consolidation deferred
   (would change script behavior); documented with cross-links instead.
7. `Archive/` scripts contain many hardcoded private paths (`/home/trim`, `/home/tet`, `Active_InferAnts/...`) — they are
   historical and non-runnable as-is. Scrubbing deferred; documented in `Archive/README.md` instead.

### Major

1. No documentation system at all: no index, no overview, no usage/pipeline guide, no data-layout reference, no
   architecture notes, no notebook description. Implemented a `docs/` set this pass.
2. `NHIM/NHIM_API.py` defines every class twice (two full copies: ~lines 30-570 and ~677-1160) plus heavy module-level
   model loading. Deferred: deduplicating and validating a 1,316-line ML module requires its runtime deps
   (spacy, transformers, torch, model downloads) and interactive execution; out of scope for a docs pass.
   Documented in `docs/nhim.md` instead.

### Validations run (Phase 1)

- Encoding scan: all 568 `.md` files in `GEN-24/` are valid UTF-8, LF line endings, no NUL bytes.
- `Call_to_Action/`: 29 language dirs x 11 audience files = 319 files; every filename prefix matches its directory.
- `Translated_Letters/`: 181 dirs = 181 files, one `translated_<Language>_written_introduction_letter_Hipolito_and_Friston.md` each.
- `Domain/`: 31 files. `Entity/` + `Organization/`: 7 `.py` worldview modules, 34 context `.md` files.
- Grep for private paths/secrets across tracked files: hits only in the live scripts (fixed) and `Archive/` (documented).
  Notebook contains no keys; only a standard Colab `drive.mount('/content/drive')`.
- Heavy suites: not run (no test suite exists; NHIM/notebook require ML runtimes and Colab/Drive data — out of scope, noted).

## Phase 3 — Implementation summary

See `TO-DO.md` for the itemized status. Additional fixes found while verifying (beyond the Phase 1 list):

- Three worldview data modules were truncated/corrupted and did not compile: `Entity/Bancroft/Bancroft.py` (unclosed dict),
  `Entity/Hipolito/Hipolito.py` (unclosed list comprehension), and
  `Organization/Active_Inference_Institute/Active_Inference_Institute.py` (orphaned fragment without a method header).
  All three were repaired minimally (content preserved; orphaned fragment wrapped into a method), and every tracked
  `.py` file now passes `python -m py_compile`.
- `Entity/Hipolito/Hipolito.py` and `Organization/Active_Inference_Institute/Active_Inference_Institute.py` also contain
  duplicate method definitions (second definition shadows the first). Left intact: the files are also read as raw text
  by pipeline stage 1, so deduplication would change future prompt content. Noted in `docs/data-layout.md` and TO-DO.

Commits (oldest to newest):

- `docs: add docs site ...` — README rewrite, `docs/` set, CONTRIBUTING/SECURITY/CITATION, folder READMEs.
- `fix: ...` — portable paths, lazy API key, accurate error messages, archive filename, `.gitignore`, requirements.
- `docs: ...` — notebook overview cell.
- `chore: ...` — REVIEW_LOG + TO-DO updates with completion refs.

## Phase 4 — Verification

- `python -m py_compile` on all edited `.py` files; `import config` without key file; notebook re-serialized and
  re-validated as nbformat 4; relative markdown links checked against the working tree.
- `git status` clean after push; `git log` on `origin/main` shows pushed commits.
