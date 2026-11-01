# GNN integration

GEN24 contains exactly one Active Inference generative model: the **NHIM
inquiry agent**. It is specified in [Generalized Notation Notation][gnn]
(GNN) and implemented once, in NumPy.

| Artifact | Path |
|---|---|
| GNN specification | [`NHIM/nhim_inquiry_agent.gnn.md`](../NHIM/nhim_inquiry_agent.gnn.md) |
| Implementation (single home) | [`NHIM/nhim_model.py`](../NHIM/nhim_model.py) |
| Upstream pin record | [`NHIM/gnn-pin.json`](../NHIM/gnn-pin.json) |
| Validator | [`scripts/validate_gnn_spec.py`](../scripts/validate_gnn_spec.py) |
| Tests | [`tests/test_gnn_spec.py`](../tests/test_gnn_spec.py) |

The rest of the repository defines no generative model: `GEN-24/` is a
prompt-and-file transformation pipeline, and the notebook is experiment code.
Nothing in `GEN-24/` is described as a generative model in these docs, and no
`.gnn.md` file is claimed for it.

## The model

A flat discrete POMDP over an inquiry loop; there is no hierarchy and no
continuous state.

| Element | Meaning | Size |
|---|---|---|
| `s` | inquiry context: `grounded`, `uncertain`, `conflicted` | 3 |
| `o` | sense data: `aligned`, `mixed`, `contradictory` | 3 |
| `u` | response mode: `attune`, `reframe`, `defer` | 3 |
| `A` | likelihood `P(o \| s)`, asymmetric (0.70 on the diagonal) | 3×3 |
| `B` | transition `P(s' \| s, u)`, `attune` identity, `reframe` shifts toward grounded, `defer` mixes | 3×3×3 |
| `C` | log-preferences `[1.0, 0.0, −1.0]`: aligned preferred, contradictory avoided | 3 |
| `D` | uniform prior over contexts | 3 |

`NHIM/nhim_model.py` adds the derived quantities the agent loop uses:
variational free energy, expected free energy per action, a softmax policy
posterior, action selection, and a bounded distress scalar driven by belief
uncertainty.

Provenance: `NHIM/NHIM_API.py` initializes its placeholders
(`P_d_given_s`, `P_s_prime_given_s_a`, `P_s`) with `numpy.random.rand` and
never calibrates them, so no reproducible model existed in the repository. The
specification replaces those placeholders with the documented values above;
the sketch's Hopfield memory and language-model response generation are
integration components and stay outside the model.

## The pin

The specification is validated against a pinned revision of the upstream
repository, never against a live working tree:

- Repository: `ActiveInferenceInstitute/Generalized_Notation_Notation`
- Release: `v5.0.3`, commit `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`
  (`Merge pull request #292 ...`, 2026-10-31)
- Parser entrypoint: `gnn.parsers.markdown_parser.MarkdownGNNParser`
- Recorded SHA-256 for every parser file the validator depends on
  (`src/gnn/parsers/*`, `src/gnn/schema/parser.py`, `docs/gnn/gnn_syntax.md`)

[`NHIM/gnn-pin.json`](../NHIM/gnn-pin.json) is the record; treat it as the
contract between this repository and the upstream parser.

## Run the validator

```bash
pip install -r requirements-dev.txt          # the upstream parser needs defusedxml
python scripts/validate_gnn_spec.py          # --gnn-root PATH, or $GNN_ROOT
```

The checkout is resolved from `--gnn-root`, then `$GNN_ROOT`, then the sibling
directory `../Generalized_Notation_Notation`. The validator:

1. checks the pin record is complete;
2. checks the pinned commit is present and the release tag points at it;
3. re-hashes every recorded parser file from that commit;
4. extracts `src/gnn` from the pinned commit into a temporary directory and
   imports the parser from *there* (never from a working tree or an installed
   package);
5. parses the specification, checks its structure and the 13 declared
   variables, compares `A`/`B`/`C`/`D` numerically with `NHIM/nhim_model.py`,
   and checks that `A`'s columns and every `B` action slice are stochastic.

Exit codes: `0` on success, `1` on a failed check, `2` when the checkout is
absent and `--require-gnn` was passed. Without `--require-gnn` an absent
checkout prints `SKIPPED` and exits `0`, so the script is safe in environments
that cannot fetch the upstream repository. CI fetches the pinned commit and
runs the validator with `--require-gnn`; the `pytest` run treats the
parser-level test as skipped when no checkout is available.

## Change the model

1. Edit the specification and `NHIM/nhim_model.py` together. The validator
   compares matrices numerically, so a change in one without the other fails.
2. Keep `A` and every `B` action slice column-stochastic.
3. Re-run `python scripts/validate_gnn_spec.py --require-gnn`; the run prints
   the specification's SHA-256, which is informational (it is not pinned).
4. If the upstream pin moves, update `commit`/`release` in
   `NHIM/gnn-pin.json`, refresh **every** `parser_files` hash from the new
   commit, and re-run the validator.

## What this does not claim

- No fitting, training, or empirical calibration: the matrices are documented
  choices, not estimated parameters.
- No live model calls. The pipeline's model-backed stages need
  `GEN-24/LLM_keys.key` and are not exercised by the tests.
- The GNN package is not imported at runtime by GEN24; it is a validation-time
  dependency only.

## Related repositories

- [Generalized_Notation_Notation][gnn] — the specification language and parser.
  See its [integration guide][integration] and
  [implementation guide][implementation].
- [Active_Inference_Ontology](https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology)
  — the ontology terms used in the spec's `ActInfOntologyAnnotation` section.
- [active_inference_language_model](https://github.com/ActiveInferenceInstitute/active_inference_language_model)
  — another fleet repository that specifies models in GNN.

[gnn]: https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation
[integration]: https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/framework_integration_guide.md
[implementation]: https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/gnn_implementation.md
