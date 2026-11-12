# Contributing to GEN24

Thank you for contributing to this exploratory Active Inference Institute repository.

## Before changing files

- Read the relevant documentation in [`docs/`](docs/README.md), especially the [pipeline guide](docs/pipeline.md) for active scripts.
- Keep generated artifacts and source material clearly distinguishable.
- Do not add credentials, local absolute paths, private data, or unverified claims.
- Preserve the MIT license and existing attribution.

## Code and pipeline changes

The active scripts are numbered `GEN-24/1_generate_pro-intro.py` through `GEN-24/4_translate_call_to_action.py`. They use paths relative to the script directory and write under `GEN-24/Inputs_and_Outputs/`.

For Python changes, run the gate arms that cover them:

```bash
./run.sh                  # `fast`: lint, test-fast, docs (the default)
./run.sh test-full        # test-fast plus the GNN conformance run
./run.sh gnn              # only if you touched the GNN spec, model, or pin
```

See the [gate and CI run guide](docs/RUN_GUIDE.md).

If a change affects an LLM stage, do not make a provider request merely for syntax validation. Use a local `GEN-24/LLM_keys.key` only when intentionally exercising the stage, and never commit it.

`NHIM/NHIM_API.py` and `ARCANGEL1_0_2.ipynb` are separate experimental artifacts with heavy dependencies; document any runtime assumptions and avoid claiming validation you did not perform.

## Documentation changes

Update the nearest relevant documentation page when behavior, paths, output names, dependencies, or limitations change. Check relative links locally before submitting. Update `TO-DO.md` and `REVIEW_LOG_*.md` only when the change is part of a documented review pass.

## Pull requests

Use a focused title such as `docs: clarify pipeline outputs` or `fix: derive output path from script directory`. Describe what was verified and identify anything not run because it requires credentials, model downloads, Colab, or other external resources.
