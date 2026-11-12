# AGENTS.md — `docs/`

- Every page is linked from [`README.md`](README.md) and from the flat
  [`documentation-index.md`](documentation-index.md). Add both links in the same
  change as the page; `python scripts/check_docs_links.py` fails otherwise.
- Describe only checked-in behavior. Label live-LLM and Colab paths as requiring
  credentials or an external environment; do not present them as tested.
- Keep the pipeline, data-layout and GNN pages in step with the scripts and
  [`../NHIM/gnn-pin.json`](../NHIM/gnn-pin.json).
- Never hand-edit a number under [`_generated/`](_generated/README.md). Change
  the producer, then regenerate; the `docs` arm runs
  `python scripts/gate_metrics.py --check`.
- Gate commands belong in [`RUN_GUIDE.md`](RUN_GUIDE.md), as arms of
  [`../run.sh`](../run.sh). Do not restate a command in a second page with a
  different form.
- Parent rules: [../AGENTS.md](../AGENTS.md).
