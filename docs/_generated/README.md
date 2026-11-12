# `docs/_generated/` — generated facts

Files here are produced by a named script and are checked by the `docs` arm of
[`../../run.sh`](../../run.sh). Never hand-edit them: change the producer, then
regenerate.

| File | Producer | Refresh | Checked by |
|---|---|---|---|
| [`GATE_METRICS.md`](GATE_METRICS.md) | [`../../scripts/gate_metrics.py`](../../scripts/gate_metrics.py) | `python scripts/gate_metrics.py` | `python scripts/gate_metrics.py --check` |

`GATE_METRICS.md` carries two kinds of content. Its **facts** block (arm list,
collected test count, interpreter major.minor) is deterministic, and `--check`
recomputes it and fails when the committed copy disagrees. Its **timings**
block is an observation of one run per arm on the machine that refreshed it;
`--check` only requires the block to be present, because wall time is not a
property the repository controls.

Prose elsewhere should link this directory instead of copying the numbers.
