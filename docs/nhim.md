# NHIM module

`NHIM/` holds a problem-solving-agent sketch and the canonical model extracted
from it.

| File | Role |
|---|---|
| [`nhim_inquiry_agent.gnn.md`](../NHIM/nhim_inquiry_agent.gnn.md) | GNN specification of the inquiry loop (the repository's only generative model) |
| [`nhim_model.py`](../NHIM/nhim_model.py) | NumPy implementation of that model: filter, free energies, action selection, distress dynamics |
| [`NHIM_API.py`](../NHIM/NHIM_API.py) | The sketch: inquiry loop, attention gating, Hopfield memory, optional language-model responses |
| [`gnn-pin.json`](../NHIM/gnn-pin.json) | Pinned upstream parser revision, parser file hashes |
| [`requirements.txt`](../NHIM/requirements.txt) | Sketch extras (spaCy, Transformers, torch, pandas, Matplotlib) |

See [GNN integration](gnn.md) for the specification, the pin, and the
validator.

## Import and dependency behaviour

- `nhim_model.py` needs NumPy only and has no import-time side effects.
- `NHIM_API.py` imports NumPy and `nhim_model` eagerly. spaCy, Transformers,
  and Matplotlib load lazily, inside the code paths that use them, so
  importing the module and running the deterministic loop works with NumPy
  alone.
- The large model and spaCy data downloads (`distilbert-base-cased-distilled-squad`,
  `en_core_web_sm`, `gpt2`) are still required for the language-model response
  path; they are not vendored.
- `handle_user_queries()` remains the interactive terminal loop, and it runs
  only under `if __name__ == "__main__":`.
- Every class and function is defined once. The earlier file contained two full
  copies of each class, with the second shadowing the first; the surviving
  definition is the one Python actually used, and the duplicated code was
  removed.

The deterministic paths are covered by [`tests/test_nhim_api.py`](../tests/test_nhim_api.py)
and [`tests/test_nhim_model.py`](../tests/test_nhim_model.py). The
language-model and attention paths that need the heavy downloads are not
exercised by the test suite, and no result from them is claimed.

NHIM is not part of the four-stage `GEN-24/` pipeline; the two share no code.
