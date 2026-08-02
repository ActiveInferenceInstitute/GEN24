# NHIM module

`NHIM/NHIM_API.py` is a separate experimental module. It imports NumPy, pandas, Matplotlib, spaCy, and Hugging Face Transformers, loads `distilbert-base-cased-distilled-squad`, `en_core_web_sm`, and `gpt2` at module import time, and exposes components for response generation, attention, memory, reinforcement learning, distress dynamics, Bayesian updates, and problem-solving inquiries.

Its `handle_user_queries()` function provides an interactive terminal loop and runs when the file is executed directly.

## Important runtime notes

- The module has heavyweight dependencies; install them from [NHIM/requirements.txt](../NHIM/requirements.txt).
- The model and spaCy data downloads are not vendored in this repository.
- Importing the module loads models immediately, so a working ML environment and model cache are required even for basic imports.
- There is no test suite or CI workflow for NHIM in this repository.

## Review status

The file currently contains two full definitions of its principal classes. The second definition replaces the first in Python name resolution, but deduplicating the file safely requires runtime validation of the experimental behavior and model dependencies. This is tracked as deferred work in [TO-DO.md](../TO-DO.md), rather than being changed during a documentation-focused pass.

NHIM is not part of the four-stage `GEN-24/` pipeline.
