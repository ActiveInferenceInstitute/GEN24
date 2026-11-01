"""Shared pytest configuration: make the repository's modules importable."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "GEN-24", ROOT / "NHIM"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
