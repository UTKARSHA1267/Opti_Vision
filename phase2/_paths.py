"""Lets Phase 2 scripts use Phase 1's code, config and outputs.

Import this first in every Phase 2 script:   import _paths  # noqa: F401
"""
import sys
from pathlib import Path

PHASE2 = Path(__file__).resolve().parent
PHASE1 = PHASE2.parent / "phase1"
OUT2 = PHASE2 / "outputs"

if str(PHASE1) not in sys.path:
    sys.path.insert(0, str(PHASE1))

# Results that depend on the evaluation target (labels / reference) go in separate folders.
from config import EVAL_TARGET  # noqa: E402
RESULTS2 = OUT2 / f"results_{EVAL_TARGET}"