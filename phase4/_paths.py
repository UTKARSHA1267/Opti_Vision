"""Lets Phase 4 scripts use Phase 1, 2 and 3 code, config and outputs.

Import this first in every Phase 4 script:   import _paths  # noqa: F401
(Phase 2 and 3 modules also do `import _paths`; they get this file, so it provides their paths too.)
"""
import sys
from pathlib import Path

PHASE4 = Path(__file__).resolve().parent
PHASE3 = PHASE4.parent / "phase3"
PHASE2 = PHASE4.parent / "phase2"
PHASE1 = PHASE4.parent / "phase1"
for p in (PHASE3, PHASE2, PHASE1):
    if str(p) not in sys.path:
        sys.path.insert(1, str(p))

from config import EVAL_TARGET  # noqa: E402

OUT2 = PHASE2 / "outputs"
RESULTS2 = OUT2 / f"results_{EVAL_TARGET}"
OUT3 = PHASE3 / "outputs"
RESULTS3 = OUT3 / f"results_{EVAL_TARGET}"
OUT4 = PHASE4 / "outputs"
