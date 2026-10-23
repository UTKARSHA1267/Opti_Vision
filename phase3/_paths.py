"""Lets Phase 3 scripts use Phase 1 and Phase 2 code, config and outputs.

Import this first in every Phase 3 script:   import _paths  # noqa: F401
(Phase 2 modules also do `import _paths`; they get this file, so it provides Phase 2's paths too.)
"""
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parent
PHASE2 = PHASE3.parent / "phase2"
PHASE1 = PHASE3.parent / "phase1"
for p in (PHASE2, PHASE1):
    if str(p) not in sys.path:
        sys.path.insert(1, str(p))

from config import EVAL_TARGET  # noqa: E402

OUT2 = PHASE2 / "outputs"                       # Phase 2 outputs (features, probes, trained PPO agents)
RESULTS2 = OUT2 / f"results_{EVAL_TARGET}"
OUT3 = PHASE3 / "outputs"
RESULTS3 = OUT3 / f"results_{EVAL_TARGET}"
