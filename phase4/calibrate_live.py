"""Recompute the live run's lateness with a calibrated GPU capacity (no rerun needed).

The simulator's costs come from a batch-64 benchmark; live steps have smaller batches, so real GPU inference took
longer. A real deployment sizes its budget to its own measured speed. Here ONE factor, the same for every method,
is measured from the live run itself: factor = total measured GPU ms / total planned ms (all methods pooled).
The real capacity of every step is multiplied by it and the backlog and deadline check is replayed step by step.

Input:  live_steps.csv from live_run.py
Output: live_calibrated.csv and fig_live_calibrated.pdf/.png next to it

Run:  python calibrate_live.py /DATA2/nilesh/live_backup/live/live_steps.csv
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEADLINE_FRACTION = 1.0     # a step is late if its work plus backlog exceeds one step of capacity
NAMES = {"fixed_best": "Fixed best", "budget_sharing": "Budget sharing", "tuned_margin": "Tuned margin",
         "mappo": "MAPPO (main)"}
COLORS = {"fixed_best": "#9e9e9e", "budget_sharing": "#80cbc4", "tuned_margin": "#4db6ac", "mappo": "#e65100"}


def replay(g, factor):
    backlog, late = 0.0, []
    for work, cap in zip(g.measured_ms, g.capacity_ms * factor):
        late.append(backlog + work > cap * DEADLINE_FRACTION)
        backlog = max(0.0, backlog + work - cap)
    return np.mean(late)


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "live_steps.csv")
    d = pd.read_csv(path)
    factor = d.measured_ms.sum() / d.planned_ms.sum()
    rows = []
    for m, g in d.groupby("method", sort=False):
        rows.append({"method": m, "err": g.err.mean(), "gpu_planned_ms": g.planned_ms.mean(),
                     "gpu_measured_ms": g.measured_ms.mean(), "late_simulator_%": 100 * g.sim_late.mean(),
                     "late_real_calibrated_%": 100 * replay(g, factor)})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(path.parent / "live_calibrated.csv", index=False)
    print(f"Calibration factor (measured / planned GPU, all methods pooled): {factor:.2f}\n")
    print(out.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(out))
    ax.bar(x - 0.18, out["late_simulator_%"], 0.36, color="#b0bec5", label="simulator")
    ax.bar(x + 0.18, out["late_real_calibrated_%"], 0.36, color=[COLORS.get(m, "#999") for m in out.method],
           label=f"real GPU (capacity x{factor:.1f})")
    for xi, v in zip(x, out["late_real_calibrated_%"]):
        ax.text(xi + 0.18, v + 1, f"{v:.0f}%", ha="center", fontsize=6.5)
    ax.set_xticks(x, [NAMES.get(m, m) for m in out.method], rotation=15, fontsize=7)
    ax.set_ylabel("Late results (%)", fontsize=8)
    ax.set_title("Deadline misses, 5-minute live run on a real GPU", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(path.parent / f"fig_live_calibrated.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nSaved live_calibrated.csv and fig_live_calibrated.pdf/.png in {path.parent}")


if __name__ == "__main__":
    main()
