"""Phase 4, Part C: every figure for the report and README, drawn from the saved results (no re-running).

Figures (phase4/outputs/figures/, each as .pdf for LaTeX and .png for the README):
  fig_pareto      accuracy vs GPU cost of all 48 YOLO settings, Pareto front        (Phase 1)
  fig_test        final test scores of every method, 6 cameras, dynamic world        (Phase 3 test eval)
  fig_scaling     score vs number of cameras (6 to 48), per budget                   (Phase 3 test eval)
  fig_training    validation score during training of the RL agents                  (training logs)
  fig_live        real GPU run: planned vs measured GPU time, simulated vs real lateness (Phase 4 live run)
Missing inputs are skipped with a message, so it can run before the live run is done.

Run:  EVAL_TARGET=reference python -u make_figures.py            (test split)
"""
import _paths  # noqa: F401
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _paths import OUT4, PHASE1, RESULTS2, RESULTS3  # noqa: E402

OUT = OUT4 / "figures"
COL_W = 3.4            # one column of a two-column paper, inches
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 200})

NAMES = {"fixed_best": "Fixed best", "count_rule": "Count rule", "periodic_reprofile": "Chameleon-style",
         "budget_sharing": "Budget sharing", "tuned_margin": "Tuned margin", "state_aware_best": "State-aware rule",
         "rl_margin": "PPO (single agent)", "ippo": "IPPO", "mappo": "MAPPO (6 cams)",
         "mappo_cam4-8": "MAPPO (4-8 cams)", "mappo_cam4-16_nocount": "MAPPO (main)"}
COLORS = {"fixed_best": "#9e9e9e", "count_rule": "#bdbdbd", "periodic_reprofile": "#d7ccc8",
          "budget_sharing": "#80cbc4", "tuned_margin": "#4db6ac", "state_aware_best": "#b0bec5",
          "rl_margin": "#7986cb", "ippo": "#ffb74d", "mappo": "#ffcc80", "mappo_cam4-8": "#ffa726",
          "mappo_cam4-16_nocount": "#e65100"}
BUDGET_NAMES = {3.0: "tight (3 ms/cam/s)", 7.0: "medium (7)", 15.0: "generous (15)"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


def fig_pareto():
    p = PHASE1 / "outputs" / "profiles" / "configs_summary.csv"
    if not p.exists():
        return print(f"skip fig_pareto: {p} not found")
    d = pd.read_csv(p)
    par = d[d.pareto].sort_values("gpu_ms_per_video_s")
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for m, mk in zip(sorted(d.model.unique()), ["o", "s", "^"]):
        s = d[d.model == m]
        ax.scatter(s.gpu_ms_per_video_s, s.err_train, s=10, marker=mk, alpha=0.45, label=m)
    ax.plot(par.gpu_ms_per_video_s, par.err_train, "-", color="#e65100", lw=1.2, label=f"Pareto front ({len(par)})")
    ax.set_xscale("log")
    ax.set_xlabel("GPU time per second of video (ms, log scale)")
    ax.set_ylabel("Counting error (train)")
    ax.set_title(f"{len(d)} YOLO settings: {d.gpu_ms_per_video_s.max() / d.gpu_ms_per_video_s.min():.0f}x cost range")
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig_pareto")


def fig_test(split):
    p = RESULTS3 / "eval" / f"eval_{split}_cam6.csv"
    if not p.exists():
        return print(f"skip fig_test: {p} not found")
    d = pd.read_csv(p)
    order = [m for m in ["fixed_best", "count_rule", "periodic_reprofile", "budget_sharing", "state_aware_best",
                         "tuned_margin", "rl_margin", "ippo", "mappo_cam4-16_nocount"] if m in set(d.method)]
    budgets = sorted(d.budget.unique())
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    w = 0.8 / len(order)
    for j, m in enumerate(order):
        s = d[d.method == m].set_index("budget").reindex(budgets)
        x = np.arange(len(budgets)) + (j - (len(order) - 1) / 2) * w
        ax.bar(x, s.score, w, color=COLORS.get(m, "#999"), label=NAMES.get(m, m),
               yerr=[s.score - s.ci_low, s.ci_high - s.score], error_kw={"lw": 0.6, "capsize": 1.2})
    ax.set_xticks(np.arange(len(budgets)), [BUDGET_NAMES.get(b, b) for b in budgets])
    ax.set_ylabel("Score (lower is better)")
    ax.set_title(f"Shared GPU, 6 {split} cameras, 95% CI")
    ax.legend(frameon=False, ncol=3, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "fig_test")


def fig_scaling(split):
    files = sorted((RESULTS3 / "eval").glob(f"eval_{split}_cam*.csv"))
    if not files:
        return print("skip fig_scaling: no eval files")
    d = pd.concat([pd.read_csv(f) for f in files])
    budgets = sorted(d.budget.unique())
    methods = [m for m in ["fixed_best", "budget_sharing", "tuned_margin", "mappo", "mappo_cam4-16_nocount"]
               if m in set(d.method)]
    fig, axes = plt.subplots(1, len(budgets), figsize=(2 * COL_W + 0.3, 1.9), sharey=False)
    for ax, b in zip(np.atleast_1d(axes), budgets):
        for m in methods:
            s = d[(d.method == m) & (d.budget == b)].sort_values("cameras")
            ax.errorbar(s.cameras, s.score, yerr=[s.score - s.ci_low, s.ci_high - s.score], marker="o", ms=3,
                        lw=1.6 if m == "mappo_cam4-16_nocount" else 1, capsize=1.5,
                        color=COLORS.get(m, "#999"), label=NAMES.get(m, m))
        ax.set_xscale("log", base=2)
        cams = sorted(d.cameras.unique())
        ax.set_xticks(cams, [str(c) for c in cams])
        ax.set_xlabel("Number of cameras")
        ax.set_title(f"Budget: {BUDGET_NAMES.get(b, b)}")
    np.atleast_1d(axes)[0].set_ylabel("Score (lower is better)")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=6.3, loc="upper right")
    save(fig, "fig_scaling")


def fig_training():
    runs = [(RESULTS3 / "mappo_cam4-16_nocount" / "training_log.csv", "steps", "MAPPO (main, 4-16 cams)", "#e65100"),
            (RESULTS3 / "mappo" / "training_log.csv", "steps", "MAPPO (6 cams)", "#ffa726"),
            (RESULTS3 / "ippo" / "training_log.csv", "steps", "IPPO (6 cams)", "#5c6bc0"),
            (RESULTS2 / "rl_dynamic_margin" / "training_log.csv", "timesteps", "PPO single agent", "#26a69a")]
    found = [(pd.read_csv(p), x, lab, c) for p, x, lab, c in runs if p.exists()]
    if not found:
        return print("skip fig_training: no training logs")
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for d, x, lab, c in found:
        ax.plot(d[x] / 1e6, d.val_score, marker="o", ms=2.5, lw=1.1, color=c, label=lab)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Validation score")
    ax.set_title("Learning curves (dynamic world)")
    ax.legend(frameon=False)
    save(fig, "fig_training")


def fig_live():
    p = OUT4 / "live" / "live_steps.csv"
    if not p.exists():
        return print(f"skip fig_live: {p} not found (run live_run.py first)")
    d = pd.read_csv(p)
    names = dict(NAMES, mappo="MAPPO (main)")          # in the live run, "mappo" is the main agent
    colors = dict(COLORS, mappo=COLORS["mappo_cam4-16_nocount"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * COL_W + 0.3, 1.9))
    for m, g in d.groupby("method", sort=False):
        a1.scatter(g.planned_ms, g.measured_ms, s=5, alpha=0.6, color=colors.get(m, "#999"), label=names.get(m, m))
    lim = max(d.planned_ms.max(), d.measured_ms.max()) * 1.05
    a1.plot([0, lim], [0, lim], "--", color="#555", lw=0.8)
    a1.set_xlabel("Simulator: planned GPU ms per 2-s step")
    a1.set_ylabel("Real GPU: measured ms")
    a1.set_title("Real YOLO on the GPU vs the simulator")
    a1.legend(frameon=False, fontsize=6)
    s = d.groupby("method", sort=False)[["sim_late", "real_late"]].mean() * 100
    x = np.arange(len(s))
    a2.bar(x - 0.18, s.sim_late, 0.36, color="#b0bec5", label="simulator")
    a2.bar(x + 0.18, s.real_late, 0.36, color="#e65100", label="real GPU")
    a2.set_xticks(x, [names.get(m, m) for m in s.index], rotation=15)
    a2.set_ylabel("Late results (%)")
    a2.set_title("Deadline misses, 5-minute live run")
    a2.legend(frameon=False)
    save(fig, "fig_live")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    for f in (fig_pareto, lambda: fig_test(args.split), lambda: fig_scaling(args.split), fig_training, fig_live):
        f()
    print(f"Figures in {OUT}")


if __name__ == "__main__":
    main()
