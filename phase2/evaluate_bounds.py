"""Phase 2, Part B (1): where does the oracle's gain come from?

Compares, on the SAME validation scenarios as Phase 1's evaluate_baselines.py:
  fixed_best        one setting for every camera (today's approach)
  count_rule        simple adaptive rule
  static_oracle     each camera gets ONE best setting for the whole episode (hindsight, within budget)
                    -> gain from per-camera differences only
  smoothed_oracle   per segment, using true errors averaged over +-SMOOTH_WINDOW segments
                    -> gain from real changes over time, with most of the 2-second luck removed
  oracle            per segment, true errors of that exact segment (Phase 1 upper bound, includes luck)

"share of oracle gain" = (fixed - method) / (fixed - oracle)

Output: phase2/outputs/results_<target>/bounds_<split>.csv

Run:  python -u evaluate_bounds.py
"""
import _paths  # noqa: F401
import argparse
import warnings

import numpy as np
import pandas as pd

from _paths import RESULTS2
from allocation import allocate
from config import BUDGETS_MS_PER_CAM_S, EVAL_SCENARIOS, SEED
from evaluate_baselines import run_episode
from policies import CountRule, FixedBest, HindsightOracle, Policy
from traffic_env import MultiCameraEnv, SegmentTable

SMOOTH_WINDOW = 5   # +-5 segments = +-10 seconds
warnings.filterwarnings("ignore", message="Mean of empty slice")


def episode_errors(env):
    """True errors for every step of the episode: (horizon, n_cameras, n_settings)."""
    rows = np.stack([env._rows(t) for t in range(env.horizon)])
    return env.tab.err[rows]


class StaticOracle(Policy):
    name = "static_oracle"

    def reset(self, env):
        E = np.nanmean(episode_errors(env), axis=0)
        E = np.where(np.isnan(E), env.tab.err_prior[None, :], E)
        self.choice = allocate(E, env.costs, env.scheduler.limit(env.budget), env.weights)

    def act(self, obs, env):
        return {"proposal": self.choice.copy()}


class SmoothedOracle(Policy):
    name = "smoothed_oracle"

    def __init__(self, window=SMOOTH_WINDOW):
        self.window = window

    def reset(self, env):
        e = episode_errors(env)
        H = len(e)
        sm = np.empty_like(e)
        for t in range(H):
            sm[t] = np.nanmean(e[max(0, t - self.window):min(H, t + self.window + 1)], axis=0)
        self.smoothed = np.where(np.isnan(sm), env.tab.err_prior[None, None, :], sm)

    def act(self, obs, env):
        cap = env.scheduler.limit(env.budget, env.backlog)
        return {"proposal": allocate(self.smoothed[obs["t"]], env.costs, cap, env.weights)}


METHODS = [FixedBest, CountRule, StaticOracle, SmoothedOracle, HindsightOracle]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--scenarios", type=int, default=EVAL_SCENARIOS)
    args = ap.parse_args()

    table = SegmentTable()
    env = MultiCameraEnv(table, split=args.split)
    rng = np.random.default_rng(SEED)
    scenarios = [env.sample_scenario(rng) for _ in range(args.scenarios)]   # same as Phase 1

    rows = []
    for budget in BUDGETS_MS_PER_CAM_S:
        for P in METHODS:
            res = [run_episode(env, P(), sc, budget, seed=SEED + k) for k, sc in enumerate(scenarios)]
            rows.append({"budget_ms_per_cam_s": budget, "method": P.name,
                         **{k: float(np.nanmean([r[k] for r in res])) for k in res[0]}})
        d = pd.DataFrame([r for r in rows if r["budget_ms_per_cam_s"] == budget])
        fixed = d[d.method == "fixed_best"].err.iloc[0]
        oracle = d[d.method == "oracle"].err.iloc[0]
        for r in rows:
            if r["budget_ms_per_cam_s"] == budget:
                r["err_vs_fixed_%"] = 100 * (r["err"] - fixed) / fixed
                r["share_of_oracle_gain_%"] = 100 * (fixed - r["err"]) / (fixed - oracle) if fixed > oracle else np.nan
        print(f"budget {budget:5.1f}")
        for r in rows[-len(METHODS):]:
            print(f"  {r['method']:16s} error {r['err']:.4f} ({r['err_vs_fixed_%']:+6.1f}% vs fixed) | "
                  f"share of oracle gain {r['share_of_oracle_gain_%']:6.1f}% | worst cam {r['worst_cam_err']:.4f} | "
                  f"GPU {r['gpu_ms_per_cam_s']:6.2f} ms/cam/s | deadline miss {r['deadline_miss_%']:4.1f}%", flush=True)

    RESULTS2.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(RESULTS2 / f"bounds_{args.split}.csv", index=False)
    print(f"\nSaved {RESULTS2 / f'bounds_{args.split}.csv'}")


if __name__ == "__main__":
    main()