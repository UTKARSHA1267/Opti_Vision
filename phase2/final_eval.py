"""Phase 2, final evaluation: the numbers for the report.

- TEST cameras (never used for training, checkpoint selection or tuning)
- both environments: static (fixed GPU budget) and dynamic (shared GPU, switching costs)
- every method runs on the SAME scenarios, so each scenario gives one PAIRED difference;
  95% confidence intervals come from bootstrapping those paired differences
- RL: every trained seed found, and the average over seeds (the main RL result)

score = error / mean prior error + RL_DEADLINE_PENALTY x share of late results   (lower is better)

Outputs (phase2/outputs/results_<target>/final/):
  final_<env>.csv     one row per (budget, method): score, error, late share, GPU, with 95% CI of the score
  paired_<env>.csv    RL vs each baseline: mean difference with 95% CI and a verdict

Run:  python -u final_eval.py                  (both environments, test split)
      python -u final_eval.py --env dynamic    (one environment only)
"""
import _paths  # noqa: F401
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from _paths import RESULTS2
from config import BUDGETS_MS_PER_CAM_S, SEED
from dynamic_env import BudgetSharingMargin, DynamicEnv, StateAwareMargin
from evaluate_bounds import SmoothedOracle
from evaluate_probing import run_episode
from evaluate_rl import tune_margin
from p2_config import RL_DEADLINE_PENALTY, STATE_AWARE_MARGINS
from policies import CountRule, FixedBest, PeriodicReprofile
from predictability_check import PastOracle
from rl_env import BudgetSharing, RLPolicy
from traffic_env import MultiCameraEnv, SegmentTable

OUT = RESULTS2 / "final"
N_BOOT = 10_000

# Which trained agents to look for, per environment: (folder prefix, label). Seeds: prefix, prefix_s1, ...
AGENTS = {"static": [("rl", "rl_steer")],
          "dynamic": [("rl_dynamic_margin", "rl_margin"), ("rl_dynamic", "rl_steer")]}
MAIN_RL = {"static": "rl_steer", "dynamic": "rl_margin"}
COMPARE_WITH = ["fixed_best", "count_rule", "periodic_reprofile", "budget_sharing", "tuned_margin",
                "state_aware_best"]


def find_seeds(prefix):
    dirs = sorted(d for d in RESULTS2.glob(prefix + "*")
                  if (d.name == prefix or d.name.startswith(prefix + "_s")) and (d / "ppo_best.zip").exists())
    return dirs


def load(d):
    model = PPO.load(d / "ppo_best.zip", device="cpu")
    with open(d / "vecnorm_best.pkl", "rb") as f:
        vn = pickle.load(f)
    vn.training = False
    return model, vn


def bootstrap_ci(x, rng):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_method(env, make, scenarios, budget, scale):
    rows = []
    for k, sc in enumerate(scenarios):
        r = run_episode(env, make(), sc, budget, seed=SEED + k)
        r["score"] = r["err"] / scale + RL_DEADLINE_PENALTY * r["deadline_miss_%"] / 100
        rows.append(r)
    return pd.DataFrame(rows)


def evaluate_env(env_name, table, split, n_scen, rng_boot):
    env_cls = DynamicEnv if env_name == "dynamic" else MultiCameraEnv
    scale = float(np.mean(table.err_prior))
    env = env_cls(table, split=split)
    rng = np.random.default_rng(SEED + 1000)
    scenarios = [env.sample_scenario(rng) for _ in range(n_scen)]

    agents = []
    for prefix, label in AGENTS[env_name]:
        seeds = find_seeds(prefix)
        if seeds:
            agents.append((label, [load(d) for d in seeds], [d.name for d in seeds]))
    print(f"\n######## {env_name} environment, {split} split, {n_scen} scenarios | RL agents: "
          + ("; ".join(f"{lab} ({len(m)} seed{'s' if len(m) > 1 else ''}: {', '.join(n)})" for lab, m, n in agents)
             or "none found"), flush=True)

    summary, paired = [], []
    for budget in BUDGETS_MS_PER_CAM_S:
        t0 = time.time()
        tuned = tune_margin(table, env_cls, budget, scale)
        methods = [("fixed_best", FixedBest), ("count_rule", CountRule), ("periodic_reprofile", PeriodicReprofile),
                   ("budget_sharing", BudgetSharing), ("tuned_margin", lambda m=tuned: BudgetSharingMargin(m))]
        if env_name == "dynamic":
            methods += [(f"state_aware{int(100 * m)}", (lambda m=m: StateAwareMargin(m))) for m in STATE_AWARE_MARGINS]
        methods += [("past_oracle", PastOracle), ("smoothed_oracle", SmoothedOracle)]

        res = {name: run_method(env, make, scenarios, budget, scale) for name, make in methods}
        if env_name == "dynamic":       # the better state-aware rule on THIS data is an optimistic baseline
            sa = [n for n in res if n.startswith("state_aware")]
            res["state_aware_best"] = res[min(sa, key=lambda n: res[n].score.mean())]
        for label, models, names in agents:
            per_seed = []
            for (model, vn), nm in zip(models, names):
                df = run_method(env, lambda model=model, vn=vn: RLPolicy(model, vn), scenarios, budget, scale)
                res[f"{label}[{nm}]"] = df
                per_seed.append(df)
            res[label] = sum(per_seed) / len(per_seed)          # average over seeds, scenario by scenario

        print(f"\n=== budget {budget:g} ms/cam/s | tuned margin (chosen on training cameras): {int(100 * tuned)}% "
              f"| {time.time() - t0:.0f} s", flush=True)
        for name, df in res.items():
            m, lo, hi = bootstrap_ci(df.score, rng_boot)
            row = {"env": env_name, "split": split, "budget": budget, "method": name, "score": m, "score_ci_low": lo,
                   "score_ci_high": hi, "err": df.err.mean(), "late_%": df["deadline_miss_%"].mean(),
                   "gpu_ms_per_cam_s": df.gpu_ms_per_cam_s.mean(),
                   "switches_per_cam_min": df.switches_per_cam_min.mean()}
            summary.append(row)
            print(f"  {name:34s} score {m:6.3f} [{lo:6.3f}, {hi:6.3f}] | error {row['err']:.4f} | late {row['late_%']:5.1f}% "
                  f"| GPU {row['gpu_ms_per_cam_s']:5.2f}", flush=True)

        main = MAIN_RL[env_name]
        if main in res:
            print(f"  --- paired comparison: {main} minus baseline (negative = RL better), 95% CI ---")
            for base in COMPARE_WITH:
                if base not in res:
                    continue
                d = res[main].score.to_numpy() - res[base].score.to_numpy()
                m, lo, hi = bootstrap_ci(d, rng_boot)
                rel = 100 * m / res[base].score.mean()
                verdict = "RL better" if hi < 0 else ("RL worse" if lo > 0 else "no significant difference")
                paired.append({"env": env_name, "split": split, "budget": budget, "rl": main, "baseline": base,
                               "diff": m, "ci_low": lo, "ci_high": hi, "relative_%": rel, "verdict": verdict})
                print(f"  {main} vs {base:18s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({rel:+5.1f}%)  -> {verdict}", flush=True)
    return pd.DataFrame(summary), pd.DataFrame(paired)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--env", default="both", choices=["both", "static", "dynamic"])
    ap.add_argument("--scenarios", type=int, default=30)
    args = ap.parse_args()
    table = SegmentTable()
    rng_boot = np.random.default_rng(SEED + 2000)
    OUT.mkdir(parents=True, exist_ok=True)
    for env_name in (["static", "dynamic"] if args.env == "both" else [args.env]):
        summary, paired = evaluate_env(env_name, table, args.split, args.scenarios, rng_boot)
        summary.round(4).to_csv(OUT / f"final_{env_name}_{args.split}.csv", index=False)
        paired.round(4).to_csv(OUT / f"paired_{env_name}_{args.split}.csv", index=False)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
