"""Phase 3, Part A (4): evaluate MAPPO.

Dynamic environment. Every method runs on the SAME scenarios; paired 95% confidence intervals.
Methods:
  traditional:  fixed_best, count_rule, periodic_reprofile
  strong:       budget_sharing, tuned_margin (best fixed margin for this budget AND camera count, chosen on
                training cameras), state_aware_best (best state-aware rule chosen with hindsight)
  Phase 2 RL:   rl_margin (centralised margin controller, average of its seeds) - only with 6 cameras,
                because it was built for exactly 6
  Phase 3 RL:   every trained group found: mappo, ippo, mappo_cam4-8, ... (average over seeds)

score = error / mean prior error + RL_DEADLINE_PENALTY x share of late results   (lower is better)

Use --split val while developing; final numbers: --split test.
Output: phase3/outputs/results_<target>/eval/eval_<split>_cam<N>.csv and paired_<split>_cam<N>.csv

Run:  python -u evaluate_mappo.py                         (validation, 6 cameras)
      python -u evaluate_mappo.py --split test            (final, 6 cameras)
      python -u evaluate_mappo.py --cameras 6 12 24 48    (scaling)
"""
import _paths  # noqa: F401
import argparse
import re
import time

import numpy as np
import pandas as pd

from _paths import RESULTS2, RESULTS3
from config import BUDGETS_MS_PER_CAM_S, SEED
from dynamic_env import BudgetSharingMargin, DynamicEnv, StateAwareMargin
from evaluate_probing import run_episode
from ma_env import MAPPOPolicy
from mappo import load
from p2_config import MARGIN_CHOICES, RL_DEADLINE_PENALTY, STATE_AWARE_MARGINS
from policies import CountRule, FixedBest, PeriodicReprofile
from rl_env import BudgetSharing
from traffic_env import SegmentTable

OUT = RESULTS3 / "eval"
N_BOOT = 10_000
TUNE_SCENARIOS = 12


def score_of(r, scale):
    return r["err"] / scale + RL_DEADLINE_PENALTY * r["deadline_miss_%"] / 100


def bootstrap_ci(x, rng):
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_method(env, make, scenarios, budget, scale):
    rows, margins = [], []
    for k, sc in enumerate(scenarios):
        pol = make()
        r = run_episode(env, pol, sc, budget, seed=SEED + k)
        r["score"] = score_of(r, scale)
        rows.append(r)
        margins += getattr(pol, "margins", [])
    return pd.DataFrame(rows), margins


def tune_margin(table, budget, n, scale):
    env = DynamicEnv(table, split="train")
    env.n = n
    rng = np.random.default_rng(SEED + 7)
    scenarios = [env.sample_scenario(rng) for _ in range(TUNE_SCENARIOS)]
    best = None
    for m in MARGIN_CHOICES:
        s = run_method(env, lambda m=m: BudgetSharingMargin(m), scenarios, budget, scale)[0].score.mean()
        if best is None or s < best[0]:
            best = (s, m)
    return best[1]


def find_groups():
    """Trained MAPPO/IPPO runs grouped by name without the seed suffix: {'mappo': [dirs], ...}."""
    groups = {}
    for d in sorted(RESULTS3.glob("*")):
        if (d / "best.pt").exists():
            groups.setdefault(re.sub(r"_s\d+$", "", d.name), []).append(d)
    return groups


def phase2_agents():
    """The Phase 2 centralised margin controller (all seeds), if Stable-Baselines3 is available."""
    try:
        import pickle
        from stable_baselines3 import PPO
        from rl_env import RLPolicy
    except Exception:
        return None
    dirs = sorted(d for d in RESULTS2.glob("rl_dynamic_margin*") if (d / "ppo_best.zip").exists())
    agents = []
    for d in dirs:
        with open(d / "vecnorm_best.pkl", "rb") as f:
            vn = pickle.load(f)
        vn.training = False
        agents.append((PPO.load(d / "ppo_best.zip", device="cpu"), vn))
    return (agents, RLPolicy) if agents else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cameras", type=int, nargs="+", default=[6])
    ap.add_argument("--scenarios", type=int, default=None, help="default: 20 (val) / 30 (test)")
    args = ap.parse_args()
    n_scen = args.scenarios or (20 if args.split == "val" else 30)

    table = SegmentTable()
    scale = float(np.mean(table.err_prior))
    groups = {g: [load(d / "best.pt") for d in dirs] for g, dirs in find_groups().items()}
    p2 = phase2_agents()
    rng_boot = np.random.default_rng(SEED + 2000)
    OUT.mkdir(parents=True, exist_ok=True)
    print("Trained groups: " + ", ".join(f"{g} ({len(v)} seed{'s' if len(v) > 1 else ''})" for g, v in groups.items())
          + (f" | Phase 2 rl_margin ({len(p2[0])} seeds)" if p2 else " | Phase 2 rl_margin: not loaded"), flush=True)

    for n in args.cameras:
        env = DynamicEnv(table, split=args.split)
        env.n = n
        rng = np.random.default_rng(SEED + 1000 + n)
        scenarios = [env.sample_scenario(rng) for _ in range(n_scen)]
        summary, paired = [], []
        print(f"\n######## {n} cameras, {args.split} split, {n_scen} scenarios", flush=True)
        for budget in BUDGETS_MS_PER_CAM_S:
            t0 = time.time()
            tuned = tune_margin(table, budget, n, scale)
            methods = [("fixed_best", FixedBest), ("count_rule", CountRule), ("periodic_reprofile", PeriodicReprofile),
                       ("budget_sharing", BudgetSharing), ("tuned_margin", lambda m=tuned: BudgetSharingMargin(m))]
            methods += [(f"state_aware{int(100 * m)}", (lambda m=m: StateAwareMargin(m))) for m in STATE_AWARE_MARGINS]
            res, margins = {}, {}
            for name, make in methods:
                res[name], _ = run_method(env, make, scenarios, budget, scale)
            sa = [k for k in res if k.startswith("state_aware")]
            res["state_aware_best"] = res[min(sa, key=lambda k: res[k].score.mean())]
            if p2 and n == 6:
                agents, RLPolicy = p2
                runs = [run_method(env, lambda m=m, v=v: RLPolicy(m, v), scenarios, budget, scale)[0] for m, v in agents]
                res["rl_margin"] = sum(runs) / len(runs)
            for g, models in groups.items():
                runs = []
                for k, (actor, norm, ckpt) in enumerate(models):
                    cc = ckpt.get("info", {}).get("camera_count", True)   # older agents: trained with the input
                    df, ms = run_method(env, lambda a=actor, o=norm, c=cc: MAPPOPolicy(a, o, name=g, camera_count=c),
                                        scenarios, budget, scale)
                    runs.append(df)
                    margins.setdefault(g, []).extend(ms)
                res[g] = sum(runs) / len(runs)

            print(f"\n=== budget {budget:g} | tuned margin {int(100 * tuned)}% | {time.time() - t0:.0f} s", flush=True)
            for name, df in res.items():
                m, lo, hi = bootstrap_ci(df.score, rng_boot)
                summary.append({"cameras": n, "budget": budget, "method": name, "score": m, "ci_low": lo, "ci_high": hi,
                                "err": df.err.mean(), "late_%": df["deadline_miss_%"].mean(),
                                "gpu_ms_per_cam_s": df.gpu_ms_per_cam_s.mean()})
                print(f"  {name:20s} score {m:6.3f} [{lo:6.3f}, {hi:6.3f}] | error {df.err.mean():.4f} | "
                      f"late {df['deadline_miss_%'].mean():5.1f}% | GPU {df.gpu_ms_per_cam_s.mean():5.2f}", flush=True)
            for g, ms in margins.items():      # the margin is the average of the votes; shown in 10% bins
                vals, counts = np.unique(np.round(np.asarray(ms) * 10) / 10, return_counts=True)
                print(f"  {g} margins -> " + ", ".join(f"{int(100 * v)}%: {100 * c / len(ms):.0f}%" for v, c in zip(vals, counts)))
            for g in [g for g in groups if g in res]:
                print(f"  --- paired: {g} minus baseline (negative = {g} better), 95% CI ---")
                for base in ["fixed_best", "count_rule", "periodic_reprofile", "budget_sharing", "tuned_margin",
                             "state_aware_best", "rl_margin"] + [h for h in groups if h != g]:
                    if base not in res:
                        continue
                    d = res[g].score.to_numpy() - res[base].score.to_numpy()
                    m, lo, hi = bootstrap_ci(d, rng_boot)
                    rel = 100 * m / res[base].score.mean()
                    verdict = f"{g} better" if hi < 0 else (f"{g} worse" if lo > 0 else "no significant difference")
                    paired.append({"cameras": n, "budget": budget, "agent": g, "baseline": base, "diff": m,
                                   "ci_low": lo, "ci_high": hi, "relative_%": rel, "verdict": verdict})
                    print(f"  {g} vs {base:18s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({rel:+5.1f}%) -> {verdict}", flush=True)
        pd.DataFrame(summary).round(4).to_csv(OUT / f"eval_{args.split}_cam{n}.csv", index=False)
        pd.DataFrame(paired).round(4).to_csv(OUT / f"paired_{args.split}_cam{n}.csv", index=False)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
