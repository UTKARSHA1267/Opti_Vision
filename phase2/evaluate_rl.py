"""Phase 2, Part E (3): evaluate the RL agents.

Same 20 scenarios per budget as every earlier evaluation. Methods:
  traditional:  fixed_best (deployment practice), count_rule, periodic_reprofile (Chameleon-style)
  strong:       budget_sharing (optimisation baseline), fixed safety margins, state-aware margin rules,
                tuned_margin (the best fixed margin per budget, chosen on TRAINING cameras, like an
                engineer would tune it before deployment)
  RL:           rl_<agent> for every trained agent found (steer / margin controller)
  ceilings:     past_oracle, smoothed_oracle (cheating, for reference)

score = error / mean prior error + RL_DEADLINE_PENALTY x share of late results   (lower is better)

The validation split was used to pick the best checkpoint during training, so FINAL numbers for the
report must come from:   python -u evaluate_rl.py --dynamic --split test

Run:  python -u evaluate_rl.py                  (static environment)
      python -u evaluate_rl.py --dynamic        (dynamic environment)
"""
import _paths  # noqa: F401
import argparse
import pickle

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from _paths import RESULTS2
from config import BUDGETS_MS_PER_CAM_S, EVAL_SCENARIOS, SEED
from dynamic_env import BudgetSharingMargin, DynamicEnv, StateAwareMargin
from evaluate_bounds import SmoothedOracle
from evaluate_probing import run_episode
from p2_config import MARGIN_CHOICES, RL_DEADLINE_PENALTY, SAFETY_MARGINS, STATE_AWARE_MARGINS, TUNE_SCENARIOS
from policies import CountRule, FixedBest, PeriodicReprofile
from predictability_check import PastOracle
from rl_env import BudgetSharing, RLPolicy
from traffic_env import MultiCameraEnv, SegmentTable

OUT = RESULTS2 / "rl"
AGENT_DIRS = {"static": "rl", "static_margin": "rl_margin",
              "dynamic": "rl_dynamic", "dynamic_margin": "rl_dynamic_margin"}


def load_agent(name, which):
    d = RESULTS2 / AGENT_DIRS[name]
    model = PPO.load(d / f"ppo_{which}.zip", device="cpu")
    with open(d / ("vecnorm_best.pkl" if which == "best" else "vecnorm.pkl"), "rb") as f:
        vecnorm = pickle.load(f)
    vecnorm.training = False
    return model, vecnorm


def score_of(r, scale):
    return r["err"] / scale + RL_DEADLINE_PENALTY * r["deadline_miss_%"] / 100


def tune_margin(table, env_cls, budget, scale):
    """Best fixed margin for this budget on TRAINING scenarios (never sees the evaluation cameras)."""
    env = env_cls(table, split="train")
    rng = np.random.default_rng(SEED + 7)
    scenarios = [env.sample_scenario(rng) for _ in range(TUNE_SCENARIOS)]
    best = None
    for m in MARGIN_CHOICES:
        res = [run_episode(env, BudgetSharingMargin(m), sc, budget, seed=SEED + 50 + k) for k, sc in enumerate(scenarios)]
        s = float(np.mean([score_of(r, scale) for r in res]))
        if best is None or s < best[0]:
            best = (s, m)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model", default="best", choices=["best", "final"])
    ap.add_argument("--dynamic", action="store_true", help="evaluate in the dynamic environment")
    ap.add_argument("--agents", nargs="+", default=None, choices=list(AGENT_DIRS),
                    help="trained agents to include (default: all found for this environment)")
    args = ap.parse_args()
    env_cls = DynamicEnv if args.dynamic else MultiCameraEnv
    if args.agents is None:
        wanted = ["dynamic", "dynamic_margin"] if args.dynamic else ["static", "static_margin"]
        args.agents = [a for a in wanted if (RESULTS2 / AGENT_DIRS[a] / f"ppo_{args.model}.zip").exists()]
    tag = ("dynamic" if args.dynamic else "static")
    print(f"Environment: {env_cls.__name__} | agents: {', '.join(args.agents) or 'none found'}", flush=True)

    table = SegmentTable()
    scale = float(np.mean(table.err_prior))
    agents = {a: load_agent(a, args.model) for a in args.agents}

    env = env_cls(table, split=args.split)
    rng = np.random.default_rng(SEED)
    scenarios = [env.sample_scenario(rng) for _ in range(EVAL_SCENARIOS)]

    rows = []
    for budget in BUDGETS_MS_PER_CAM_S:
        tuned = tune_margin(table, env_cls, budget, scale)
        methods = [("fixed_best", FixedBest), ("count_rule", CountRule), ("periodic_reprofile", PeriodicReprofile),
                   ("budget_sharing", BudgetSharing)]
        methods += [(f"margin{int(100 * m)}", (lambda m=m: BudgetSharingMargin(m))) for m in SAFETY_MARGINS]
        methods += [(f"tuned_margin({int(100 * tuned)})", (lambda m=tuned: BudgetSharingMargin(m)))]
        if args.dynamic:
            methods += [(f"state_aware{int(100 * m)}", (lambda m=m: StateAwareMargin(m))) for m in STATE_AWARE_MARGINS]
        methods += [(f"rl_{a}", (lambda a=a: RLPolicy(*agents[a], name=f"rl_{a}"))) for a in args.agents]
        methods += [("past_oracle", PastOracle), ("smoothed_oracle", SmoothedOracle)]

        out, chosen = {}, {}
        for name, make in methods:
            res, margins = [], []
            for k, sc in enumerate(scenarios):
                pol = make()
                res.append(run_episode(env, pol, sc, budget, seed=SEED + k))
                margins += getattr(pol, "margins", [])
            out[name] = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
            out[name]["err_std"] = float(np.nanstd([r["err"] for r in res]))
            out[name]["score"] = score_of(out[name], scale)
            out[name]["score_std"] = float(np.std([score_of(r, scale) for r in res]))
            if margins:
                chosen[name] = margins
        fixed, share = out["fixed_best"]["err"], out["budget_sharing"]["err"]
        best = min(out[m]["score"] for m in out if m not in ("past_oracle", "smoothed_oracle"))
        print(f"\nbudget {budget:5.1f}  ({args.split})   score = error/{scale:.3f} + {RL_DEADLINE_PENALTY:g} x late share "
              f"(lower is better); tuned margin chosen on training cameras: {int(100 * tuned)}%")
        for name, r in out.items():
            mark = "  <- best non-cheating method" if r["score"] == best else ""
            print(f"  {name:22s} score {r['score']:.3f} +- {r['score_std']:.3f} | error {r['err']:.4f} "
                  f"(vs fixed {100 * (r['err'] - fixed) / fixed:+6.1f}%, vs sharing {100 * (r['err'] - share) / share:+6.1f}%) "
                  f"| late {r['deadline_miss_%']:4.1f}% | GPU {r['gpu_ms_per_cam_s']:5.2f} | "
                  f"switches {r['switches_per_cam_min']:4.1f}/cam/min{mark}", flush=True)
            rows.append({"split": args.split, "env": tag, "budget_ms_per_cam_s": budget, "method": name, **r})
        for name, ms in chosen.items():
            vals, counts = np.unique(np.round(ms, 2), return_counts=True)
            dist = ", ".join(f"{int(100 * v)}%: {100 * c / len(ms):.0f}%" for v, c in zip(vals, counts))
            print(f"  {name} chose margins -> {dist}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rl_{args.split}_{tag}.csv"
    pd.DataFrame(rows).round(4).to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
