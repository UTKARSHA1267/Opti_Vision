"""Step 1.9: Evaluate the baselines and the hindsight oracle.

Every method runs on the SAME scenarios (same cameras, start points and random GPU-time
variation) at each budget level, so the comparison is fair.

Use the VALIDATION split while developing. Keep the TEST split for the final report only.

Outputs (in outputs/env/):
  baselines_<split>.csv     one row per (budget, method)
  baselines_<split>.png     error vs compute, one panel per budget

Run:  python evaluate_baselines.py                 (validation split)
      python evaluate_baselines.py --split test    (final report only)
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import BUDGETS_MS_PER_CAM_S, ENV_DIR, EVAL_SCENARIOS, SEED, SEGMENT_SECONDS  # noqa: E402
from policies import ALL_BASELINES  # noqa: E402
from traffic_env import MultiCameraEnv, SegmentTable  # noqa: E402


def run_episode(env, policy, scenario, budget, seed):
    env.rng = np.random.default_rng(seed)          # identical GPU-time variation for every method
    obs = env.reset(scenario=scenario, budget_ms_per_cam_s=budget)
    policy.reset(env)
    cam_err = [[] for _ in range(env.n)]
    work = misses = overrides = infeasible = 0.0
    reward = 0.0
    done = False
    while not done:
        a = policy.act(obs, env)
        obs, r, done, info = env.step(a["proposal"], enforce=a.get("enforce", True),
                                      extra_cost=a.get("extra_cost"), floors=a.get("floors"))
        policy.update(info, env)
        for i in np.flatnonzero(info["valid"]):
            cam_err[i].append(info["err"][i])
        work += info["work_ms"]
        misses += info["deadline_miss"]
        overrides += info["overridden"].mean()
        infeasible += not info["feasible"]
        reward += r
    per_cam = np.array([np.mean(e) if e else np.nan for e in cam_err])
    steps = env.horizon
    return {"err": np.nanmean(np.concatenate([np.array(e) for e in cam_err if e])) if any(cam_err) else np.nan,
            "worst_cam_err": np.nanmax(per_cam),
            "gpu_ms_per_cam_s": work / (env.n * steps * SEGMENT_SECONDS),
            "budget_used_%": 100 * work / (env.budget * steps),
            "deadline_miss_%": 100 * misses / steps,
            "override_%": 100 * overrides / steps,
            "infeasible_%": 100 * infeasible / steps,
            "final_backlog_ms": env.backlog,
            "reward": reward / steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--scenarios", type=int, default=EVAL_SCENARIOS)
    args = ap.parse_args()

    table = SegmentTable()
    env = MultiCameraEnv(table, split=args.split)
    rng = np.random.default_rng(SEED)
    scenarios = [env.sample_scenario(rng) for _ in range(args.scenarios)]
    print(f"Split {args.split}: {args.scenarios} scenarios x {env.n} cameras x {env.horizon} segments "
          f"({env.horizon * SEGMENT_SECONDS / 60:.0f} min each)\n")

    rows = []
    for budget in BUDGETS_MS_PER_CAM_S:
        for P in ALL_BASELINES:
            res = [run_episode(env, P(), sc, budget, seed=SEED + k) for k, sc in enumerate(scenarios)]
            row = {"budget_ms_per_cam_s": budget, "method": P.name}
            for key in res[0]:
                vals = np.array([r[key] for r in res], dtype=float)
                row[key] = round(float(np.nanmean(vals)), 4)
                if key in ("err", "gpu_ms_per_cam_s"):
                    row[f"{key}_std"] = round(float(np.nanstd(vals)), 4)
            rows.append(row)
            print(f"budget {budget:5.1f} | {P.name:18s} | error {row['err']:.4f} | worst cam {row['worst_cam_err']:.4f} | "
                  f"GPU {row['gpu_ms_per_cam_s']:6.2f} ms/cam/s | deadline miss {row['deadline_miss_%']:5.1f}% | "
                  f"overrides {row['override_%']:5.1f}%", flush=True)
        print()

    df = pd.DataFrame(rows)
    for b in BUDGETS_MS_PER_CAM_S:
        m = df.budget_ms_per_cam_s == b
        fixed = df[m & (df.method == "fixed_best")].iloc[0]
        df.loc[m, "err_vs_fixed_%"] = (100 * (df[m].err - fixed.err) / fixed.err).round(1)
        df.loc[m, "compute_vs_fixed_%"] = (100 * (df[m].gpu_ms_per_cam_s - fixed.gpu_ms_per_cam_s)
                                          / fixed.gpu_ms_per_cam_s).round(1)
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENV_DIR / f"baselines_{args.split}.csv", index=False)

    fig, axes = plt.subplots(1, len(BUDGETS_MS_PER_CAM_S), figsize=(5 * len(BUDGETS_MS_PER_CAM_S), 4), sharey=True)
    for ax, b in zip(np.atleast_1d(axes), BUDGETS_MS_PER_CAM_S):
        d = df[df.budget_ms_per_cam_s == b]
        for r in d.itertuples():
            ax.errorbar(r.gpu_ms_per_cam_s, r.err, yerr=r.err_std, fmt="o", capsize=3, label=r.method)
        ax.axvline(b, color="grey", ls="--", lw=0.8)
        ax.set_title(f"budget {b} ms per camera-second")
        ax.set_xlabel("GPU ms per camera-second used")
    np.atleast_1d(axes)[0].set_ylabel(f"counting error ({args.split})")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ENV_DIR / f"baselines_{args.split}.png", dpi=140)

    pd.set_option("display.width", 200)
    cols = ["budget_ms_per_cam_s", "method", "err", "worst_cam_err", "gpu_ms_per_cam_s", "err_vs_fixed_%",
            "compute_vs_fixed_%", "deadline_miss_%", "override_%", "infeasible_%"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved {ENV_DIR / f'baselines_{args.split}.csv'} and .png")


if __name__ == "__main__":
    main()
