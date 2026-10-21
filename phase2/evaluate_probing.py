"""Phase 2, Part B (6): what is probing worth in practice?

Same validation scenarios as before. Methods:
  fixed_best              one setting everywhere
  probe_allocate_r0       estimator v3 with NO probes (falls back to the typical error per setting)
  probe_allocate_rX       on average X probes per segment (0.25 = one every 4 segments), oldest first
  past_oracle             TRUE errors of the last 10 s (cheating; realistic ceiling for history-based methods)
  smoothed_oracle         TRUE errors averaged over +-10 s (cheating; peeks into the future)

Output: phase2/outputs/results_<target>/probing_val.csv

Run:  python -u evaluate_probing.py
"""
import _paths  # noqa: F401

import numpy as np
import pandas as pd

from _paths import RESULTS2
from config import BUDGETS_MS_PER_CAM_S, EVAL_SCENARIOS, SEED, SEGMENT_SECONDS
from evaluate_bounds import SmoothedOracle
from p2_config import PROBE_RATES
from policies import FixedBest
from predictability_check import PastOracle
from probing import LinearProbeEstimator, ProbeAllocate, ProbeData, ProbeEnv
from traffic_env import SegmentTable


def run_episode(env, policy, scenario, budget, seed):
    env.rng = np.random.default_rng(seed)
    obs = env.reset(scenario=scenario, budget_ms_per_cam_s=budget)
    policy.reset(env)
    errs, work, probe_ms, misses, probes, switches = [], 0.0, 0.0, 0, 0, 0
    done = False
    while not done:
        a = policy.act(obs, env)
        proposal = a.pop("proposal")
        obs, _, done, info = env.step(proposal, **a)
        policy.update(info, env)
        errs.extend(info["err"][info["valid"]])
        work += info["work_ms"]
        probe_ms += info.get("probe_ms", 0.0)
        probes += int(np.sum(info.get("probe", 0)))
        misses += info["deadline_miss"]
        switches += info.get("switches", 0)
    steps = env.horizon
    return {"err": float(np.mean(errs)) if errs else np.nan,
            "gpu_ms_per_cam_s": work / (env.n * steps * SEGMENT_SECONDS),
            "probe_share_%": 100 * probe_ms / max(work, 1e-9),
            "probes_per_segment": probes / steps,
            "deadline_miss_%": 100 * misses / steps,
            "switches_per_cam_min": switches / env.n / (steps * SEGMENT_SECONDS / 60)}


def main():
    tab = SegmentTable()
    data = ProbeData(tab)
    est = LinearProbeEstimator()
    env = ProbeEnv(tab, data, split="val")
    rng = np.random.default_rng(SEED)
    scenarios = [env.sample_scenario(rng) for _ in range(EVAL_SCENARIOS)]

    methods = [("fixed_best", FixedBest)]
    methods += [(f"probe_allocate_r{r:g}", (lambda r=r: ProbeAllocate(est, r))) for r in PROBE_RATES]
    methods += [("past_oracle", PastOracle), ("smoothed_oracle", SmoothedOracle)]

    rows = []
    for budget in BUDGETS_MS_PER_CAM_S:
        out = {}
        for name, make in methods:
            res = [run_episode(env, make(), sc, budget, seed=SEED + k) for k, sc in enumerate(scenarios)]
            out[name] = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
        fixed, past = out["fixed_best"]["err"], out["past_oracle"]["err"]
        print(f"budget {budget:5.1f}")
        for name, r in out.items():
            share = 100 * (fixed - r["err"]) / (fixed - past) if fixed > past else np.nan
            print(f"  {name:22s} error {r['err']:.4f} ({100 * (r['err'] - fixed) / fixed:+6.1f}% vs fixed) | "
                  f"share of past-oracle gain {share:6.1f}% | GPU {r['gpu_ms_per_cam_s']:6.2f} ms/cam/s | "
                  f"probing {r['probe_share_%']:5.1f}% of GPU | deadline miss {r['deadline_miss_%']:4.1f}%",
                  flush=True)
            rows.append({"budget_ms_per_cam_s": budget, "method": name, **r})
    RESULTS2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(RESULTS2 / "probing_val.csv", index=False)
    print(f"\nSaved {RESULTS2 / 'probing_val.csv'}")


if __name__ == "__main__":
    main()
