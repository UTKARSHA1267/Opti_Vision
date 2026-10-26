"""Run the trained MAPPO agent without retraining (CPU, no GPU or dataset needed, under a minute).

Loads the main checkpoint (outputs/results_<target>/mappo_cam4-16_nocount/best.pt) and the precomputed simulator
table (phase1/outputs/env/segment_table.npz), then plays the same 5-minute test scenarios with MAPPO, the fixed
setting used in practice and budget sharing, in the shared-GPU environment, and prints error, late results and score.

Run (from phase3/):
  EVAL_TARGET=reference python demo_agent.py                       (6 cameras, medium budget, 5 scenarios)
  EVAL_TARGET=reference python demo_agent.py --cameras 24 --budget 3
"""
import _paths  # noqa: F401
import argparse

import numpy as np

from _paths import RESULTS3
from config import SEED
from dynamic_env import DynamicEnv
from evaluate_probing import run_episode
from ma_env import MAPPOPolicy
from mappo import load
from p2_config import RL_DEADLINE_PENALTY
from policies import FixedBest
from rl_env import BudgetSharing
from traffic_env import SegmentTable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=6)
    ap.add_argument("--budget", type=float, default=7.0, help="GPU ms per camera-second (3 tight, 7 medium, 15 generous)")
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--checkpoint", default=str(RESULTS3 / "mappo_cam4-16_nocount" / "best.pt"))
    args = ap.parse_args()

    table = SegmentTable()
    actor, norm, ckpt = load(args.checkpoint)
    cc = ckpt.get("info", {}).get("camera_count", True)
    env = DynamicEnv(table, split="test")
    env.n = args.cameras
    rng = np.random.default_rng(SEED + 1000 + args.cameras)
    scenarios = [env.sample_scenario(rng) for _ in range(args.scenarios)]
    scale = float(np.mean(table.err_prior))

    print(f"Trained agent: {args.checkpoint}\n{args.cameras} test cameras, budget {args.budget:g} ms/cam/s, "
          f"{args.scenarios} scenarios of 5 minutes, shared GPU with capacity drops\n")
    print(f"{'method':16s} {'error':>7s} {'late %':>7s} {'score':>7s}   (score = error/{scale:.3f} + late share; lower is better)")
    methods = [("fixed_best", FixedBest), ("budget_sharing", BudgetSharing),
               ("MAPPO (trained)", lambda: MAPPOPolicy(actor, norm, camera_count=cc))]
    for name, make in methods:
        res = [run_episode(env, make(), sc, args.budget, seed=SEED + k) for k, sc in enumerate(scenarios)]
        err = np.mean([r["err"] for r in res])
        late = np.mean([r["deadline_miss_%"] for r in res])
        print(f"{name:16s} {err:7.4f} {late:7.1f} {err / scale + RL_DEADLINE_PENALTY * late / 100:7.3f}")


if __name__ == "__main__":
    main()
