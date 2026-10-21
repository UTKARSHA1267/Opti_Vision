"""Phase 2, Part E (2): train the RL agent (PPO, Stable-Baselines3).

Trains on TRAIN cameras, with a random budget between RL_BUDGET_RANGE for every episode, so one agent
works at every budget level. Observations and rewards are normalised (VecNormalize); the statistics
are saved next to the model and reused for evaluation.

Every eval_every steps the agent is scored on a few VALIDATION scenarios and the best version is kept.

Outputs (phase2/outputs/results_<target>/rl/ or rl_dynamic/): ppo_final.zip, ppo_best.zip, vecnorm.pkl, training_log.csv

Run:  python -u train_ppo.py                    (RL_TIMESTEPS steps, see p2_config.py)
      python -u train_ppo.py --timesteps 20000  (quick test)
      python -u train_ppo.py --dynamic          (dynamic environment)
      python -u train_ppo.py --dynamic --mode margin   (RL margin controller)
"""
import _paths  # noqa: F401
import argparse
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from _paths import RESULTS2
from config import SEED
from p2_config import RL_DEADLINE_PENALTY, RL_N_ENVS, RL_TIMESTEPS
from dynamic_env import DynamicEnv
from rl_env import BudgetSharing, RLPolicy, VideoAllocEnv
from traffic_env import MultiCameraEnv, SegmentTable

OUT = RESULTS2 / "rl"
ENV_CLS = MultiCameraEnv
EVAL_BUDGETS = (3.0, 7.0, 15.0)
EVAL_SCENARIOS = 6


def quick_eval(policy, table, scenarios):
    """Score over a few validation scenarios at each eval budget, using the SAME trade-off as the reward:
    score = error / mean prior error + RL_DEADLINE_PENALTY x share of late results (lower is better)."""
    from evaluate_probing import run_episode
    env = ENV_CLS(table, split="val")
    scale = float(np.mean(table.err_prior))
    errs, misses = [], []
    for b in EVAL_BUDGETS:
        for k, sc in enumerate(scenarios):
            r = run_episode(env, policy, sc, b, seed=SEED + k)
            errs.append(r["err"])
            misses.append(r["deadline_miss_%"] / 100)
    err, miss = float(np.mean(errs)), float(np.mean(misses))
    return err / scale + RL_DEADLINE_PENALTY * miss, err, miss


class EvalCallback(BaseCallback):
    def __init__(self, table, scenarios, vecnorm, every, baseline):
        super().__init__()
        self.table, self.scenarios, self.vecnorm = table, scenarios, vecnorm
        self.every, self.baseline = every, baseline
        self.best, self.log, self.t0 = np.inf, [], time.time()

    def _on_step(self):
        if self.n_calls % self.every:
            return True
        self.vecnorm.training = False
        score, err, miss = quick_eval(RLPolicy(self.model, self.vecnorm), self.table, self.scenarios)
        self.vecnorm.training = True
        self.log.append({"timesteps": self.num_timesteps, "val_score": score, "val_err": err, "val_late": miss,
                         "budget_sharing_score": self.baseline[0], "minutes": (time.time() - self.t0) / 60})
        flag = ""
        if score < self.best:
            self.best = score
            self.model.save(OUT / "ppo_best.zip")
            self.vecnorm.save(OUT / "vecnorm_best.pkl")
            flag = "  <- best so far, saved"
        b = self.baseline
        print(f"[{self.num_timesteps:>9,} steps, {(time.time() - self.t0) / 60:5.1f} min] validation score {score:.4f} "
              f"(error {err:.4f}, late {100 * miss:4.1f}%) | budget sharing {b[0]:.4f} "
              f"(error {b[1]:.4f}, late {100 * b[2]:4.1f}%){flag}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=RL_TIMESTEPS)
    ap.add_argument("--eval-every", type=int, default=None, help="steps per environment between evaluations")
    ap.add_argument("--dynamic", action="store_true", help="train in the dynamic environment (shared GPU, switching costs)")
    ap.add_argument("--mode", default="steer", choices=["steer", "margin"],
                    help="steer: per-camera importance + reserve; margin: pick one safety margin per segment")
    args = ap.parse_args()
    global OUT, ENV_CLS
    if args.dynamic:
        OUT, ENV_CLS = RESULTS2 / "rl_dynamic", DynamicEnv
    if args.mode == "margin":
        OUT = OUT.parent / (OUT.name + "_margin")
    print(f"Environment: {ENV_CLS.__name__}, action mode: {args.mode}, saving to {OUT.name}/", flush=True)
    torch.set_num_threads(4)

    table = SegmentTable()
    OUT.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([lambda i=i: VideoAllocEnv(table, split="train", seed=SEED + 100 * i, core_cls=ENV_CLS,
                                                  mode=args.mode)
                        for i in range(RL_N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.95)

    rng = np.random.default_rng(SEED)
    scen_env = ENV_CLS(table, split="val")
    scenarios = [scen_env.sample_scenario(rng) for _ in range(EVAL_SCENARIOS)]
    baseline = quick_eval(BudgetSharing(), table, scenarios)
    print(f"Budget sharing on the quick validation set: score {baseline[0]:.4f} "
          f"(error {baseline[1]:.4f}, late {100 * baseline[2]:.1f}%)", flush=True)

    if args.mode == "margin":       # discrete choice: a little entropy bonus keeps it exploring the margins
        kwargs = dict(ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]))
    else:                           # continuous steering: start close to the all-zero action (= budget sharing)
        kwargs = dict(ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0))
    model = PPO("MlpPolicy", venv, n_steps=300, batch_size=600, n_epochs=10, learning_rate=3e-4,
                gamma=0.95, gae_lambda=0.95, clip_range=0.2, seed=SEED, verbose=0, device="cpu", **kwargs)
    every = args.eval_every or max(1, args.timesteps // (10 * RL_N_ENVS))
    cb = EvalCallback(table, scenarios, venv, every, baseline)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    model.save(OUT / "ppo_final.zip")
    venv.save(OUT / "vecnorm.pkl")
    pd.DataFrame(cb.log).to_csv(OUT / "training_log.csv", index=False)
    print(f"Best quick-validation score {cb.best:.4f} vs budget sharing {baseline[0]:.4f} (lower is better)")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
