"""Phase 3, Part A (3): train MAPPO (or independent PPO) in the dynamic environment.

MAPPO = PPO for many agents with:
  - parameter sharing: all camera agents use ONE actor network
  - centralised training: the critic sees the central state (all cameras summarised)
  - decentralised execution: at run time each agent decides from its own observation
  - team reward: every agent gets the same reward (error + lateness of the whole system)
--critic local trains INDEPENDENT PPO instead (each agent's critic sees only its own observation),
the ablation that shows whether centralised training helps.

Every update, each parallel environment plays one full episode (150 segments) on TRAINING cameras, with a
random budget (and a random number of cameras within MAPPO_CAMERA_RANGE). About MAPPO_EVAL_TIMES times
during training the agent is scored on VALIDATION scenarios (same score as Phase 2); the best is kept.

Outputs (phase3/outputs/results_<target>/<run>/): best.pt, final.pt, training_log.csv
  <run> = mappo | ippo, plus _cam<a>-<b> if the camera range is not 6-6, plus _s<seed> for other seeds

Run:  python -u train_mappo.py                          (MAPPO, MAPPO_TIMESTEPS steps)
      python -u train_mappo.py --critic local           (independent PPO ablation)
      python -u train_mappo.py --timesteps 30000        (quick test)
      python -u train_mappo.py --cameras 4 16 --no-camera-count   (scaling version)
"""
import _paths  # noqa: F401
import argparse
import time

import numpy as np
import pandas as pd
import torch

from _paths import RESULTS3
from config import SEED
from dynamic_env import DynamicEnv
from evaluate_probing import run_episode
from ma_env import MAPPOPolicy, MultiAgentEnv, OBS_DIM, STATE_DIM
from mappo import Actor, Critic, RunningNorm, save
from p2_config import RL_DEADLINE_PENALTY
from p3_config import (MAPPO_CAMERA_RANGE, MAPPO_CLIP, MAPPO_ENTROPY, MAPPO_EPOCHS, MAPPO_EVAL_TIMES, MAPPO_GAMMA,
                       MAPPO_HIDDEN, MAPPO_LAMBDA, MAPPO_LR, MAPPO_MINIBATCHES, MAPPO_N_ENVS, MAPPO_REWARD_SCALE,
                       MAPPO_TIMESTEPS)
from traffic_env import SegmentTable

EVAL_BUDGETS = (3.0, 7.0, 15.0)
EVAL_SCENARIOS = 6


def quick_eval(actor, obs_norm, table, scenarios, camera_count=True):
    env = DynamicEnv(table, split="val")
    scale = float(np.mean(table.err_prior))
    errs, lates = [], []
    for b in EVAL_BUDGETS:
        for k, sc in enumerate(scenarios):
            r = run_episode(env, MAPPOPolicy(actor, obs_norm, camera_count=camera_count), sc, b, seed=SEED + k)
            errs.append(r["err"])
            lates.append(r["deadline_miss_%"] / 100)
    err, late = float(np.mean(errs)), float(np.mean(lates))
    return err / scale + RL_DEADLINE_PENALTY * late, err, late


def collect(envs, actor, critic, obs_norm, state_norm, gen):
    """Every environment plays one full episode. Returns per-environment lists of transitions."""
    starts = [e.reset() for e in envs]
    obs = [s[0] for s in starts]
    states = [s[1] for s in starts]
    traj = [{"obs": [], "state": [], "act": [], "logp": [], "rew": [], "val": [], "margin": []} for _ in envs]
    done = False
    while not done:
        sizes = [len(o) for o in obs]
        x = torch.as_tensor(obs_norm.normalize(np.concatenate(obs)), dtype=torch.float32)
        with torch.no_grad():
            di, dm = actor.dist(x)
            ai, am = di.sample(), dm.sample()
            logp = (di.log_prob(ai) + dm.log_prob(am)).numpy()
            if critic.kind == "central":
                s = torch.as_tensor(state_norm.normalize(np.stack(states)), dtype=torch.float32)
                vals = critic(s).numpy()
            else:
                vals = critic(x).numpy()
        acts = np.stack([ai.numpy(), am.numpy()], axis=1)
        offs = np.cumsum([0] + sizes)
        new_obs, new_states = [], []
        for k, e in enumerate(envs):
            sl = slice(offs[k], offs[k + 1])
            tr = traj[k]
            tr["obs"].append(x[sl].numpy())
            tr["state"].append(state_norm.normalize(states[k]))
            tr["act"].append(acts[sl])
            tr["logp"].append(logp[sl])
            tr["val"].append(vals[k] if critic.kind == "central" else vals[sl])
            o, st, r, d, info = e.step(acts[sl])
            tr["rew"].append(r * MAPPO_REWARD_SCALE)
            tr["margin"].append(info["margin"])
            new_obs.append(o)
            new_states.append(st)
            done = d
        obs_norm.update(np.concatenate(obs))
        state_norm.update(np.stack(states))
        obs, states = new_obs, new_states
    return traj


def gae(rew, val):
    """Generalised advantage estimation for one episode (terminal at the end). val: (T,) or (T, N)."""
    val = np.asarray(val, dtype=np.float64)
    T = len(rew)
    adv = np.zeros_like(val)
    last = np.zeros_like(val[0])
    for t in reversed(range(T)):
        nxt = val[t + 1] if t + 1 < T else np.zeros_like(val[0])
        delta = rew[t] + MAPPO_GAMMA * nxt - val[t]
        last = delta + MAPPO_GAMMA * MAPPO_LAMBDA * last
        adv[t] = last
    return adv, adv + val


def update(actor, critic, opt, traj):
    A_obs, A_act, A_logp, A_adv, C_x, C_ret = [], [], [], [], [], []
    for tr in traj:
        adv, ret = gae(tr["rew"], tr["val"])
        for t in range(len(tr["rew"])):
            n = len(tr["obs"][t])
            A_obs.append(tr["obs"][t])
            A_act.append(tr["act"][t])
            A_logp.append(tr["logp"][t])
            if critic.kind == "central":
                A_adv.append(np.full(n, adv[t]))
                C_x.append(tr["state"][t][None])
                C_ret.append(np.array([ret[t]]))
            else:
                A_adv.append(adv[t])
                C_x.append(tr["obs"][t])
                C_ret.append(ret[t])
    A_obs = torch.as_tensor(np.concatenate(A_obs), dtype=torch.float32)
    A_act = torch.as_tensor(np.concatenate(A_act), dtype=torch.long)
    A_logp = torch.as_tensor(np.concatenate(A_logp), dtype=torch.float32)
    A_adv = np.concatenate(A_adv)
    A_adv = torch.as_tensor((A_adv - A_adv.mean()) / (A_adv.std() + 1e-8), dtype=torch.float32)
    C_x = torch.as_tensor(np.concatenate(C_x), dtype=torch.float32)
    C_ret = torch.as_tensor(np.concatenate(C_ret), dtype=torch.float32)

    na, nc = len(A_obs), len(C_x)
    stats = []
    for _ in range(MAPPO_EPOCHS):
        pa, pc = torch.randperm(na), torch.randperm(nc)
        for i in range(MAPPO_MINIBATCHES):
            ia = pa[i * na // MAPPO_MINIBATCHES:(i + 1) * na // MAPPO_MINIBATCHES]
            ic = pc[i * nc // MAPPO_MINIBATCHES:(i + 1) * nc // MAPPO_MINIBATCHES]
            di, dm = actor.dist(A_obs[ia])
            logp = di.log_prob(A_act[ia, 0]) + dm.log_prob(A_act[ia, 1])
            ratio = torch.exp(logp - A_logp[ia])
            adv = A_adv[ia]
            pol_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - MAPPO_CLIP, 1 + MAPPO_CLIP) * adv).mean()
            entropy = (di.entropy() + dm.entropy()).mean()
            val_loss = ((critic(C_x[ic]) - C_ret[ic]) ** 2).mean()
            loss = pol_loss - MAPPO_ENTROPY * entropy + 0.5 * val_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            opt.step()
            stats.append((pol_loss.item(), val_loss.item(), entropy.item()))
    return np.mean(stats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=MAPPO_TIMESTEPS)
    ap.add_argument("--critic", default="central", choices=["central", "local"])
    ap.add_argument("--cameras", type=int, nargs=2, default=list(MAPPO_CAMERA_RANGE), metavar=("MIN", "MAX"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-camera-count", action="store_true",
                    help="do not give agents the number of cameras as an input (better generalisation to large systems)")
    args = ap.parse_args()
    camera_count = not args.no_camera_count
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    run = "mappo" if args.critic == "central" else "ippo"
    if tuple(args.cameras) != (6, 6):
        run += f"_cam{args.cameras[0]}-{args.cameras[1]}"
    if not camera_count:
        run += "_nocount"
    if args.seed != SEED:
        run += f"_s{args.seed}"
    out = RESULTS3 / run
    out.mkdir(parents=True, exist_ok=True)
    print(f"Training {run}: critic {args.critic}, cameras {args.cameras[0]}-{args.cameras[1]}, seed {args.seed}, "
          f"{args.timesteps:,} steps -> {out}", flush=True)

    table = SegmentTable()
    envs = [MultiAgentEnv(table, split="train", seed=args.seed + 100 * i, camera_range=tuple(args.cameras),
                          camera_count=camera_count) for i in range(MAPPO_N_ENVS)]
    actor, critic = Actor(MAPPO_HIDDEN), Critic(args.critic, MAPPO_HIDDEN)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=MAPPO_LR, eps=1e-5)
    obs_norm, state_norm = RunningNorm(OBS_DIM), RunningNorm(STATE_DIM)

    rng = np.random.default_rng(SEED)
    scen_env = DynamicEnv(table, split="val")
    scenarios = [scen_env.sample_scenario(rng) for _ in range(EVAL_SCENARIOS)]
    steps_per_update = MAPPO_N_ENVS * envs[0].core.horizon
    n_updates = max(1, args.timesteps // steps_per_update)
    eval_every = max(1, n_updates // MAPPO_EVAL_TIMES)

    best, log, t0 = np.inf, [], time.time()
    for u in range(1, n_updates + 1):
        traj = collect(envs, actor, critic, obs_norm, state_norm, None)
        pol, val, ent = update(actor, critic, opt, traj)
        if u % eval_every == 0 or u == n_updates:
            actor.eval()
            score, err, late = quick_eval(actor, obs_norm, table, scenarios, camera_count)
            actor.train()
            margins = np.concatenate([tr["margin"] for tr in traj])
            flag = ""
            if score < best:
                best = score
                save(out / "best.pt", actor, critic, obs_norm, state_norm,
                     {"steps": u * steps_per_update, "camera_count": camera_count})
                flag = "  <- best so far, saved"
            steps = u * steps_per_update
            log.append({"steps": steps, "val_score": score, "val_err": err, "val_late": late,
                        "train_return": float(np.mean([sum(tr["rew"]) for tr in traj]) / MAPPO_REWARD_SCALE),
                        "entropy": ent, "value_loss": val, "mean_margin": float(margins.mean()),
                        "minutes": (time.time() - t0) / 60})
            print(f"[{steps:>9,} steps, {(time.time() - t0) / 60:5.1f} min] validation score {score:.4f} "
                  f"(error {err:.4f}, late {100 * late:4.1f}%) | entropy {ent:.2f} | mean margin "
                  f"{100 * margins.mean():4.1f}%{flag}", flush=True)
    save(out / "final.pt", actor, critic, obs_norm, state_norm,
         {"steps": n_updates * steps_per_update, "camera_count": camera_count})
    pd.DataFrame(log).to_csv(out / "training_log.csv", index=False)
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min. Best validation score {best:.4f}. Saved {out}")


if __name__ == "__main__":
    main()
