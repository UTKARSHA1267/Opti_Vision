"""Phase 3, Part A (1): the multi-agent environment (one agent per camera).

Every 2-second segment, EACH camera's agent picks:
  importance  low / normal / high   -> how much this camera wants extra compute right now
  margin vote 0 ... 50%              -> how much of the GPU budget it thinks should be kept free
The system keeps the AVERAGE of the votes as the safety margin (so every agent's vote always counts,
which gives each agent a clear learning signal) and allocates the rest of the budget with budget sharing,
weighting each camera's errors by its importance (and operator priority).

Each agent sees (decentralised execution):
  its own camera: last count, its change, its rolling mean, its current setting, its relative priority
  a system summary (the same for all agents): planning budget, share used, backlog, nominal budget,
  measured GPU availability, time since availability changed, number of cameras
The critic sees the central STATE (centralised training): the system summary + the mean and the max
of all cameras' own features. Mean/max make it work for ANY number of cameras.

Team reward (shared by all agents): -(error / mean prior error) - RL_DEADLINE_PENALTY x late
"""
import _paths  # noqa: F401

import numpy as np

from allocation import allocate
from dynamic_env import DynamicEnv
from p2_config import MARGIN_CHOICES, RL_BUDGET_RANGE, RL_COUNT_WINDOW, RL_DEADLINE_PENALTY, RL_WEIGHT_SCALE
from p3_config import IMPORTANCE_LEVELS, MAPPO_CAMERA_RANGE
from policies import Policy
from traffic_env import MultiCameraEnv

COUNT_SCALE, BUDGET_SCALE, LIMIT = 20.0, 15.0, 5.0
LOCAL_DIM = 5
GLOBAL_DIM = 7
OBS_DIM = LOCAL_DIM + GLOBAL_DIM          # what each agent sees
STATE_DIM = GLOBAL_DIM + 2 * LOCAL_DIM    # what the central critic sees
N_IMPORTANCE = len(IMPORTANCE_LEVELS)
N_MARGIN = len(MARGIN_CHOICES)


class AgentObsBuilder:
    """Builds every agent's observation and the central state from the core environment's observation.
    camera_count=False replaces the "number of cameras / 20" input with 0: an input whose value at large
    camera counts (e.g. 48 -> 2.4) lies far outside anything seen in training (4-8 -> 0.2-0.4)."""

    def __init__(self, camera_count=True):
        self.camera_count = camera_count

    def reset(self):
        self.history = []

    def build(self, obs):
        self.history.append(obs["last_count"].copy())
        self.history = self.history[-RL_COUNT_WINDOW:]
        mean = np.mean(self.history, axis=0)
        w = obs["weights"]
        local = np.column_stack([obs["last_count"] / COUNT_SCALE,
                                 np.clip(obs["count_change"] / 10.0, -3, 3),
                                 mean / COUNT_SCALE,
                                 obs["current_setting"],
                                 w / max(float(w.mean()), 1e-9)])
        n = len(local)
        glob = np.array([obs["budget_ms_per_cam_s"] / BUDGET_SCALE,
                         min(obs["budget_used_share"], 3.0),
                         min(obs["backlog_share"], 3.0),
                         obs.get("nominal_budget_ms_per_cam_s", obs["budget_ms_per_cam_s"]) / BUDGET_SCALE,
                         obs.get("capacity_share", 1.0),
                         min(obs.get("since_change", 0), 30) / 30.0,
                         n / 20.0 if self.camera_count else 0.0])
        agent_obs = np.concatenate([local, np.tile(glob, (n, 1))], axis=1)
        state = np.concatenate([glob, local.mean(axis=0), local.max(axis=0)])
        return (np.clip(agent_obs, -LIMIT, LIMIT).astype(np.float32),
                np.clip(state, -LIMIT, LIMIT).astype(np.float32))


def joint_to_proposal(core, importance_idx, margin_idx):
    """All agents' choices -> one setting per camera (within the budget), the error estimate, the margin."""
    margin = float(np.mean(np.asarray(MARGIN_CHOICES)[np.asarray(margin_idx, dtype=int)]))
    w = np.exp(RL_WEIGHT_SCALE * np.asarray(IMPORTANCE_LEVELS)[np.asarray(importance_idx, dtype=int)]) * core.weights
    cap = core.scheduler.limit(core.budget, core.backlog) * (1 - margin)
    err_est = np.tile(core.tab.err_prior, (core.n, 1))
    return allocate(err_est, core.costs, cap, w), err_est * w[:, None], margin


class MultiAgentEnv:
    """N camera agents around the core environment (dynamic by default). N can change per episode."""

    def __init__(self, table, split="train", dynamic=True, seed=0, budget_range=RL_BUDGET_RANGE,
                 camera_range=MAPPO_CAMERA_RANGE, camera_count=True):
        self.core = (DynamicEnv if dynamic else MultiCameraEnv)(table, split=split, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.budget_range, self.camera_range = budget_range, camera_range
        self.builder = AgentObsBuilder(camera_count)
        self.scale = float(np.mean(table.err_prior))

    def reset(self, budget=None, scenario=None, n=None):
        self.core.n = int(n or self.rng.integers(self.camera_range[0], self.camera_range[1] + 1))
        budget = budget if budget is not None else float(self.rng.uniform(*self.budget_range))
        scenario = scenario or self.core.sample_scenario(self.rng)
        obs = self.core.reset(scenario=scenario, budget_ms_per_cam_s=budget)
        self.builder.reset()
        return self.builder.build(obs)

    def step(self, actions):
        actions = np.asarray(actions, dtype=int)
        proposal, err_est, margin = joint_to_proposal(self.core, actions[:, 0], actions[:, 1])
        obs, _, done, info = self.core.step(proposal, err_est=err_est)
        valid = info["valid"]
        err = float(np.mean(info["err"][valid])) if valid.any() else 0.0
        reward = -err / self.scale - RL_DEADLINE_PENALTY * float(info["deadline_miss"])
        info["margin"] = margin
        agent_obs, state = self.builder.build(obs)
        return agent_obs, state, reward, done, info


class MAPPOPolicy(Policy):
    """A trained MAPPO actor used like any other policy (each camera decides from its own observation)."""

    def __init__(self, actor, obs_norm, name="mappo", camera_count=True):
        self.actor, self.obs_norm, self.name, self.camera_count = actor, obs_norm, name, camera_count

    def reset(self, env):
        self.builder = AgentObsBuilder(self.camera_count)
        self.builder.reset()
        self.margins = []

    def act(self, obs, env):
        import torch
        agent_obs, _ = self.builder.build(obs)
        x = torch.as_tensor(self.obs_norm.normalize(agent_obs), dtype=torch.float32)
        with torch.no_grad():
            li, lm = self.actor(x)
        proposal, err_est, margin = joint_to_proposal(env, li.argmax(-1).numpy(), lm.argmax(-1).numpy())
        self.margins.append(margin)
        return {"proposal": proposal, "err_est": err_est}
