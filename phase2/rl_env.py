"""Phase 2, Part E (1): the Gymnasium environment for RL.

The agent STEERS the budget allocation instead of picking 16 settings for each camera directly
(16^6 = 17 million combinations per step). Each step it outputs:
  importance a_i in [-1, 1] per camera  -> weight exp(RL_WEIGHT_SCALE * a_i) on that camera's errors
  reserve    r   in [-1, 1]             -> hold back max(r, 0) * RL_MAX_RESERVE of the budget
Then the same allocation as "budget sharing" turns these into settings within the budget.
The all-zero action is exactly budget sharing. A fresh PPO agent starts with actions around zero,
so RL starts from that strong baseline and only has to learn when to deviate from it.

Observation (only what a real system knows):
  per camera: last observed count, its change, its rolling mean, its current setting
  global:     budget level, share of budget used last segment, backlog
Reward: -(error / mean prior error) - RL_DEADLINE_PENALTY * deadline missed
"""
import _paths  # noqa: F401

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from allocation import allocate
from p2_config import (MARGIN_CHOICES, RL_BUDGET_RANGE, RL_COUNT_WINDOW, RL_DEADLINE_PENALTY,
                       RL_MAX_RESERVE, RL_WEIGHT_SCALE)
from policies import Policy
from traffic_env import MultiCameraEnv

COUNT_SCALE = 20.0
BUDGET_SCALE = 15.0
OBS_LIMIT = 5.0


class ObsBuilder:
    """Turns the core environment's observation into a fixed-size vector, keeping a short count history.
    capacity=True adds what a shared-GPU monitor reports: nominal budget, measured availability share,
    and segments since availability last changed (dynamic environment)."""

    def __init__(self, n_cameras, capacity=False):
        self.n = n_cameras
        self.capacity = capacity
        self.dim = 4 * n_cameras + 3 + (3 if capacity else 0)

    def reset(self):
        self.history = []

    def build(self, obs):
        self.history.append(obs["last_count"].copy())
        self.history = self.history[-RL_COUNT_WINDOW:]
        mean = np.mean(self.history, axis=0)
        per_cam = np.column_stack([obs["last_count"] / COUNT_SCALE,
                                   np.clip(obs["count_change"] / 10.0, -3, 3),
                                   mean / COUNT_SCALE,
                                   obs["current_setting"]])
        glob = [obs["budget_ms_per_cam_s"] / BUDGET_SCALE,
                min(obs["budget_used_share"], 3.0),
                min(obs["backlog_share"], 3.0)]
        if self.capacity:
            glob += [obs["nominal_budget_ms_per_cam_s"] / BUDGET_SCALE, obs["capacity_share"],
                     min(obs["since_change"], 30) / 30.0]
        glob = np.array(glob)
        return np.clip(np.concatenate([per_cam.ravel(), glob]), -OBS_LIMIT, OBS_LIMIT).astype(np.float32)


def action_to_proposal(core, action):
    """Map the agent's action to settings per camera (and the error estimate used to choose them)."""
    action = np.clip(np.asarray(action, dtype=float), -1, 1)
    importance, reserve = action[:core.n], action[core.n]
    w = np.exp(RL_WEIGHT_SCALE * importance) * core.weights
    keep = 1.0 - max(reserve, 0.0) * RL_MAX_RESERVE
    cap = core.scheduler.limit(core.budget, core.backlog) * keep
    err_est = np.tile(core.tab.err_prior, (core.n, 1))
    proposal = allocate(err_est, core.costs, cap, w)
    return proposal, err_est * w[:, None]


def margin_to_proposal(core, action):
    """Margin controller: action = index into MARGIN_CHOICES; budget sharing within what is left."""
    m = MARGIN_CHOICES[int(np.asarray(action).reshape(-1)[0])]
    cap = core.scheduler.limit(core.budget, core.backlog) * (1 - m)
    err_est = np.tile(core.tab.err_prior, (core.n, 1))
    return allocate(err_est, core.costs, cap, core.weights), err_est


class VideoAllocEnv(gym.Env):
    """Gymnasium wrapper around MultiCameraEnv (Phase 1).
    mode="steer":  continuous action, per-camera importance + reserve (see action_to_proposal)
    mode="margin": discrete action, one safety margin from MARGIN_CHOICES per segment"""

    metadata = {"render_modes": []}

    def __init__(self, table, split="train", budget_range=RL_BUDGET_RANGE, seed=0, core_cls=MultiCameraEnv,
                 mode="steer"):
        super().__init__()
        self.mode = mode
        self.core = core_cls(table, split=split, seed=seed)
        self.budget_range = budget_range
        self.obs_builder = ObsBuilder(self.core.n, capacity=hasattr(self.core, "plan_share"))
        self.err_scale = float(np.mean(table.err_prior))
        self.observation_space = spaces.Box(-OBS_LIMIT, OBS_LIMIT, shape=(self.obs_builder.dim,), dtype=np.float32)
        if mode == "margin":
            self.action_space = spaces.Discrete(len(MARGIN_CHOICES))
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.core.n + 1,), dtype=np.float32)
        self.np_rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)
            self.core.rng = np.random.default_rng(seed + 1)
        options = options or {}
        budget = options.get("budget", self.np_rng.uniform(*self.budget_range))
        scenario = options.get("scenario", self.core.sample_scenario(self.np_rng))
        obs = self.core.reset(scenario=scenario, budget_ms_per_cam_s=budget)
        self.obs_builder.reset()
        return self.obs_builder.build(obs), {}

    def step(self, action):
        to_proposal = margin_to_proposal if self.mode == "margin" else action_to_proposal
        proposal, err_est = to_proposal(self.core, action)
        obs, _, done, info = self.core.step(proposal, err_est=err_est)
        valid = info["valid"]
        err = float(np.mean(info["err"][valid])) if valid.any() else 0.0
        reward = -err / self.err_scale - RL_DEADLINE_PENALTY * float(info["deadline_miss"])
        return self.obs_builder.build(obs), reward, done, False, info


class BudgetSharing(Policy):
    """The strong baseline: allocate the whole budget using each setting's typical error."""
    name = "budget_sharing"

    def act(self, obs, env):
        cap = env.scheduler.limit(env.budget, env.backlog)
        return {"proposal": allocate(np.tile(env.tab.err_prior, (env.n, 1)), env.costs, cap, env.weights)}


class RLPolicy(Policy):
    """A trained agent used like any other policy (for evaluation on the same scenarios)."""

    def __init__(self, model, vecnorm=None, name="rl_ppo"):
        self.model, self.vecnorm, self.name = model, vecnorm, name

    def reset(self, env):
        # Use the capacity inputs only if this agent was trained with them (works for old agents too).
        base_dim = ObsBuilder(env.n).dim
        self.obs_builder = ObsBuilder(env.n, capacity=self.model.observation_space.shape[0] > base_dim)
        self.obs_builder.reset()
        self.margins = []           # margins chosen by a margin controller (for analysis)

    def act(self, obs, env):
        x = self.obs_builder.build(obs)
        if self.vecnorm is not None:
            x = self.vecnorm.normalize_obs(x[None])[0]
        action, _ = self.model.predict(x, deterministic=True)
        if isinstance(self.model.action_space, spaces.Discrete):
            proposal, err_est = margin_to_proposal(env, action)
            self.margins.append(MARGIN_CHOICES[int(np.asarray(action).reshape(-1)[0])])
        else:
            proposal, err_est = action_to_proposal(env, action)
        return {"proposal": proposal, "err_est": err_est}
