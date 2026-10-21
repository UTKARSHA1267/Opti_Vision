"""Phase 2, Part E (4): the dynamic environment.

Same cameras and errors as MultiCameraEnv, plus two realistic features:

1. Shared GPU with fluctuating availability. Another job starts and stops on the GPU, so the capacity
   available to video analytics switches between full and DYN_LOW_SHARE of the budget (on average
   every 1 / DYN_SWITCH_PROB segments), with small random variation. Methods only learn the capacity
   AFTER a segment: they plan each segment with the capacity observed in the previous one.
   When capacity drops, work planned for full capacity overloads the GPU and results arrive late.

2. Switching cost. When a camera changes to a different model/resolution, the first
   SWITCH_COST_FRAMES frames of the new model are extra work (warm-up).

Every method is evaluated under exactly the same conditions (same random sequence per scenario).
"""
import _paths  # noqa: F401

import numpy as np

from allocation import allocate
from config import DEADLINE_MS, REWARD_COST_WEIGHT, REWARD_DEADLINE_WEIGHT, REWARD_OVERRIDE_WEIGHT, SEGMENT_SECONDS
from p2_config import DYN_LOW_SHARE, DYN_NOISE, DYN_SWITCH_PROB, SAFETY_MARGIN, SWITCH_COST_FRAMES
from policies import Policy
from traffic_env import SEGMENT_MS, MultiCameraEnv


class DynamicEnv(MultiCameraEnv):
    def __init__(self, table, low_share=DYN_LOW_SHARE, switch_prob=DYN_SWITCH_PROB, noise=DYN_NOISE,
                 switch_frames=SWITCH_COST_FRAMES, **kw):
        super().__init__(table, **kw)
        self.low_share, self.switch_prob, self.noise, self.switch_frames = low_share, switch_prob, noise, switch_frames
        base = np.array([n.rsplit("_k", 1)[0] for n in table.config_names])
        _, self.base_id = np.unique(base, return_inverse=True)
        self.frame_cost = table.costs / (table.fps * SEGMENT_SECONDS)   # ms per processed frame
        self.plan_share = self.true_share = 1.0
        self.low = self.observed_low = False
        self.since_change = 0

    @property
    def base_budget(self):
        return self.budget_ms_per_cam_s * self.n * SEGMENT_SECONDS

    @property
    def budget(self):
        """The capacity methods PLAN with: what was observed in the previous segment."""
        return self.base_budget * self.plan_share

    def reset(self, scenario=None, budget_ms_per_cam_s=None, weights=None):
        self.plan_share = 1.0
        self.true_share = 1.0
        self.low = False
        self.observed_low = False      # level seen in the last segment (what a monitor reports)
        self.since_change = 0          # segments since the observed level last changed
        obs = super().reset(scenario=scenario, budget_ms_per_cam_s=budget_ms_per_cam_s, weights=weights)
        return obs

    def observation(self):
        obs = super().observation()
        obs["budget_ms_per_cam_s"] = self.budget_ms_per_cam_s * self.plan_share   # what the system can see
        obs["nominal_budget_ms_per_cam_s"] = self.budget_ms_per_cam_s            # its allocation on paper
        obs["capacity_share"] = self.plan_share                                   # measured availability
        obs["since_change"] = self.since_change
        return obs

    def _advance_capacity(self):
        if self.rng.random() < self.switch_prob:
            self.low = not self.low
        level = self.low_share if self.low else 1.0
        self.true_share = level * float(self.rng.lognormal(0.0, self.noise))

    def step(self, proposal, enforce=True, extra_cost=None, floors=None, err_est=None):
        proposal = np.asarray(proposal, dtype=int)
        extra = np.zeros(self.n) if extra_cost is None else np.asarray(extra_cost, dtype=float)
        plan_budget = self.budget
        if enforce:
            executed, feasible, overridden = self.scheduler.enforce(
                proposal, plan_budget, self.backlog, reserved=extra.sum(),
                weights=self.weights, floors=floors, err_est=err_est)
        else:
            executed, feasible, overridden = proposal, True, np.zeros(self.n, dtype=bool)

        self._advance_capacity()
        capacity = self.base_budget * self.true_share

        switched = self.base_id[executed] != self.base_id[self.current]
        if self.t == 0:
            switched[:] = False
        switch_ms = np.where(switched, self.switch_frames * self.frame_cost[executed], 0.0)

        rows = self._rows(self.t)
        planned = self.costs[executed] + extra + switch_ms
        actual = planned * self.rng.lognormal(0.0, self.jitter, self.n) if self.jitter > 0 else planned
        work = float(actual.sum())
        latency_ms = (self.backlog + work) / capacity * SEGMENT_MS
        deadline_miss = latency_ms > DEADLINE_MS
        self.backlog = max(0.0, self.backlog + work - capacity)

        err = self.tab.err[rows, executed]
        valid = self.tab.valid[rows]
        w = self.weights
        err_term = float(np.nansum(w * err) / max(w[valid].sum(), 1e-9)) if valid.any() else 0.0
        used_share = work / capacity
        reward = -(err_term + REWARD_COST_WEIGHT * used_share + REWARD_DEADLINE_WEIGHT * deadline_miss
                   + REWARD_OVERRIDE_WEIGHT * overridden.mean())
        info = {"executed": executed, "proposal": proposal, "overridden": overridden, "feasible": feasible,
                "err": err, "valid": valid, "work_ms": work, "planned_ms": float(planned.sum()),
                "budget_ms": capacity, "plan_budget_ms": plan_budget, "latency_ms": latency_ms,
                "deadline_miss": bool(deadline_miss), "backlog_ms": self.backlog, "rows": rows,
                "gt_mean": self.tab.gt_mean[rows], "switches": int(switched.sum()), "switch_ms": float(switch_ms.sum())}

        self.prev_pred = self.last_pred
        self.last_pred = self.tab.pred[rows, executed]
        self.current = executed
        self.last_used_share = used_share
        self.plan_share = self.true_share            # observed after the segment, used for the next plan
        if self.low != self.observed_low:
            self.observed_low, self.since_change = self.low, 0
        else:
            self.since_change += 1
        self.t += 1
        return self.observation(), reward, self.t >= self.horizon, info


class BudgetSharingMargin(Policy):
    """Budget sharing that keeps a fixed safety margin: the standard engineering fix for uncertain capacity."""

    def __init__(self, margin=SAFETY_MARGIN):
        self.margin = margin
        self.name = f"budget_sharing_margin{int(100 * margin)}"

    def act(self, obs, env):
        cap = env.scheduler.limit(env.budget, env.backlog) * (1 - self.margin)
        return {"proposal": allocate(np.tile(env.tab.err_prior, (env.n, 1)), env.costs, cap, env.weights)}


class StateAwareMargin(Policy):
    """Hand-designed rule: keep a safety margin only while the GPU is at full capacity (a drop could come
    at any time); after a drop has been observed, capacity can only go back up, so use the whole budget."""

    def __init__(self, margin=0.5, threshold=0.75):
        self.margin, self.threshold = margin, threshold
        self.name = f"state_aware_margin{int(100 * margin)}"

    def act(self, obs, env):
        at_full = obs.get("capacity_share", 1.0) >= self.threshold
        m = self.margin if at_full else 0.0
        cap = env.scheduler.limit(env.budget, env.backlog) * (1 - m)
        return {"proposal": allocate(np.tile(env.tab.err_prior, (env.n, 1)), env.costs, cap, env.weights)}
