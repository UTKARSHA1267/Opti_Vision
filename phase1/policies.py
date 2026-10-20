"""Baseline policies and the hindsight oracle.

Every policy has:
  reset(env)                    called at the start of an episode
  act(obs, env) -> dict         proposal (+ optional step options: enforce, extra_cost, floors)
  update(info, env)             called after each step (for policies that learn from what they saw)
"""
import numpy as np

from config import (COUNT_RULE_THRESHOLDS, REPROFILE_FRAMES, REPROFILE_PERIOD, REPROFILE_TAU,
                    SEGMENT_SECONDS, UTILIZATION_TARGET)


def best_fixed_setting(costs, err_prior, n_cameras, budget_ms_per_segment, utilization=UTILIZATION_TARGET):
    """Most accurate single setting that fits the budget when every camera uses it."""
    fits = np.flatnonzero(n_cameras * costs <= utilization * budget_ms_per_segment)
    return int(fits[np.argmin(err_prior[fits])]) if len(fits) else 0


def nearest_by_cost(costs, target):
    return int(np.argmin(np.abs(costs - target)))


class Policy:
    name = "policy"

    def reset(self, env):
        pass

    def update(self, info, env):
        pass


class FixedBest(Policy):
    """What most deployments do: one setting, tuned once, for every camera."""
    name = "fixed_best"

    def reset(self, env):
        self.cfg = best_fixed_setting(env.costs, env.tab.err_prior, env.n, env.budget)

    def act(self, obs, env):
        return {"proposal": np.full(env.n, self.cfg)}


class MaxQuality(Policy):
    """Most expensive setting everywhere, budget ignored: the accuracy ceiling and its cost."""
    name = "max_quality"

    def act(self, obs, env):
        return {"proposal": np.full(env.n, env.top), "enforce": False}


class CountRule(Policy):
    """Quiet camera -> cheap, medium -> the fixed-best setting, busy -> expensive."""
    name = "count_rule"

    def __init__(self, thresholds=COUNT_RULE_THRESHOLDS):
        self.lo_thr, self.hi_thr = thresholds

    def reset(self, env):
        mid = best_fixed_setting(env.costs, env.tab.err_prior, env.n, env.budget)
        self.mid = mid
        self.low = nearest_by_cost(env.costs, env.costs[mid] / 2)
        self.high = nearest_by_cost(env.costs, env.costs[mid] * 2)

    def act(self, obs, env):
        c = obs["last_count"]
        p = np.where(c < self.lo_thr, self.low, np.where(c > self.hi_thr, self.high, self.mid))
        return {"proposal": p.astype(int)}


class PeriodicReprofile(Policy):
    """Chameleon-style: every PERIOD segments a camera briefly profiles every model/resolution on a
    short snippet (REPROFILE_FRAMES frames each), then keeps the cheapest setting whose output is
    within TAU of the most expensive one. The camera keeps running its current setting meanwhile;
    the snippet's GPU time is charged on top. Cameras are staggered so they profile at different times.
    (The segment's measured difference stands in for the snippet's estimate.)"""
    name = "periodic_reprofile"

    def __init__(self, period=REPROFILE_PERIOD, tau=REPROFILE_TAU, frames=REPROFILE_FRAMES):
        self.period, self.tau, self.frames = period, tau, frames

    def reset(self, env):
        self.choice = np.full(env.n, best_fixed_setting(env.costs, env.tab.err_prior, env.n, env.budget))
        self.profiling = np.zeros(env.n, dtype=bool)
        # GPU ms for one frame of each distinct model/resolution (frame skips need no extra runs).
        per_frame = env.costs / (env.tab.fps * SEGMENT_SECONDS)
        base = np.array([n.rsplit("_k", 1)[0] for n in env.tab.config_names])
        _, first = np.unique(base, return_index=True)
        self.snippet_cost = float(per_frame[first].sum() * self.frames)

    def act(self, obs, env):
        t = obs["t"]
        self.profiling = (t + np.arange(env.n)) % self.period == 0
        extra = np.where(self.profiling, self.snippet_cost, 0.0)
        return {"proposal": self.choice.copy(), "extra_cost": extra}

    def update(self, info, env):
        if self.profiling.any():
            rd = env.tab.refdiff[info["rows"]]
            for i in np.flatnonzero(self.profiling):
                ok = np.flatnonzero(rd[i] <= self.tau)
                self.choice[i] = int(ok[0]) if len(ok) else env.top


class HindsightOracle(Policy):
    """Knows the TRUE error of every setting in the current segment (cheating, upper bound) and
    picks the allocation that minimises total weighted error within the budget (Lagrangian search)."""
    name = "oracle"

    def act(self, obs, env):
        err = env.true_errors() * env.weights[:, None]
        cap = env.scheduler.limit(env.budget, env.backlog)
        costs = env.costs
        if np.argmin(err, axis=1).size and costs[np.argmin(err, axis=1)].sum() <= cap:
            return {"proposal": np.argmin(err, axis=1)}
        lo, hi = 0.0, 10.0
        while costs[np.argmin(err + hi * costs[None, :], axis=1)].sum() > cap and hi < 1e6:
            hi *= 10
        for _ in range(40):
            lam = (lo + hi) / 2
            if costs[np.argmin(err + lam * costs[None, :], axis=1)].sum() > cap:
                lo = lam
            else:
                hi = lam
        choice = np.argmin(err + hi * costs[None, :], axis=1)
        # Spend leftover budget greedily on the best remaining upgrades.
        spare = cap - costs[choice].sum()
        while True:
            best, gain_best = None, 0.0
            for i in range(env.n):
                for j in range(choice[i] + 1, env.tab.n_configs):
                    dc = costs[j] - costs[choice[i]]
                    gain = err[i, choice[i]] - err[i, j]
                    if dc <= spare and gain > 0 and gain / dc > gain_best:
                        best, gain_best = (i, j, dc), gain / dc
            if best is None:
                break
            i, j, dc = best
            choice[i], spare = j, spare - dc
        return {"proposal": choice}


ALL_BASELINES = [FixedBest, MaxQuality, CountRule, PeriodicReprofile, HindsightOracle]
