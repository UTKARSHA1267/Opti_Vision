"""The multi-camera environment (the RL "playground").

N cameras play their streams side by side. Every 2-second segment, a policy proposes one setting
per camera. The environment then:
  1. passes the proposal through the budget scheduler (unless enforcement is switched off),
  2. charges the GPU work, with random variation around the measured cost,
  3. updates a GPU queue: work above the budget waits for the next segment (backlog),
  4. computes latency and deadline misses, counting error, and the reward.

Everything comes from the precomputed segment table, so one step is just array lookups.

Observation (what a real system could know): per camera, the count it SAW last segment, the
change from the segment before, its current setting; globally, the budget share used and backlog.
The true error is only used for scoring (and by the hindsight oracle, which is allowed to cheat).
"""
import numpy as np

from config import (COST_JITTER_SIGMA, DEADLINE_MS, ENV_DIR, EVAL_TARGET, HORIZON_SEGMENTS, N_CAMERAS,
                    REWARD_COST_WEIGHT, REWARD_DEADLINE_WEIGHT, REWARD_OVERRIDE_WEIGHT,
                    SEGMENT_SECONDS, UTILIZATION_TARGET)
from scheduler import BudgetScheduler

SEGMENT_MS = SEGMENT_SECONDS * 1000


class SegmentTable:
    def __init__(self, path=ENV_DIR / "segment_table.npz", target=EVAL_TARGET, verbose=True):
        d = np.load(path, allow_pickle=False)
        for k in d.files:
            setattr(self, k, d[k])
        self.n_configs = len(self.config_names)
        self.target = target
        if target == "reference":
            # Error = difference from the most expensive setting. Needs no labels, so every segment counts.
            self.label_err, self.label_valid = self.err, self.valid
            self.err = self.refdiff.astype(float).copy()
            self.valid = np.ones(len(self.err), dtype=bool)
            train = self.stream_split[self.stream_of_seg] == "train"
            self.err_prior = np.nanmean(self.err[train], axis=0)
        if verbose:
            print(f"[evaluation target: {target}]", flush=True)

    def streams_in(self, split):
        return np.flatnonzero(self.stream_split == split)


class MultiCameraEnv:
    def __init__(self, table, split="train", n_cameras=N_CAMERAS, horizon=HORIZON_SEGMENTS,
                 budget_ms_per_cam_s=7.0, jitter=COST_JITTER_SIGMA, seed=0):
        self.tab = table
        self.split = split
        self.n = n_cameras
        self.horizon = horizon
        self.budget_ms_per_cam_s = budget_ms_per_cam_s
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.scheduler = BudgetScheduler(table.costs, table.err_prior, UTILIZATION_TARGET)
        self.costs = table.costs
        self.top = table.n_configs - 1

    # ---------- scenarios ----------
    def sample_scenario(self, rng=None):
        """Pick cameras (streams of this split) and random start points. Repeats only if the split
        has fewer streams than cameras."""
        rng = rng or self.rng
        pool = self.tab.streams_in(self.split)
        streams = rng.choice(pool, size=self.n, replace=len(pool) < self.n)
        starts = np.array([rng.integers(self.tab.stream_len[s]) for s in streams])
        return {"streams": streams, "starts": starts}

    @property
    def budget(self):
        """GPU milliseconds available per segment for all cameras together."""
        return self.budget_ms_per_cam_s * self.n * SEGMENT_SECONDS

    def _rows(self, t):
        s, st = self.scenario["streams"], self.scenario["starts"]
        return self.tab.stream_start[s] + (st + t) % self.tab.stream_len[s]

    # ---------- episode ----------
    def reset(self, scenario=None, budget_ms_per_cam_s=None, weights=None):
        if budget_ms_per_cam_s is not None:
            self.budget_ms_per_cam_s = budget_ms_per_cam_s
        self.scenario = scenario or self.sample_scenario()
        self.weights = np.ones(self.n) if weights is None else np.asarray(weights, dtype=float)
        self.t = 0
        self.backlog = 0.0
        self.current = np.full(self.n, self.top // 2)
        rows = self._rows(0)
        # Before the first decision the system has seen nothing: use the middle setting's count.
        self.last_pred = self.tab.pred[rows, self.current]
        self.prev_pred = self.last_pred.copy()
        self.last_used_share = 0.0
        return self.observation()

    def observation(self):
        return {
            "last_count": self.last_pred.copy(),
            "count_change": self.last_pred - self.prev_pred,
            "current_setting": self.current / max(self.top, 1),
            "budget_used_share": self.last_used_share,
            "backlog_share": self.backlog / self.budget,
            "budget_ms_per_cam_s": self.budget_ms_per_cam_s,
            "weights": self.weights.copy(),
            "t": self.t,
        }

    def true_errors(self):
        """(n, C) true errors of the CURRENT segment. Only the hindsight oracle may use this."""
        e = self.tab.err[self._rows(self.t)]
        return np.where(np.isnan(e), self.tab.err_prior[None, :], e)

    def current_refdiff(self):
        """(n, C) difference from the best setting in the current segment (needs all settings run)."""
        return self.tab.refdiff[self._rows(self.t)]

    def step(self, proposal, enforce=True, extra_cost=None, floors=None, err_est=None):
        """proposal: setting index per camera. extra_cost: GPU ms per camera spent on top of the
        executed setting (e.g. profiling). floors: lowest allowed setting per camera."""
        proposal = np.asarray(proposal, dtype=int)
        extra = np.zeros(self.n) if extra_cost is None else np.asarray(extra_cost, dtype=float)
        if enforce:
            executed, feasible, overridden = self.scheduler.enforce(
                proposal, self.budget, self.backlog, reserved=extra.sum(),
                weights=self.weights, floors=floors, err_est=err_est)
        else:
            executed, feasible, overridden = proposal, True, np.zeros(self.n, dtype=bool)

        rows = self._rows(self.t)
        planned = self.costs[executed] + extra
        actual = planned * self.rng.lognormal(0.0, self.jitter, self.n) if self.jitter > 0 else planned
        work = float(actual.sum())
        latency_ms = (self.backlog + work) / self.budget * SEGMENT_MS
        deadline_miss = latency_ms > DEADLINE_MS
        self.backlog = max(0.0, self.backlog + work - self.budget)

        err = self.tab.err[rows, executed]
        valid = self.tab.valid[rows]
        w = self.weights
        err_term = float(np.nansum(w * err) / max(w[valid].sum(), 1e-9)) if valid.any() else 0.0
        used_share = work / self.budget
        reward = -(err_term + REWARD_COST_WEIGHT * used_share + REWARD_DEADLINE_WEIGHT * deadline_miss
                   + REWARD_OVERRIDE_WEIGHT * overridden.mean())

        info = {"executed": executed, "proposal": proposal, "overridden": overridden, "feasible": feasible,
                "err": err, "valid": valid, "work_ms": work, "planned_ms": float(planned.sum()),
                "budget_ms": self.budget, "latency_ms": latency_ms, "deadline_miss": bool(deadline_miss),
                "backlog_ms": self.backlog, "rows": rows, "gt_mean": self.tab.gt_mean[rows]}

        self.prev_pred = self.last_pred
        self.last_pred = self.tab.pred[rows, executed]
        self.current = executed
        self.last_used_share = used_share
        self.t += 1
        done = self.t >= self.horizon
        return self.observation(), reward, done, info