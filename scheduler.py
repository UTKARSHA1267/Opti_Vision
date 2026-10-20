"""Budget scheduler: makes sure the settings requested for all cameras fit the GPU budget.

Settings are indexed 0..C-1 from cheapest to most expensive (Pareto order), so "one step down"
always means cheaper and usually less accurate.

If the requested allocation is too expensive, the scheduler repeatedly lowers by one step the
camera where it hurts least: smallest (priority x predicted error increase) / (milliseconds saved).
It never lowers a camera below its floor (e.g. a fairness minimum or a locked setting).
"""
import numpy as np


class BudgetScheduler:
    def __init__(self, costs, err_prior, utilization=0.9):
        self.costs = np.asarray(costs, dtype=float)
        self.err_prior = np.asarray(err_prior, dtype=float)
        self.utilization = utilization

    def limit(self, budget, backlog=0.0, reserved=0.0):
        """GPU milliseconds the allocation may use this segment."""
        return self.utilization * budget - backlog - reserved

    def enforce(self, proposal, budget, backlog=0.0, reserved=0.0, weights=None, floors=None, err_est=None):
        """Returns (executed settings, feasible flag, overridden mask).

        proposal: requested setting index per camera
        reserved: GPU ms already committed this segment (e.g. profiling work)
        weights:  camera priorities (default 1)
        floors:   lowest allowed setting per camera (default 0)
        err_est:  (n_cameras, C) predicted error per camera and setting (default: the same prior for all)
        """
        proposal = np.asarray(proposal, dtype=int)
        n = len(proposal)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        fl = np.zeros(n, dtype=int) if floors is None else np.asarray(floors, dtype=int)
        err = np.tile(self.err_prior, (n, 1)) if err_est is None else np.asarray(err_est, dtype=float)

        a = np.maximum(proposal, fl)
        cap = self.limit(budget, backlog, reserved)
        total = self.costs[a].sum()
        while total > cap + 1e-9:
            can = np.flatnonzero(a > fl)
            if len(can) == 0:
                break
            cur, nxt = a[can], a[can] - 1
            saved = self.costs[cur] - self.costs[nxt]
            hurt = w[can] * (err[can, nxt] - err[can, cur])
            i = can[np.argmin(hurt / np.maximum(saved, 1e-9))]
            total -= self.costs[a[i]] - self.costs[a[i] - 1]
            a[i] -= 1
        return a, bool(total <= cap + 1e-9), a != proposal


if __name__ == "__main__":
    s = BudgetScheduler(costs=[1, 2, 4, 8], err_prior=[0.30, 0.20, 0.16, 0.14], utilization=1.0)
    a, ok, over = s.enforce([3, 3, 3], budget=12)
    assert ok and s.costs[a].sum() <= 12, a
    a, ok, _ = s.enforce([3, 3], budget=12, floors=[3, 0])
    assert a[0] == 3 and ok, a
    a, ok, _ = s.enforce([0, 0], budget=1)
    assert not ok
    print("scheduler.py self-test passed")
