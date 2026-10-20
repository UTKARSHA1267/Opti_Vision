"""Budget-constrained allocation: given an error estimate for every camera and setting, choose one
setting per camera that minimises total (weighted) error while the total cost stays within cap.

Used by the hindsight oracles (with TRUE errors) and later by the scheduler-only baseline
(with PREDICTED errors from the doubt detector).

Method: put a "price" lambda on GPU time, let every camera pick the setting with the lowest
error + lambda x cost, and search lambda until the total fits. Then spend any leftover budget on
the upgrades with the best error reduction per millisecond.
"""
import numpy as np


def allocate(err, costs, cap, weights=None):
    err = np.asarray(err, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n, _ = err.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    e = err * w[:, None]

    choice = np.argmin(e, axis=1)
    if costs[choice].sum() <= cap:
        return choice

    def pick(lam):
        return np.argmin(e + lam * costs[None, :], axis=1)

    lo, hi = 0.0, 1.0
    while costs[pick(hi)].sum() > cap and hi < 1e9:
        hi *= 10
    for _ in range(50):
        mid = (lo + hi) / 2
        if costs[pick(mid)].sum() > cap:
            lo = mid
        else:
            hi = mid
    choice = pick(hi)

    spare = cap - costs[choice].sum()
    rows = np.arange(n)
    while True:
        gain = e[rows, choice][:, None] - e
        dc = costs[None, :] - costs[choice][:, None]
        ok = (dc > 0) & (dc <= spare) & (gain > 0)
        if not ok.any():
            break
        ratio = np.where(ok, gain / np.where(dc > 0, dc, 1.0), -np.inf)
        i, j = np.unravel_index(np.argmax(ratio), ratio.shape)
        spare -= dc[i, j]
        choice[i] = j
    return choice


if __name__ == "__main__":
    costs = np.array([1.0, 2.0, 4.0, 8.0])
    err = np.array([[0.30, 0.20, 0.15, 0.14],    # camera that benefits from upgrades
                    [0.10, 0.10, 0.09, 0.09]])   # camera that barely benefits
    c = allocate(err, costs, cap=6.0)
    assert costs[c].sum() <= 6.0 and c[0] > c[1], c
    print("allocation.py self-test passed:", c)
