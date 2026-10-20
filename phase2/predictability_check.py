"""Phase 2, Part B (3): does the recent past predict the next segment?

An agent can only look backwards. This checks the best that ANY backward-looking predictor could do.

Part 1, prediction test on VALIDATION segments. Predictors of each setting's error in segment t:
  prior                 average error per setting (no information about the segment)
  past true errors      mean TRUE error of each setting over the previous PAST segments (cheating, upper limit)
  camera so far         mean TRUE error of each setting over ALL previous segments of this camera (cheating)
  past refdiff          mean difference from the best setting over the previous PAST segments.
                        Measurable without labels (by occasionally probing the expensive setting).
  doubt detector        the LightGBM model from doubt_detector.py
Scored against the raw error of segment t, and against the error averaged over t..t+2 (less luck).

Part 2, a past-only oracle in the environment: allocates using TRUE errors averaged over the previous
PAST segments of the episode. If it gains a lot, tracking recent history is worth it.

Run:  python -u predictability_check.py
"""
import _paths  # noqa: F401
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from _paths import OUT2, RESULTS2
from allocation import allocate
from config import BUDGETS_MS_PER_CAM_S, EVAL_SCENARIOS, SEED
from doubt_detector import candidate_table, history_indices, make_rows, row_spearman
from evaluate_baselines import run_episode
from evaluate_bounds import SmoothedOracle, StaticOracle
from policies import FixedBest, HindsightOracle, Policy
from traffic_env import MultiCameraEnv, SegmentTable

PAST = 5            # previous segments used (10 seconds)
AHEAD = 3           # smoothed target: mean error over t .. t+AHEAD-1
warnings.filterwarnings("ignore", message="Mean of empty slice")


def window_mean(arr, stream_of_seg, start_off, end_off):
    """For each segment t: nan-mean of arr[t+start_off : t+end_off] within the same stream."""
    out = np.full(arr.shape, np.nan)
    S = len(arr)
    for t in range(S):
        a, b = t + start_off, t + end_off
        if a < 0 or b > S or a >= b:
            continue
        if stream_of_seg[a] != stream_of_seg[t] or stream_of_seg[b - 1] != stream_of_seg[t]:
            continue
        out[t] = np.nanmean(arr[a:b], axis=0)
    return out


def expanding_mean(arr, stream_of_seg):
    out = np.full(arr.shape, np.nan)
    for s in np.unique(stream_of_seg):
        idx = np.flatnonzero(stream_of_seg == s)
        x = arr[idx]
        cnt = np.cumsum(~np.isnan(x), axis=0)
        csum = np.cumsum(np.nan_to_num(x), axis=0)
        m = np.where(cnt > 0, csum / np.maximum(cnt, 1), np.nan)
        out[idx[1:]] = m[:-1]           # only segments BEFORE t
    return out


def score(name, P, Y, has_scale=True):
    ok = ~np.isnan(P).any(axis=1) & ~np.isnan(Y).any(axis=1)
    P, Y = P[ok], Y[ok]
    tg = Y[:, 0] - Y.min(axis=1)
    pg = P[:, 0] - P.min(axis=1)
    return {"predictor": name, "segments": int(ok.sum()),
            "MAE": float(np.abs(P - Y).mean()) if has_scale else np.nan,
            "rank_corr_across_settings": float(np.nanmean(row_spearman(P, Y))),
            "gap_corr": float(pd.Series(pg).corr(pd.Series(tg), method="spearman"))}


class PastOracle(Policy):
    """Allocates using TRUE errors averaged over the previous PAST segments of this episode."""
    name = "past_oracle"

    def reset(self, env):
        self.history = []

    def act(self, obs, env):
        if self.history:
            e = np.nanmean(np.stack([env.tab.err[r] for r in self.history[-PAST:]]), axis=0)
            e = np.where(np.isnan(e), env.tab.err_prior[None, :], e)
        else:
            e = np.tile(env.tab.err_prior, (env.n, 1))
        cap = env.scheduler.limit(env.budget, env.backlog)
        return {"proposal": allocate(e, env.costs, cap, env.weights)}

    def update(self, info, env):
        self.history.append(info["rows"])


def main():
    tab = SegmentTable()
    feats = dict(np.load(OUT2 / "features.npz"))
    sos = tab.stream_of_seg
    seg_split = tab.stream_split[sos]
    err = tab.err

    past_err = window_mean(err, sos, -PAST, 0)
    so_far = expanding_mean(err, sos)
    past_rd = window_mean(tab.refdiff, sos, -PAST, 0)
    ahead = window_mean(err, sos, 0, AHEAD)

    p1, p2, p3 = history_indices(tab)
    val = np.flatnonzero((seg_split == "val") & tab.valid & (p1 >= 0))
    C = tab.n_configs
    j = np.full(len(val), C // 2)                       # detector told a middle setting was running
    det_path = RESULTS2 / "doubt_detector" / "point.txt"
    det_pred = None
    if det_path.exists():
        booster = lgb.Booster(model_file=str(det_path))
        det_pred = booster.predict(make_rows(feats["det"], feats["img"], candidate_table(tab), tab.costs,
                                             p1[val], p2[val], p3[val], j)).reshape(-1, C)
    prior = np.tile(tab.err_prior, (len(val), 1))

    for target_name, Y in [("raw error of segment t", err[val]),
                           (f"error averaged over t..t+{AHEAD - 1} (less luck)", ahead[val])]:
        res = pd.DataFrame([
            score("prior", prior, Y),
            score(f"past true errors (last {PAST})  [cheating]", past_err[val], Y),
            score("camera so far  [cheating]", so_far[val], Y),
            score(f"past refdiff (last {PAST})  [measurable by probing]", past_rd[val], Y, has_scale=False),
        ] + ([score("doubt detector v1", det_pred, Y)] if det_pred is not None else [])).round(4)
        pd.set_option("display.width", 170)
        print(f"\n=== Prediction test on validation, target: {target_name} ===")
        print(res.to_string(index=False))

    print("\n=== Past-only oracle in the environment (validation, same scenarios as before) ===")
    env = MultiCameraEnv(tab, split="val")
    rng = np.random.default_rng(SEED)
    scenarios = [env.sample_scenario(rng) for _ in range(EVAL_SCENARIOS)]
    rows = []
    for budget in BUDGETS_MS_PER_CAM_S:
        out = {}
        for P in [FixedBest, StaticOracle, PastOracle, SmoothedOracle, HindsightOracle]:
            res = [run_episode(env, P(), sc, budget, seed=SEED + k) for k, sc in enumerate(scenarios)]
            out[P.name] = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
        fixed, oracle = out["fixed_best"]["err"], out["oracle"]["err"]
        print(f"budget {budget:5.1f}")
        for name, r in out.items():
            share = 100 * (fixed - r["err"]) / (fixed - oracle)
            print(f"  {name:16s} error {r['err']:.4f} ({100 * (r['err'] - fixed) / fixed:+6.1f}% vs fixed) | "
                  f"share of oracle gain {share:6.1f}% | GPU {r['gpu_ms_per_cam_s']:6.2f} ms/cam/s")
            rows.append({"budget_ms_per_cam_s": budget, "method": name, **r})
    RESULTS2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(RESULTS2 / "predictability_val.csv", index=False)
    print(f"\nSaved {RESULTS2 / 'predictability_val.csv'}")


if __name__ == "__main__":
    main()