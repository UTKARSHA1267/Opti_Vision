"""Phase 2, Part B (5): the probe-aware estimator ("doubt detector v2").

What changed from v1, and why:
  - inputs built around the LATEST PROBE (its disagreement, its age, which model was probed),
    because recent probe results were by far the most predictive clue;
  - image clues RELATIVE to the camera's own recent past, so it cannot memorise cameras;
  - target = error averaged over the next AHEAD segments, so it learns the predictable part, not luck.

Training rows sample realistic "information states" for each target segment t: which setting ran at
t-1, and how old the latest probe is (or no probe yet).

Train split for fitting (some train cameras held out for early stopping); results on VALIDATION.

Outputs (phase2/outputs/estimator_v2/): point.txt, evaluation.csv, importance.csv, meta.json

Run:  python -u estimator_v2.py
"""
import _paths  # noqa: F401
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from _paths import RESULTS2
from config import SEED
from doubt_detector import MAX_ROUNDS, PARAMS, row_spearman
from p2_config import AHEAD, NO_HISTORY_SHARE, NO_PROBE_SHARE, PROBE_MAX_AGE, SAMPLES_PER_SEGMENT
from predictability_check import window_mean
from probing import FEATURES, ProbeData
from traffic_env import SegmentTable

OUT = RESULTS2 / "estimator_v2"
HOLDOUT_SHARE = 0.15


def sample_states(tab, data, segs, rng):
    """Information states for each target segment. Returns arrays describing each (t, state)."""
    sos = tab.stream_of_seg
    C, B = tab.n_configs, data.probe_rd.shape[1]
    t = np.repeat(segs, SAMPLES_PER_SEGMENT)
    m = len(t)
    same = lambda a, b: (a >= 0) & (sos[np.maximum(a, 0)] == sos[b])  # noqa: E731
    prev1 = np.where(same(t - 1, t), t - 1, -1)
    prev2 = np.where(same(t - 2, t), t - 2, -1)
    j = rng.integers(0, C, m)
    no_hist = rng.random(m) < NO_HISTORY_SHARE
    prev1[no_hist], prev2[no_hist], j[no_hist] = -1, -1, -1

    age = rng.integers(1, PROBE_MAX_AGE + 1, m).astype(float)
    pbase = np.where(rng.random(m) < 0.5, data.config_base[np.maximum(j, 0)], rng.integers(0, B, m))
    src = t - age.astype(int)
    has = same(src, t) & (rng.random(m) >= NO_PROBE_SHARE) & ~no_hist
    rd = np.where(has, data.probe_rd[np.maximum(src, 0), pbase], np.nan)
    age[~has] = np.nan
    pbase = np.where(has, pbase, -1)
    return t, prev1, prev2, j, rd, age, pbase


def build(tab, data, segs, target, rng):
    t, prev1, prev2, j, rd, age, pbase = sample_states(tab, data, segs, rng)
    X = data.make_rows(prev1, prev2, j, rd, age, pbase)
    Y = target[t]
    keep = ~np.isnan(Y).any(axis=1)
    C = tab.n_configs
    return X[np.repeat(keep, C)], Y[keep].reshape(-1), {"age": age[keep], "t": t[keep]}


def scores(name, P, Y):
    tg = Y[:, 0] - Y.min(axis=1)
    pg = P[:, 0] - P.min(axis=1)
    return {"predictor": name, "states": len(Y), "MAE": float(np.abs(P - Y).mean()),
            "rank_corr_across_settings": float(np.nanmean(row_spearman(P, Y))),
            "gap_corr": float(pd.Series(pg).corr(pd.Series(tg), method="spearman"))}


def main():
    tab = SegmentTable()
    data = ProbeData(tab)
    C = tab.n_configs
    sos = tab.stream_of_seg
    split = tab.stream_split[sos]
    target = window_mean(tab.err, sos, 0, AHEAD)

    rng = np.random.default_rng(SEED)
    train_streams = tab.streams_in("train")
    hold = rng.choice(train_streams, size=max(1, int(HOLDOUT_SHARE * len(train_streams))), replace=False)
    is_hold = np.isin(sos, hold)
    fit_s = np.flatnonzero((split == "train") & ~is_hold)
    hold_s = np.flatnonzero((split == "train") & is_hold)
    val_s = np.flatnonzero(split == "val")

    Xf, yf, _ = build(tab, data, fit_s, target, rng)
    Xh, yh, _ = build(tab, data, hold_s, target, rng)
    Xv, yv, mv = build(tab, data, val_s, target, np.random.default_rng(SEED + 1))
    print(f"Rows: fit {len(yf):,}  holdout {len(yh):,}  validation {len(yv):,}  ({len(FEATURES)} inputs)", flush=True)

    dtrain = lgb.Dataset(Xf, yf, free_raw_data=False)
    model = lgb.train(PARAMS, dtrain, num_boost_round=MAX_ROUNDS,
                      valid_sets=[lgb.Dataset(Xh, yh, reference=dtrain)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"Trees: {model.best_iteration}", flush=True)

    P = model.predict(Xv).reshape(-1, C)
    Y = yv.reshape(-1, C)
    prior = np.tile(tab.err_prior, (len(Y), 1))
    age = mv["age"]

    rows = [scores("prior", prior, Y), scores("estimator v2, all states", P, Y)]
    for label, mask in [("  no probe yet", np.isnan(age)), ("  probe 1 segment ago", age == 1),
                        ("  probe 2-3 segments ago", (age >= 2) & (age <= 3)),
                        (f"  probe 4-{PROBE_MAX_AGE} segments ago", age >= 4)]:
        if mask.sum() > 20:
            rows.append(scores(label, P[mask], Y[mask]))
    res = pd.DataFrame(rows).round(4)
    pd.set_option("display.width", 160)
    print("\n=== Validation (target: error averaged over the next "
          f"{AHEAD} segments) ===")
    print(res.to_string(index=False))
    print("\nFor reference, from predictability_check.py: past refdiff of ALL settings over 10 s had gap_corr ~0.55,"
          "\nand the doubt detector v1 had ~0.07. A single cheap probe can't reach the first; it must clearly beat the second.")

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES)
    imp = (100 * imp / imp.sum()).sort_values(ascending=False).round(1)
    print("\n=== What the estimator relies on (share of importance, %) ===")
    print(imp.head(12).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT / "point.txt"))
    res.to_csv(OUT / "evaluation.csv", index=False)
    imp.to_csv(OUT / "importance.csv", header=["importance_%"])
    json.dump({"features": FEATURES, "trees": int(model.best_iteration),
               "holdout_streams": [str(tab.streams[s]) for s in hold]}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()