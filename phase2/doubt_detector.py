"""Phase 2, Part B (2): the doubt detector.

Predicts the counting error of EVERY setting c in segment t, using only what a real system knows at
decision time: the clues of segments t-1, t-2, t-3, observed with the setting j that was running.

One training row = (segment t, observed setting j, candidate setting c):
  inputs  detection clues of (t-1, j), count trend over t-2 and t-3, image clues of t-1 (+ motion at t-2),
          which setting j was running, and which candidate c is being asked about
  target  true error of setting c in segment t

Three LightGBM models: the expected error, and the 10% / 90% quantiles (an error range).
Compared with two simpler predictors: a fixed prior per setting, and confidence-only.

Training uses the TRAIN split (with some train cameras held out for early stopping).
Results are reported on the VALIDATION split. The test split is never touched.

Outputs (phase2/outputs/doubt_detector/): point.txt, q10.txt, q90.txt, meta.json,
          evaluation.csv, importance.csv, calibration.png

Run:  python -u doubt_detector.py
"""
import _paths  # noqa: F401
import json

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _paths import OUT2  # noqa: E402
from config import SEED  # noqa: E402
from traffic_env import SegmentTable  # noqa: E402

OUT = OUT2 / "doubt_detector"
HOLDOUT_SHARE = 0.15
MAX_ROUNDS = 2000
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
              verbose=-1, seed=SEED)
MODEL_CODE = {"n": 0, "s": 1, "m": 2}


def prev_index(tab, steps):
    idx = np.arange(len(tab.valid))
    p = idx - steps
    ok = (p >= 0) & (tab.stream_of_seg[np.maximum(p, 0)] == tab.stream_of_seg)
    return np.where(ok, p, -1)


def history_indices(tab):
    p1 = prev_index(tab, 1)
    p2 = prev_index(tab, 2)
    p3 = prev_index(tab, 3)
    p2 = np.where(p2 >= 0, p2, p1)
    p3 = np.where(p3 >= 0, p3, p2)
    return p1, p2, p3


def candidate_table(tab):
    """Per setting: index, cost, prior error, model size code, input resolution, fps."""
    rows = []
    for c, name in enumerate(tab.config_names):
        model, size, _ = name.split("_")
        rows.append([c, tab.costs[c], tab.err_prior[c], MODEL_CODE.get(model[-1], -1), int(size), tab.fps[c]])
    return np.array(rows, dtype=float)


def feature_names(det_names, img_names):
    return ([f"det_{n}" for n in det_names] + ["count_mean_t-2", "count_mean_t-3"]
            + [f"img_{n}" for n in img_names] + ["img_motion_mean_t-2", "observed_setting", "observed_cost",
                                                 "cand_setting", "cand_cost", "cand_prior_err", "cand_model",
                                                 "cand_imgsz", "cand_fps", "cand_is_observed"])


def make_rows(det, img, cand, costs, p1, p2, p3, j):
    """Feature rows for every candidate setting. p1, p2, p3, j are arrays of equal length m.
    Returns an (m * C, n_features) array ordered as (row, candidate)."""
    C = len(cand)
    base = np.column_stack([det[p1, j, :], det[p2, j, 0], det[p3, j, 0], img[p1, :], img[p2, 3],
                            j, costs[j]])
    X = np.repeat(base, C, axis=0)
    cf = np.tile(cand, (len(base), 1))
    same = (np.repeat(j, C) == cf[:, 0]).astype(float)
    return np.column_stack([X, cf, same])


def build_dataset(tab, feats, segs):
    """All (t, j, c) rows for the given target segments t. Returns X, y, groups of C rows."""
    p1, p2, p3 = history_indices(tab)
    segs = segs[(p1[segs] >= 0) & tab.valid[segs]]
    C = tab.n_configs
    t = np.repeat(segs, C)                  # every observed setting j for every t
    j = np.tile(np.arange(C), len(segs))
    X = make_rows(feats["det"], feats["img"], candidate_table(tab), tab.costs, p1[t], p2[t], p3[t], j)
    y = tab.err[np.repeat(t, C), np.tile(np.arange(C), len(t))]
    keep = ~np.isnan(y).reshape(-1, C).any(axis=1)          # drop whole groups with any missing target
    keep_rows = np.repeat(keep, C)
    return X[keep_rows], y[keep_rows], np.repeat(t, 1)[keep]


def ranks(a):
    return a.argsort(axis=1).argsort(axis=1).astype(float)


def row_spearman(pred, true):
    rp, rt = ranks(pred), ranks(true)
    rp -= rp.mean(axis=1, keepdims=True)
    rt -= rt.mean(axis=1, keepdims=True)
    den = np.sqrt((rp ** 2).sum(axis=1) * (rt ** 2).sum(axis=1))
    return np.where(den > 0, (rp * rt).sum(axis=1) / np.maximum(den, 1e-12), np.nan)


def evaluate(name, pred, y, C, q10=None, q90=None):
    P, Y = pred.reshape(-1, C), y.reshape(-1, C)
    true_gap = Y[:, 0] - Y.min(axis=1)
    pred_gap = P[:, 0] - P.min(axis=1)
    out = {"predictor": name, "MAE": float(np.abs(P - Y).mean()),
           "rank_corr_across_settings": float(np.nanmean(row_spearman(P, Y))),
           "gap_corr": float(pd.Series(pred_gap).corr(pd.Series(true_gap), method="spearman"))}
    if q10 is not None:
        out["coverage_10_90_%"] = float(100 * np.mean((y >= q10) & (y <= q90)))
    return out


def fit(X, y, Xh=None, yh=None, objective="regression", alpha=None, n_rounds=None):
    """LightGBM's own interface (no scikit-learn needed). With a holdout set: early stopping."""
    params = dict(PARAMS)
    if objective == "quantile":
        params.update(objective="quantile", alpha=alpha)
    dtrain = lgb.Dataset(X, y, free_raw_data=False)
    if n_rounds:
        return lgb.train(params, dtrain, num_boost_round=n_rounds)
    dhold = lgb.Dataset(Xh, yh, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dhold],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    tab = SegmentTable()
    feats = dict(np.load(OUT2 / "features.npz"))
    if list(feats["config_names"]) != list(tab.config_names):
        raise SystemExit("features.npz does not match segment_table.npz. Re-run build_features.py.")
    names = feature_names(feats["det_names"], feats["img_names"])
    C = tab.n_configs
    seg_split = tab.stream_split[tab.stream_of_seg]

    rng = np.random.default_rng(SEED)
    train_streams = tab.streams_in("train")
    hold = rng.choice(train_streams, size=max(1, int(HOLDOUT_SHARE * len(train_streams))), replace=False)
    is_hold = np.isin(tab.stream_of_seg, hold)
    fit_segs = np.flatnonzero((seg_split == "train") & ~is_hold)
    hold_segs = np.flatnonzero((seg_split == "train") & is_hold)
    val_segs = np.flatnonzero(seg_split == "val")

    Xf, yf, _ = build_dataset(tab, feats, fit_segs)
    Xh, yh, _ = build_dataset(tab, feats, hold_segs)
    Xv, yv, _ = build_dataset(tab, feats, val_segs)
    print(f"Rows: fit {len(yf):,}  early-stop holdout {len(yh):,}  validation {len(yv):,}  "
          f"({len(names)} inputs, {C} settings)\n", flush=True)

    # Full model (expected error) and an error range (10% / 90% quantiles).
    point = fit(Xf, yf, Xh, yh)
    best_n = point.best_iteration or MAX_ROUNDS
    print(f"Expected-error model: {best_n} trees", flush=True)
    q10 = fit(Xf, yf, objective="quantile", alpha=0.10, n_rounds=best_n)
    q90 = fit(Xf, yf, objective="quantile", alpha=0.90, n_rounds=best_n)

    # Simpler predictors for comparison.
    conf_cols = [names.index(c) for c in ["det_conf_mean", "observed_setting", "observed_cost", "cand_setting",
                                          "cand_cost", "cand_prior_err", "cand_is_observed"]]
    conf_only = fit(Xf[:, conf_cols], yf, Xh[:, conf_cols], yh)
    prior = pd.Series(yf).groupby(Xf[:, names.index("cand_setting")]).mean()
    prior_pred = prior.reindex(Xv[:, names.index("cand_setting")]).to_numpy()

    pv = point.predict(Xv)
    lo, hi = q10.predict(Xv), q90.predict(Xv)
    res = pd.DataFrame([
        evaluate("prior (average error per setting)", prior_pred, yv, C),
        evaluate("confidence only", conf_only.predict(Xv[:, conf_cols]), yv, C),
        evaluate("doubt detector (all clues)", pv, yv, C, lo, hi),
    ]).round(4)
    pd.set_option("display.width", 160)
    print("\n=== Validation results ===")
    print(res.to_string(index=False))
    print("\nMAE: average prediction error (lower is better).")
    print("rank_corr_across_settings: does it order the 16 settings correctly within a segment? (1 = perfect)")
    print("gap_corr: does it know WHEN the cheapest setting loses a lot vs the best? (the key quantity)")
    print("coverage_10_90_%: how often the true error falls inside the predicted range (target 80%)")

    imp = pd.Series(point.feature_importance("gain"), index=names)
    imp = (100 * imp / imp.sum()).sort_values(ascending=False).round(1)
    print("\n=== What the detector relies on (share of importance, %) ===")
    print(imp.head(12).to_string())

    # Calibration: predicted vs actual error, in 10 bins of predicted error.
    bins = pd.qcut(pv, 10, duplicates="drop")
    cal = pd.DataFrame({"pred": pv, "true": yv}).groupby(bins, observed=True).mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(cal.pred, cal.true, "o-", label="doubt detector")
    m = max(cal.pred.max(), cal.true.max()) * 1.05
    ax.plot([0, m], [0, m], "k--", lw=0.8, label="perfect")
    ax.set_xlabel("predicted error")
    ax.set_ylabel("actual error (validation)")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "calibration.png", dpi=140)
    point.save_model(str(OUT / "point.txt"))
    q10.save_model(str(OUT / "q10.txt"))
    q90.save_model(str(OUT / "q90.txt"))
    res.to_csv(OUT / "evaluation.csv", index=False)
    imp.to_csv(OUT / "importance.csv", header=["importance_%"])
    json.dump({"feature_names": names, "config_names": list(tab.config_names), "n_trees": int(best_n),
               "holdout_streams": [str(tab.streams[s]) for s in hold]}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nSaved models and results to {OUT}")


if __name__ == "__main__":
    main()