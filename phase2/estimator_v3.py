"""Phase 2, Part B (7): the simple probe-based estimator ("doubt detector v3").

Why simple: v1 and v2 (LightGBM with many inputs) learned camera-specific patterns that did not carry
over to new cameras, and ignored the probe. v3 has only the probe-based inputs, so it can only learn
the general relationship "the cheap setting disagrees with the reference -> settings are losing accuracy".

For each setting c (fitted on the TRAIN split):
    predicted error = w0 + w1 * latest probe x exp(-(age - 1) / tau)
                         + w2 * mean of this camera's probes in the last LONG_WINDOW segments
                         + w3 * (camera has been probed)
tau (how fast a probe fades) is chosen on train cameras held out from fitting.

Training states are simulated probe histories: probes every P segments (several P), random phase.
Target: error averaged over the next AHEAD segments.

Outputs (phase2/outputs/results_<target>/estimator_v3/): coefficients.json, evaluation.csv

Run:  python -u estimator_v3.py
"""
import _paths  # noqa: F401
import json

import numpy as np
import pandas as pd

from _paths import RESULTS2
from config import SEED
from doubt_detector import row_spearman
from p2_config import AHEAD, FADE_TAUS, LONG_WINDOW, NO_PROBE_SHARE, TRAIN_PROBE_INTERVALS
from predictability_check import window_mean
from probing import LinearProbeEstimator, ProbeData
from traffic_env import SegmentTable

OUT = RESULTS2 / "estimator_v3"
HOLDOUT_SHARE = 0.15
STATES_PER_SEGMENT = 6


def simulate_states(tab, data, segs, rng):
    """For each target segment t: a probe history with interval P and random phase.
    Returns t, latest probe value, its age, mean of probes in the long window."""
    sos = tab.stream_of_seg
    rd_all = data.probe_rd[:, data.cheap_base]
    T, RL, AG, RM = [], [], [], []
    for t in segs:
        for _ in range(STATES_PER_SEGMENT):
            T.append(t)
            if rng.random() < NO_PROBE_SHARE:
                RL.append(np.nan); AG.append(np.nan); RM.append(np.nan)
                continue
            P = int(rng.choice(TRAIN_PROBE_INTERVALS))
            a = int(rng.integers(1, P + 1))
            times = [t - a - k * P for k in range(LONG_WINDOW // P + 1) if a + k * P <= LONG_WINDOW]
            times = [s for s in times if s >= 0 and sos[s] == sos[t]]
            vals = [rd_all[s] for s in times if not np.isnan(rd_all[s])]
            if not vals:
                RL.append(np.nan); AG.append(np.nan); RM.append(np.nan)
                continue
            RL.append(vals[0]); AG.append(float(a)); RM.append(float(np.mean(vals)))
    return np.array(T), np.array(RL), np.array(AG), np.array(RM)


def fit_weights(X, Y):
    """Least squares per setting: X (m, 4), Y (m, C) -> W (C, 4)."""
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W.T


def scores(name, P, Y):
    tg = Y[:, 0] - Y.min(axis=1)
    pg = P[:, 0] - P.min(axis=1)
    return {"predictor": name, "states": len(Y), "MAE": float(np.abs(P - Y).mean()),
            "rank_corr_across_settings": float(np.nanmean(row_spearman(P, Y))),
            "gap_corr": float(pd.Series(pg).corr(pd.Series(tg), method="spearman"))}


def main():
    tab = SegmentTable()
    data = ProbeData(tab)
    sos = tab.stream_of_seg
    split = tab.stream_split[sos]
    target = window_mean(tab.err, sos, 0, AHEAD)
    ok = ~np.isnan(target).any(axis=1)

    rng = np.random.default_rng(SEED)
    train_streams = tab.streams_in("train")
    hold = rng.choice(train_streams, size=max(1, int(HOLDOUT_SHARE * len(train_streams))), replace=False)
    is_hold = np.isin(sos, hold)
    fit_s = np.flatnonzero((split == "train") & ~is_hold & ok)
    hold_s = np.flatnonzero((split == "train") & is_hold & ok)
    val_s = np.flatnonzero((split == "val") & ok)

    Tf, RLf, AGf, RMf = simulate_states(tab, data, fit_s, rng)
    Th, RLh, AGh, RMh = simulate_states(tab, data, hold_s, rng)
    Tv, RLv, AGv, RMv = simulate_states(tab, data, val_s, np.random.default_rng(SEED + 1))
    rd_fill = float(np.nanmean(RMf))
    print(f"States: fit {len(Tf):,}  holdout {len(Th):,}  validation {len(Tv):,}")

    best = None
    for tau in FADE_TAUS:
        W = fit_weights(LinearProbeEstimator.design(RLf, AGf, RMf, tau, rd_fill), target[Tf])
        P = LinearProbeEstimator.design(RLh, AGh, RMh, tau, rd_fill) @ W.T
        mse = float(np.mean((P - target[Th]) ** 2))
        print(f"  fade tau {tau:>2}: holdout MSE {mse:.6f}")
        if best is None or mse < best[0]:
            best = (mse, tau, W)
    _, tau, W = best
    print(f"Chosen tau = {tau} segments ({2 * tau} s)")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"weights": W.tolist(), "tau": tau, "rd_mean_train": rd_fill,
               "config_names": list(tab.config_names), "columns": ["intercept", "faded_latest_probe",
                                                                   "recent_probe_mean", "has_probe"]},
              open(OUT / "coefficients.json", "w"), indent=1)

    est = LinearProbeEstimator(OUT / "coefficients.json")
    P = est.predict(RLv, AGv, RMv)
    Y = target[Tv]
    prior = np.tile(tab.err_prior, (len(Y), 1))
    rows = [scores("prior", prior, Y), scores("estimator v3, all states", P, Y)]
    for label, mask in [("  no probe yet", np.isnan(AGv)), ("  probe 1 segment ago", AGv == 1),
                        ("  probe 2-3 segments ago", (AGv >= 2) & (AGv <= 3)),
                        ("  probe 4-8 segments ago", (AGv >= 4) & (AGv <= 8)),
                        ("  probe 9-16 segments ago", AGv >= 9)]:
        if mask.sum() > 20:
            rows.append(scores(label, P[mask], Y[mask]))
    res = pd.DataFrame(rows).round(4)
    pd.set_option("display.width", 160)
    print("\n=== Validation (target: error averaged over the next "
          f"{AHEAD} segments) ===")
    print(res.to_string(index=False))
    res.to_csv(OUT / "evaluation.csv", index=False)

    w = pd.DataFrame(W, index=tab.config_names, columns=["intercept", "faded_probe", "recent_mean", "has_probe"])
    print("\n=== Weights per setting (a positive faded_probe weight = a disagreeing probe predicts more error) ===")
    print(w.round(4).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()