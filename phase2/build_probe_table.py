"""Phase 2, Part B (4): what would a probe measure?

A probe on segment t runs a model/resolution b AND the reference (the base of the most expensive
Pareto setting) on PROBE_FRAMES evenly spaced frames, and measures their disagreement:
    probe_rd[t, b] = mean |count_b - count_ref| / (mean count_ref + 1)      on those frames only
This needs no labels, so a real system can measure it.

Output: phase2/outputs/probe.npz (aligned with phase1/outputs/env/segment_table.npz)

Run:  python -u build_probe_table.py
"""
import _paths  # noqa: F401

import numpy as np
import pandas as pd

from _paths import OUT2
from config import COST_COLUMN, ENV_DIR, PROFILE_DIR, PROFILE_STEP, SEGMENT_FRAMES
from p2_config import PROBE_FRAMES

PER_SEGMENT = SEGMENT_FRAMES // PROFILE_STEP


def main():
    pc = pd.read_parquet(PROFILE_DIR / "profile_counts.parquet").sort_values(["stream", "t"]).reset_index(drop=True)
    pc["off"] = pc.pos - pc.segment * PER_SEGMENT
    seg_id = pc.groupby(["stream", "segment"], sort=True).ngroup().to_numpy()
    S = seg_id.max() + 1
    tab = np.load(ENV_DIR / "segment_table.npz")
    if S != len(tab["valid"]):
        raise SystemExit("Segment count mismatch with segment_table.npz. Re-run phase1 build_segment_table.py.")

    names = list(tab["config_names"])
    config_base = [n.rsplit("_k", 1)[0] for n in names]
    bases = sorted(set(config_base))
    ref = config_base[-1]                                  # base of the most expensive Pareto setting
    costs = pd.read_csv(PROFILE_DIR / "costs.csv")
    frame_cost = {f"{m}_{s}": c for m, s, c in zip(costs.model, costs.imgsz, costs[COST_COLUMN])}

    offsets = np.unique(np.linspace(0, PER_SEGMENT - 1, PROBE_FRAMES).round().astype(int))
    m = pc.off.isin(offsets).to_numpy().astype(float)
    n_m = np.bincount(seg_id, weights=m, minlength=S)
    ref_c = pc[ref].to_numpy(float)
    ref_mean = np.bincount(seg_id, weights=ref_c * m, minlength=S) / np.maximum(n_m, 1)

    rd = np.full((S, len(bases)), np.nan)
    for k, b in enumerate(bases):
        d = np.abs(pc[b].to_numpy(float) - ref_c) * m
        rd[:, k] = np.where(n_m > 0, np.bincount(seg_id, weights=d, minlength=S) / np.maximum(n_m, 1) / (ref_mean + 1), np.nan)

    base_idx = {b: k for k, b in enumerate(bases)}
    OUT2.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT2 / "probe.npz", probe_rd=rd, base_names=np.array(bases),
                        config_base=np.array([base_idx[b] for b in config_base]),
                        base_frame_cost=np.array([frame_cost[b] for b in bases]),
                        ref_base=base_idx[ref], probe_frames=PROBE_FRAMES)
    print(f"Probe table: {S:,} segments x {len(bases)} model/resolutions, reference {ref}, frames at offsets {[int(o) for o in offsets]}")
    pc_cost = PROBE_FRAMES * (np.array([frame_cost[b] for b in bases]) + frame_cost[ref])
    print("Probe cost (ms) when the camera runs each model/resolution:")
    for b, c in zip(bases, pc_cost):
        print(f"  {b:14s} {c:6.2f}")

    # Early check (train): does yesterday's probe predict when the cheapest setting fails next?
    err, valid = tab["err"], tab["valid"]
    sos = tab["stream_of_seg"]
    split = tab["stream_split"][sos]
    t = np.flatnonzero(valid & (split == "train"))
    t = t[(t > 0) & (sos[np.maximum(t - 1, 0)] == sos[t])]
    gap = err[t, 0] - np.nanmin(err[t], axis=1)
    probe_prev = rd[t - 1, base_idx[config_base[0]]]
    c = pd.Series(probe_prev).corr(pd.Series(gap), method="spearman")
    print(f"\nTrain: rank correlation of a probe on the cheapest setting at t-1 with its loss at t: {c:.3f}")
    print(f"Saved {OUT2 / 'probe.npz'}")


if __name__ == "__main__":
    main()
