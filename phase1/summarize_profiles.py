"""Step 1.7: Turn raw detections into vehicle counts and find the best settings.

For every combination of model x resolution x frame rate (3 x 4 x 4 = 48 settings):
  - counting error (per 2-second segment, normalised so empty and busy roads are comparable)
  - GPU cost (milliseconds of GPU time per second of video)
Then keeps only the PARETO-OPTIMAL settings: ones that no other setting beats on both error and cost.

Inputs:  outputs/profiles/raw/...   (from profile_detections.py)
         outputs/profiles/costs.csv (from benchmark_costs.py)
Outputs (in outputs/profiles/):
  profile_counts.parquet   predicted count of every (model, size) on every profiled frame (used by the RL environment)
  configs_summary.csv      error and cost of all 48 settings, per split
  pareto_configs.yaml      the pruned setting list the RL agent will choose from
  pareto.png               error vs cost plot  <- your first key result figure
  thresholds.yaml          confidence threshold chosen for each (model, size), on TRAIN only

Run:  python summarize_profiles.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import (CONF_THRESHOLDS, COST_COLUMN, DEDUP_IOU, COUNT_CLASSES, DATA_OUT, FRAME_H, FRAME_SKIPS, FRAME_W, PROFILE_DIR,  # noqa: E402
                    PROFILE_FPS, PROFILE_STEP)
from counting import dedupe_detections, keep_mask  # noqa: E402

THRESHOLDS = CONF_THRESHOLDS


def load_dets(cfg_dir, seqs, ignored):
    """Detections of one (model, size): counted classes only, outside ignored regions."""
    files = {p.stem: p for p in cfg_dir.glob("*.parquet")}
    missing = [s for s in seqs if s not in files]
    if missing:
        return None, missing
    parts = []
    for seq in seqs:
        d = pd.read_parquet(files[seq], columns=["seq", "frame", "left", "top", "width", "height", "conf", "cls"])
        d = d[d.cls.isin(COUNT_CLASSES)]
        if len(d):
            d = d[keep_mask(d[["left", "top", "width", "height"]].to_numpy(), ignored.get(seq, []), FRAME_W, FRAME_H)]
        parts.append(d)
    dets = dedupe_detections(pd.concat(parts, ignore_index=True), DEDUP_IOU)
    return dets[["seq", "frame", "conf"]], []


def counts_at(dets, keys, thr):
    """Vehicle count per row of keys (seq, frame) using confidence threshold thr."""
    c = dets[dets.conf >= thr].groupby(["seq", "frame"]).size().rename("n")
    return keys.join(c, on=["seq", "frame"]).n.fillna(0).astype("int16").to_numpy()


def segment_error(df, pred_col):
    """Mean over evaluation segments of: mean|pred - gt| / (mean gt + 1)."""
    e = df[df.eval_ok]
    g = e.assign(abs_err=(e[pred_col] - e.gt_count).abs()).groupby(["stream", "segment"])
    seg = g.abs_err.mean() / (g.gt_count.mean() + 1)
    return seg


def main():
    idx = pd.read_parquet(DATA_OUT / "stream_index.parquet")
    ignored = json.load(open(DATA_OUT / "ignored_regions.json"))
    costs_path = PROFILE_DIR / "costs.csv"
    if not costs_path.exists():
        raise SystemExit("costs.csv not found. Run benchmark_costs.py first.")
    costs = pd.read_csv(costs_path)
    if COST_COLUMN not in costs.columns:
        raise SystemExit(f"costs.csv has no column {COST_COLUMN}. Re-run benchmark_costs.py.")
    cost_of = {(m, int(s)): c for m, s, c in zip(costs.model, costs.imgsz, costs[COST_COLUMN])}
    print(f"Using cost column: {COST_COLUMN}")

    grid = idx[idx.t % PROFILE_STEP == 0].copy()
    grid["pos"] = grid.t // PROFILE_STEP
    grid["eval_ok"] = grid.annotated & ~grid.near_boundary
    grid = grid.sort_values(["stream", "t"]).reset_index(drop=True)
    seqs = sorted(grid.seq.unique())

    cfg_dirs = sorted((PROFILE_DIR / "raw").glob("*_*"))
    keys = grid[["seq", "frame"]]
    train = grid.split == "train"
    det_cols, thresholds = [], {}
    for cfg_dir in cfg_dirs:
        model, size = cfg_dir.name.rsplit("_", 1)
        dets, missing = load_dets(cfg_dir, seqs, ignored)
        if dets is None:
            print(f"Skipping {cfg_dir.name}: {len(missing)} sequences not profiled yet")
            continue
        col = f"{model}_{size}"
        # Choose this setting's confidence threshold on the TRAIN split only (every frame processed).
        errs = {}
        for thr in THRESHOLDS:
            tmp = grid[train].assign(pred=counts_at(dets, keys[train], thr))
            errs[thr] = segment_error(tmp, "pred").mean()
        best = min(errs, key=errs.get)
        thresholds[col] = float(best)
        grid[col] = counts_at(dets, keys, best)
        det_cols.append((col, model, int(size)))
        print(f"Loaded {col}: best conf {best:.2f} (train error {errs[best]:.4f}; at 0.25 it was {errs.get(0.25, float('nan')):.4f})")
    if not det_cols:
        raise SystemExit("No complete profiles found.")
    with open(PROFILE_DIR / "thresholds.yaml", "w") as f:
        yaml.safe_dump(thresholds, f, sort_keys=True)

    keep = ["stream", "split", "t", "pos", "segment", "seq", "frame", "gt_count", "weather", "eval_ok"]
    grid[keep + [c for c, _, _ in det_cols]].to_parquet(PROFILE_DIR / "profile_counts.parquet", index=False)

    rows = []
    for col, model, size in det_cols:
        if (model, size) not in cost_of:
            print(f"WARNING: no cost measured for {col}, skipped")
            continue
        for k in FRAME_SKIPS:
            # Process every k-th profiled frame; hold the last count in between.
            held = grid[col].where(grid.pos % k == 0).groupby(grid.stream).ffill()
            tmp = grid.assign(pred=held)
            row = {"config": f"{col}_k{k}", "model": model, "imgsz": size, "skip": k,
                   "fps": round(PROFILE_FPS / k, 2),
                   "gpu_ms_per_video_s": round(cost_of[(model, size)] * PROFILE_FPS / k, 2)}
            for split in ("train", "val", "test"):
                seg = segment_error(tmp[tmp.split == split], "pred")
                row[f"err_{split}"] = round(seg.mean(), 4)
            rows.append(row)
    summary = pd.DataFrame(rows).sort_values("gpu_ms_per_video_s").reset_index(drop=True)

    # Pareto front on the TRAIN split (val/test are never used to make choices).
    best, flags = float("inf"), []
    for e in summary.err_train:
        flags.append(e < best)
        best = min(best, e)
    summary["pareto"] = flags
    summary.to_csv(PROFILE_DIR / "configs_summary.csv", index=False)

    pareto = summary[summary.pareto]
    with open(PROFILE_DIR / "pareto_configs.yaml", "w") as f:
        yaml.safe_dump([{"config": r.config, "model": r.model, "imgsz": int(r.imgsz), "skip": int(r.skip),
                         "fps": float(r.fps), "gpu_ms_per_video_s": float(r.gpu_ms_per_video_s),
                         "err_train": float(r.err_train)} for r in pareto.itertuples()], f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary.gpu_ms_per_video_s, summary.err_train, s=18, alpha=0.5, label="all settings")
    ax.plot(pareto.gpu_ms_per_video_s, pareto.err_train, "o-", color="C3", label="Pareto-optimal")
    for r in pareto.itertuples():
        ax.annotate(f"{r.model} {r.imgsz} {r.fps}fps", (r.gpu_ms_per_video_s, r.err_train),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("GPU time per second of video (ms, log scale)")
    ax.set_ylabel("normalised counting error (train)")
    ax.set_title("Accuracy vs compute for every setting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROFILE_DIR / "pareto.png", dpi=140)

    pd.set_option("display.width", 140)
    print("\n" + summary.to_string(index=False))
    print(f"\n{len(pareto)} Pareto-optimal settings kept out of {len(summary)}:")
    print(pareto[["config", "fps", "gpu_ms_per_video_s", "err_train", "err_val"]].to_string(index=False))
    print(f"\nSaved results to {PROFILE_DIR}")


if __name__ == "__main__":
    main()