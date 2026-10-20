"""Step 1.7b: Diagnose the counting error (run after summarize_profiles.py).

Why: stronger settings (bigger model, higher resolution) showed HIGHER error than weaker ones.
That usually means YOLO finds vehicles the dataset did not label, which we wrongly count as mistakes.

Checks, all on the TRAIN split only:
  1. Bias: does each setting over-count (+) or under-count (-)? Overall, by weather, by traffic level.
  2. Confidence threshold sweep: error for thresholds 0.20 ... 0.70 for each setting.
  3. Box matching on sampled frames: extra detections (FP) vs missed vehicles (FN), and box sizes.
  4. Pictures: the frames where the strongest setting over-counts the most.

Outputs (in outputs/profiles/diagnostics/):
  bias.csv, threshold_sweep.csv, matching.csv, examples/*.jpg

Run:  python diagnose_counts.py
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from config import (COUNT_CLASSES, COUNT_CONF, DEDUP_IOU, DATA_OUT, DETRAC_DIR, FRAME_H, FRAME_W, PROFILE_DIR,
                    PROFILE_STEP, SEED)
from counting import dedupe_detections, keep_mask
from parse_detrac import find_files

DEFAULT_CONFIGS = ["yolov8n_320", "yolov8n_640", "yolov8s_640", "yolov8m_640", "yolov8m_960", "yolov8s_960"]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
N_MATCH_FRAMES = 3000
N_EXAMPLES = 6
DIAG = PROFILE_DIR / "diagnostics"


def density_bin(n):
    return pd.cut(n, [-1, 5, 10, 20, 1000], labels=["0-5", "6-10", "11-20", "20+"])


def load_dets(cfg, seqs, ignored):
    """All saved detections of one setting, restricted to counted classes and outside ignored regions."""
    parts = []
    for seq in seqs:
        p = PROFILE_DIR / "raw" / cfg / f"{seq}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        d = d[d.cls.isin(COUNT_CLASSES)]
        if len(d):
            d = d[keep_mask(d[["left", "top", "width", "height"]].to_numpy(), ignored.get(seq, []), FRAME_W, FRAME_H)]
        parts.append(d)
    return dedupe_detections(pd.concat(parts, ignore_index=True), DEDUP_IOU)


def counts_at(dets, grid_keys, thr):
    c = dets[dets.conf >= thr].groupby(["seq", "frame"]).size().rename("pred")
    return grid_keys.join(c, on=["seq", "frame"]).pred.fillna(0).to_numpy()


def norm_seg_error(grid, pred):
    g = grid.assign(abs_err=np.abs(pred - grid.gt_count.to_numpy())).groupby(["stream", "segment"])
    return float((g.abs_err.mean() / (g.gt_count.mean() + 1)).mean())


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) as left, top, width, height."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    iy = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = ix * iy
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-6)


def greedy_match(pred, gt, thr=0.3):
    iou = iou_matrix(pred, gt)
    matched_p, matched_g = set(), set()
    if iou.size:
        for i, j in sorted(np.argwhere(iou >= thr), key=lambda ij: -iou[ij[0], ij[1]]):
            if i not in matched_p and j not in matched_g:
                matched_p.add(i)
                matched_g.add(j)
    return matched_p, matched_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--detrac-dir", default=str(DETRAC_DIR))
    args = ap.parse_args()
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "examples").mkdir(exist_ok=True)

    idx = pd.read_parquet(DATA_OUT / "stream_index.parquet")
    ignored = json.load(open(DATA_OUT / "ignored_regions.json"))
    grid = idx[(idx.t % PROFILE_STEP == 0) & (idx.split == "train") & idx.annotated & ~idx.near_boundary]
    grid = grid.reset_index(drop=True)
    grid["density"] = density_bin(grid.gt_count)
    keys = grid[["seq", "frame"]]
    seqs = sorted(grid.seq.unique())

    dets_by_cfg = {}
    for cfg in args.configs:
        d = load_dets(cfg, seqs, ignored)
        if d is None:
            print(f"Skipping {cfg}: not fully profiled")
            continue
        dets_by_cfg[cfg] = d
    if not dets_by_cfg:
        raise SystemExit("No profiled settings found.")

    # 1. Bias.
    bias_rows = []
    for cfg, d in dets_by_cfg.items():
        pred = counts_at(d, keys, COUNT_CONF)
        diff = pred - grid.gt_count.to_numpy()
        row = {"config": cfg, "mean_gt": round(grid.gt_count.mean(), 2), "mean_pred": round(pred.mean(), 2),
               "bias": round(diff.mean(), 2), "mae": round(np.abs(diff).mean(), 2),
               "overcount_frames_%": round(100 * (diff > 0).mean(), 1),
               "undercount_frames_%": round(100 * (diff < 0).mean(), 1)}
        for w, m in grid.groupby("weather").groups.items():
            row[f"bias_{w}"] = round(diff[m].mean(), 2)
        for b, m in grid.groupby("density", observed=True).groups.items():
            row[f"bias_gt_{b}"] = round(diff[m].mean(), 2)
        bias_rows.append(row)
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(DIAG / "bias.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== 1. Bias at conf {COUNT_CONF} (positive = YOLO counts MORE than the labels) ===")
    print(bias.to_string(index=False))

    # 2. Threshold sweep.
    sweep = []
    for cfg, d in dets_by_cfg.items():
        for thr in THRESHOLDS:
            pred = counts_at(d, keys, thr)
            sweep.append({"config": cfg, "conf": thr, "norm_seg_error": round(norm_seg_error(grid, pred), 4),
                          "bias": round((pred - grid.gt_count.to_numpy()).mean(), 2)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(DIAG / "threshold_sweep.csv", index=False)
    table = sweep.pivot(index="config", columns="conf", values="norm_seg_error")
    best = sweep.loc[sweep.groupby("config").norm_seg_error.idxmin()].set_index("config")
    table["best_conf"] = best.conf
    table["best_error"] = best.norm_seg_error
    print("\n=== 2. Normalised segment error for each confidence threshold (train) ===")
    print(table.round(3).to_string())

    # 3. Box matching on a sample of frames.
    gt_boxes = pd.read_parquet(DATA_OUT / "boxes.parquet")
    sample = keys.sample(n=min(N_MATCH_FRAMES, len(keys)), random_state=SEED)
    gt_boxes = gt_boxes.merge(sample, on=["seq", "frame"])
    gt_keep = []
    for seq, g in gt_boxes.groupby("seq"):
        gt_keep.append(g[keep_mask(g[["left", "top", "width", "height"]].to_numpy(), ignored.get(seq, []), FRAME_W, FRAME_H)])
    gt_boxes = pd.concat(gt_keep) if gt_keep else gt_boxes.iloc[:0]
    gt_groups = {k: v[["left", "top", "width", "height"]].to_numpy() for k, v in gt_boxes.groupby(["seq", "frame"])}

    match_rows = []
    for cfg, d in dets_by_cfg.items():
        ds = d[d.conf >= COUNT_CONF].merge(sample, on=["seq", "frame"])
        pred_groups = {k: v[["left", "top", "width", "height"]].to_numpy() for k, v in ds.groupby(["seq", "frame"])}
        tp = fp = fn = 0
        fp_h, tp_h, fn_h = [], [], []
        for key in sample.itertuples(index=False):
            k = (key.seq, key.frame)
            p = pred_groups.get(k, np.zeros((0, 4)))
            g = gt_groups.get(k, np.zeros((0, 4)))
            mp, mg = greedy_match(p, g)
            tp += len(mp)
            fp += len(p) - len(mp)
            fn += len(g) - len(mg)
            for i in range(len(p)):
                (tp_h if i in mp else fp_h).append(p[i, 3])
            fn_h.extend(g[j, 3] for j in range(len(g)) if j not in mg)
        n = len(sample)
        match_rows.append({"config": cfg, "frames": n,
                           "matched_per_frame": round(tp / n, 2),
                           "extra_per_frame(FP)": round(fp / n, 2),
                           "missed_per_frame(FN)": round(fn / n, 2),
                           "median_h_matched": round(float(np.median(tp_h)), 1) if tp_h else np.nan,
                           "median_h_extra": round(float(np.median(fp_h)), 1) if fp_h else np.nan,
                           "extra_smaller_than_20px_%": round(100 * np.mean(np.array(fp_h) < 20), 1) if fp_h else np.nan,
                           "median_h_missed": round(float(np.median(fn_h)), 1) if fn_h else np.nan})
    match = pd.DataFrame(match_rows)
    match.to_csv(DIAG / "matching.csv", index=False)
    print(f"\n=== 3. Box matching on {len(sample)} sampled frames (IoU >= 0.3, conf {COUNT_CONF}) ===")
    print(match.to_string(index=False))

    # 4. Example pictures: where the strongest setting over-counts the most.
    strongest = list(dets_by_cfg)[-1] if "yolov8m_960" not in dets_by_cfg else "yolov8m_960"
    d = dets_by_cfg[strongest]
    pred = counts_at(d, keys, COUNT_CONF)
    over = grid.assign(over=pred - grid.gt_count.to_numpy()).sort_values("over", ascending=False)
    picks = over.drop_duplicates("seq").head(N_EXAMPLES)
    _, img_dirs = find_files(args.detrac_dir)
    all_gt = pd.read_parquet(DATA_OUT / "boxes.parquet")
    for r in picks.itertuples():
        img = cv2.imread(os.path.join(img_dirs[r.seq], f"img{r.frame:05d}.jpg"))
        if img is None:
            continue
        for x, y, w, h in ignored.get(r.seq, []):
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (128, 128, 128), 2)
        for b in all_gt[(all_gt.seq == r.seq) & (all_gt.frame == r.frame)].itertuples():
            cv2.rectangle(img, (int(b.left), int(b.top)), (int(b.left + b.width), int(b.top + b.height)), (0, 255, 0), 2)
        for b in d[(d.seq == r.seq) & (d.frame == r.frame) & (d.conf >= COUNT_CONF)].itertuples():
            cv2.rectangle(img, (int(b.left), int(b.top)), (int(b.left + b.width), int(b.top + b.height)), (0, 0, 255), 1)
        label = f"{r.seq} f{r.frame}  labels={r.gt_count}  {strongest}={int(r.gt_count + r.over)}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(str(DIAG / "examples" / f"{r.seq}_f{r.frame}.jpg"), img)
    print(f"\n=== 4. Saved {len(picks)} example pictures to {DIAG / 'examples'} ===")
    print("Green = dataset labels, red = YOLO detections, grey = ignored regions.")


if __name__ == "__main__":
    main()