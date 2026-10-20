"""Phase 2, Part A: observable features ("clues") for every segment.

Only things a real system could know. When choosing settings for segment t, the agent may use the
clues of segment t-1, computed with the setting that was actually running then.

Detection clues (per segment AND per setting, from the frames that setting processes):
  count_mean, count_std, conf_mean, unsure_share, near_per_frame, small_share, height_mean, crowded_share
Image clues (per segment, same for every setting; from small grayscale copies of the frames):
  brightness, contrast, blur, motion_mean, motion_max

Inputs:  Phase 1 outputs (profile_counts.parquet, raw detections, thresholds.yaml, pareto_configs.yaml,
         segment_table.npz) and the images.
Output:  phase2/outputs/features.npz  (aligned row-by-row with phase1/outputs/env/segment_table.npz)
         phase2/outputs/image_features/<seq>.parquet  (cache, so re-runs are fast)

Run:  python -u build_features.py
"""
import _paths  # noqa: F401
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import yaml

from _paths import OUT2
from config import (COUNT_CLASSES, DATA_OUT, DEDUP_IOU, DETRAC_DIR, ENV_DIR, FRAME_H, FRAME_W, PROFILE_DIR,
                    PROFILE_STEP, SEGMENT_FRAMES)
from counting import dedupe_detections, keep_mask
from parse_detrac import find_files

PER_SEGMENT = SEGMENT_FRAMES // PROFILE_STEP
UNSURE_BAND = 0.15      # counted boxes with conf < threshold + this are "unsure"
NEAR_BAND = 0.10        # boxes with threshold - this <= conf < threshold are "almost counted"
SMALL_H = 30            # boxes shorter than this (pixels) are "small"
CROWD_IOU = 0.20        # a counted box overlapping another by at least this IoU is "crowded"
READ_WORKERS = 16

DET_FEATURES = ["count_mean", "count_std", "conf_mean", "unsure_share", "near_per_frame",
                "small_share", "height_mean", "crowded_share"]
IMG_FEATURES = ["brightness", "contrast", "blur", "motion_mean", "motion_max"]


def overlap_flags(df, iou_thr):
    """True for each box that overlaps another box in the same frame by IoU >= iou_thr."""
    flag = np.zeros(len(df), dtype=bool)
    if len(df) == 0:
        return flag
    df = df.reset_index(drop=True)
    cols = ["frame", "left", "top", "width", "height"]
    for _, g in df.groupby("seq", sort=False):
        a = g[cols].assign(_i=g.index.to_numpy())
        p = a.merge(a, on="frame", suffixes=("_a", "_b"))
        p = p[p._i_a != p._i_b]
        if p.empty:
            continue
        ix = np.clip(np.minimum(p.left_a + p.width_a, p.left_b + p.width_b) - np.maximum(p.left_a, p.left_b), 0, None)
        iy = np.clip(np.minimum(p.top_a + p.height_a, p.top_b + p.height_b) - np.maximum(p.top_a, p.top_b), 0, None)
        inter = ix * iy
        iou = inter / np.maximum(p.width_a * p.height_a + p.width_b * p.height_b - inter, 1e-6)
        flag[p._i_a[iou >= iou_thr].unique()] = True
    return flag


def per_frame_detection_stats(model_size, thr, seqs, ignored):
    """Per (seq, frame): counted boxes, conf sum, unsure, near, small, height sum, crowded."""
    parts = []
    for seq in seqs:
        d = pd.read_parquet(PROFILE_DIR / "raw" / model_size / f"{seq}.parquet")
        d = d[d.cls.isin(COUNT_CLASSES) & (d.conf >= thr - NEAR_BAND)]
        if len(d):
            d = d[keep_mask(d[["left", "top", "width", "height"]].to_numpy(), ignored.get(seq, []), FRAME_W, FRAME_H)]
        parts.append(d)
    d = dedupe_detections(pd.concat(parts, ignore_index=True), DEDUP_IOU)
    near = d[d.conf < thr]
    c = d[d.conf >= thr].reset_index(drop=True)
    c = c.assign(unsure=c.conf < thr + UNSURE_BAND, small=c.height < SMALL_H,
                 crowded=overlap_flags(c, CROWD_IOU))
    g = c.groupby(["seq", "frame"])
    stats = pd.DataFrame({"n": g.size(), "conf_sum": g.conf.sum(), "unsure": g.unsure.sum(),
                          "small": g.small.sum(), "h_sum": g.height.sum(), "crowded": g.crowded.sum()})
    stats["near"] = near.groupby(["seq", "frame"]).size()
    return stats.fillna(0)


def image_stats_for_seq(seq, frames, img_dir, cache_dir):
    """Brightness, contrast, blur and motion for each profiled frame of one sequence (cached)."""
    path = cache_dir / f"{seq}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    paths = [os.path.join(img_dir, f"img{f:05d}.jpg") for f in frames]
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        imgs = list(pool.map(lambda p: cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4), paths))
    rows, prev = [], None
    for f, g in zip(frames, imgs):
        if g is None:
            rows.append((f, np.nan, np.nan, np.nan, np.nan))
            continue
        g32 = g.astype(np.float32)
        motion = float(np.abs(g32 - prev).mean()) if prev is not None and prev.shape == g32.shape else np.nan
        rows.append((f, float(g32.mean()), float(g32.std()), float(cv2.Laplacian(g, cv2.CV_32F).var()), motion))
        prev = g32
    df = pd.DataFrame(rows, columns=["frame", "brightness", "contrast", "blur", "motion"]).assign(seq=seq)
    df.to_parquet(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detrac-dir", default=str(DETRAC_DIR))
    args = ap.parse_args()
    OUT2.mkdir(parents=True, exist_ok=True)

    pc = pd.read_parquet(PROFILE_DIR / "profile_counts.parquet").sort_values(["stream", "t"]).reset_index(drop=True)
    pc["off"] = pc.pos - pc.segment * PER_SEGMENT
    seg_id = pc.groupby(["stream", "segment"], sort=True).ngroup().to_numpy()
    S = seg_id.max() + 1
    table = np.load(ENV_DIR / "segment_table.npz")
    if S != len(table["valid"]):
        raise SystemExit(f"Segment count mismatch ({S} vs {len(table['valid'])}). Re-run phase1 build_segment_table.py.")
    names = list(table["config_names"])

    pareto = {r["config"]: r for r in yaml.safe_load(open(PROFILE_DIR / "pareto_configs.yaml"))}
    thresholds = yaml.safe_load(open(PROFILE_DIR / "thresholds.yaml"))
    ignored = json.load(open(DATA_OUT / "ignored_regions.json"))
    seqs = sorted(pc.seq.unique())
    keys = pc[["seq", "frame"]]

    # ---- Detection clues ----
    det = np.zeros((S, len(names), len(DET_FEATURES)))
    cache = {}
    for j, name in enumerate(names):
        r = pareto[name]
        ms = f"{r['model']}_{r['imgsz']}"
        if ms not in cache:
            print(f"Detection statistics for {ms} (threshold {thresholds[ms]:.2f}) ...", flush=True)
            cache[ms] = keys.join(per_frame_detection_stats(ms, thresholds[ms], seqs, ignored),
                                  on=["seq", "frame"]).fillna(0)
        st = cache[ms]
        proc = (pc.off % r["skip"] == 0).to_numpy().astype(float)
        def ssum(x):
            return np.bincount(seg_id, weights=np.asarray(x, dtype=float) * proc, minlength=S)
        n_proc = np.maximum(ssum(np.ones(len(pc))), 1)
        n_sum = ssum(st.n)
        nb = np.maximum(n_sum, 1)
        mean = n_sum / n_proc
        det[:, j, 0] = mean
        det[:, j, 1] = np.sqrt(np.maximum(ssum(st.n ** 2) / n_proc - mean ** 2, 0))
        det[:, j, 2] = np.where(n_sum > 0, ssum(st.conf_sum) / nb, 0)
        det[:, j, 3] = ssum(st.unsure) / nb
        det[:, j, 4] = ssum(st.near) / n_proc
        det[:, j, 5] = ssum(st.small) / nb
        det[:, j, 6] = np.where(n_sum > 0, ssum(st.h_sum) / nb, 0)
        det[:, j, 7] = ssum(st.crowded) / nb

    # ---- Image clues ----
    _, img_dirs = find_files(args.detrac_dir)
    cache_dir = OUT2 / "image_features"
    cache_dir.mkdir(exist_ok=True)
    parts = []
    for k, seq in enumerate(seqs, 1):
        frames = sorted(pc.frame[pc.seq == seq].unique())
        parts.append(image_stats_for_seq(seq, frames, img_dirs[seq], cache_dir))
        if k % 10 == 0 or k == len(seqs):
            print(f"Image statistics: {k}/{len(seqs)} sequences", flush=True)
    im = keys.merge(pd.concat(parts), on=["seq", "frame"], how="left")
    cnt = np.maximum(np.bincount(seg_id, weights=im.brightness.notna().to_numpy(float), minlength=S), 1)
    img = np.zeros((S, len(IMG_FEATURES)))
    for c, col in enumerate(["brightness", "contrast", "blur"]):
        img[:, c] = np.bincount(seg_id, weights=im[col].fillna(0).to_numpy(), minlength=S) / cnt
    mot = im.motion.to_numpy()
    mcnt = np.maximum(np.bincount(seg_id, weights=~np.isnan(mot), minlength=S), 1)
    img[:, 3] = np.bincount(seg_id, weights=np.nan_to_num(mot), minlength=S) / mcnt
    img[:, 4] = pd.Series(mot).groupby(seg_id).max().reindex(range(S)).fillna(0).to_numpy()

    np.savez_compressed(OUT2 / "features.npz", det=det, det_names=np.array(DET_FEATURES),
                        img=img, img_names=np.array(IMG_FEATURES), config_names=np.array(names))
    print(f"\nSaved {OUT2 / 'features.npz'}: detection clues {det.shape}, image clues {img.shape}")

    # ---- Early check: which clues predict when the cheapest setting fails? (train only) ----
    err, valid = table["err"], table["valid"]
    split = table["stream_split"][table["stream_of_seg"]]
    m = valid & (split == "train")
    gap = err[m, 0] - np.nanmin(err[m], axis=1)          # how much the cheapest setting loses vs the best
    feats = {f"det_{n} (cheapest setting)": det[m, 0, i] for i, n in enumerate(DET_FEATURES)}
    feats.update({f"img_{n}": img[m, i] for i, n in enumerate(IMG_FEATURES)})
    corr = pd.Series({k: pd.Series(v).corr(pd.Series(gap), method="spearman") for k, v in feats.items()})
    print("\nRank correlation with 'how much the cheapest setting loses vs the best' (train segments):")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).round(3).to_string())
    print("\n|correlation| above ~0.3 = a useful clue on its own; the estimator combines them all.")


if __name__ == "__main__":
    main()
