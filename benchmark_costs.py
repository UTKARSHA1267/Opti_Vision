"""Step 1.6b: Measure the cost of every (model, resolution).

Three measurements per setting:
  ms_batched   GPU time per frame when frames from many cameras are processed together in
               batches (how a real multi-camera server runs). THIS IS THE MAIN COST.
  ms_single    time per frame when frames are processed one at a time (mostly fixed overhead
               on a fast GPU), with a breakdown into preprocess / inference / postprocess.
  gflops       billions of calculations per frame: hardware-independent, reproducible anywhere.

Output: outputs/profiles/costs.csv

Run on an IDLE GPU (check nvidia-smi: GPU-Util near 0%):
      python benchmark_costs.py
"""
import argparse
import logging
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import (BENCH_BATCH, DATA_OUT, DETRAC_DIR, FRAME_H, FRAME_W, IMG_SIZES, MODELS,
                    PROFILE_DIR, SAVE_CONF, SEED, VEHICLE_CLASSES)
from parse_detrac import find_files

N_WARMUP = 20


def letterboxed_shape(imgsz, stride=32):
    """Input size YOLO actually uses for a 960x540 frame (long side = imgsz, short side padded to stride)."""
    scale = imgsz / max(FRAME_W, FRAME_H)
    w = math.ceil(FRAME_W * scale / stride) * stride
    h = math.ceil(FRAME_H * scale / stride) * stride
    return h, w


def gflops_of(model, imgsz):
    h, w = letterboxed_shape(imgsz)
    try:
        from ultralytics.utils.torch_utils import get_flops
        g = get_flops(model.model, [h, w])
        if g and g > 0:
            return round(float(g), 2)
    except Exception:
        pass
    try:  # fallback: FLOPs at 640x640 scaled by pixel count
        g640 = model.info(verbose=False)[3]
        return round(float(g640) * (h * w) / (640 * 640), 2)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detrac-dir", default=str(DETRAC_DIR))
    ap.add_argument("--n-frames", type=int, default=320)
    ap.add_argument("--batch", type=int, default=BENCH_BATCH)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    half = device != "cpu"
    dev_name = torch.cuda.get_device_name(0) if device != "cpu" else "CPU"
    print(f"Benchmarking on {dev_name}, batch size {args.batch}")

    idx = pd.read_parquet(DATA_OUT / "stream_index.parquet")
    pool = idx[idx.split == "train"][["seq", "frame"]].drop_duplicates()
    sample = pool.sample(n=min(args.n_frames, len(pool)), random_state=SEED)
    _, img_dirs = find_files(args.detrac_dir)
    images = [cv2.imread(os.path.join(img_dirs[s], f"img{f:05d}.jpg")) for s, f in zip(sample.seq, sample.frame)]
    images = [im for im in images if im is not None]
    print(f"Loaded {len(images)} test frames (decoding excluded from timing)\n")

    def sync():
        if device != "cpu":
            torch.cuda.synchronize()

    rows = []
    print(f"{'setting':16s} {'batched':>10s} {'single':>9s} {'(pre/infer/post)':>20s} {'GFLOPs':>8s}")
    for weights in MODELS:
        model = YOLO(weights)
        for size in IMG_SIZES:
            kw = dict(imgsz=size, conf=SAVE_CONF, classes=VEHICLE_CLASSES, device=device, half=half, verbose=False)

            # Batched: the main cost.
            for i in range(0, min(len(images), args.batch * 3), args.batch):
                model(images[i:i + args.batch], **kw)
            sync()
            t0 = time.perf_counter()
            for i in range(0, len(images), args.batch):
                res = model(images[i:i + args.batch], **kw)
                for r in res:
                    _ = r.boxes.xyxy.cpu()
            sync()
            ms_batched = (time.perf_counter() - t0) * 1000 / len(images)

            # One frame at a time, with breakdown.
            for im in images[:N_WARMUP]:
                model(im, **kw)
            times, pre, inf, post = [], [], [], []
            for im in images:
                sync()
                t0 = time.perf_counter()
                r = model(im, **kw)
                _ = r[0].boxes.xyxy.cpu()
                sync()
                times.append((time.perf_counter() - t0) * 1000)
                sp = getattr(r[0], "speed", None) or {}
                pre.append(sp.get("preprocess", np.nan))
                inf.append(sp.get("inference", np.nan))
                post.append(sp.get("postprocess", np.nan))
            t = np.array(times)

            g = gflops_of(model, size)
            row = {"model": Path(weights).stem, "imgsz": size,
                   "ms_batched": round(ms_batched, 3),
                   "ms_single": round(t.mean(), 3), "ms_single_p95": round(np.percentile(t, 95), 3),
                   "ms_pre": round(np.nanmean(pre), 3) if not np.all(np.isnan(pre)) else np.nan,
                   "ms_infer": round(np.nanmean(inf), 3) if not np.all(np.isnan(inf)) else np.nan,
                   "ms_post": round(np.nanmean(post), 3) if not np.all(np.isnan(post)) else np.nan,
                   "gflops": g, "batch": args.batch, "device": dev_name}
            rows.append(row)
            print(f"{row['model']:8s} {size:4d}px {ms_batched:8.2f} ms {t.mean():6.2f} ms "
                  f"({row['ms_pre']}/{row['ms_infer']}/{row['ms_post']}) {g:8.1f}", flush=True)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PROFILE_DIR / "costs.csv", index=False)
    print(f"\nSaved {PROFILE_DIR / 'costs.csv'}")
    print("Main cost used by summarize_profiles.py: ms_batched (see COST_COLUMN in config.py)")


if __name__ == "__main__":
    main()