"""Step 1.6a: Run YOLO with every (model, resolution) on every sampled frame and save all detections.

Each frame is read from disk ONCE and then run through all model/resolution settings while it is
in memory (reading images is usually slower than YOLO itself). Images are loaded in parallel.

RESUMABLE: results are saved per sequence and setting; finished ones are skipped on the next run.

Outputs: outputs/profiles/raw/<model>_<size>/<seq>.parquet
         columns: seq, frame, left, top, width, height, conf, cls

Run:  python profile_detections.py                       (everything)
      python profile_detections.py --max-seqs 2          (quick test)
"""
import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from config import (BATCH_SIZE_DEFAULT, DATA_OUT, DETRAC_DIR, IMG_SIZES, MODELS, PROFILE_DIR,
                    PROFILE_STEP, SAVE_CONF, VEHICLE_CLASSES)
from parse_detrac import find_files

CHUNK_FRAMES = 400   # frames held in memory at once (about 0.6 GB)
READ_WORKERS = 16    # parallel image readers

COLUMNS = ["seq", "frame", "left", "top", "width", "height", "conf", "cls"]
DTYPES = {"frame": "int32", "left": "float32", "top": "float32", "width": "float32",
          "height": "float32", "conf": "float32", "cls": "int8"}


def sampled_frames(stream_index):
    """Frames at every PROFILE_STEP-th position of each stream (same grid the RL environment uses)."""
    s = stream_index[stream_index.t % PROFILE_STEP == 0]
    return s[["seq", "frame"]].drop_duplicates().sort_values(["seq", "frame"])


def detect(model, images, imgsz, device, half, batch):
    """Run YOLO on in-memory images. Returns one list of [left, top, width, height, conf, cls] per image."""
    out = []
    for i in range(0, len(images), batch):
        results = model(images[i:i + batch], imgsz=imgsz, conf=SAVE_CONF, classes=VEHICLE_CLASSES,
                        device=device, half=half, verbose=False)
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append([])
                continue
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            out.append([[x1, y1, x2 - x1, y2 - y1, c, k] for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls)])
    return out


def save(rows, out_path):
    df = pd.DataFrame(rows, columns=COLUMNS).astype(DTYPES)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)  # file only appears when complete
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detrac-dir", default=str(DETRAC_DIR))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--sizes", nargs="+", type=int, default=IMG_SIZES)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--max-seqs", type=int, default=None, help="only first N sequences (for testing)")
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' for CPU; default: auto")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)  # hide repeated deprecation warnings

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    half = device != "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    stream_index = pd.read_parquet(DATA_OUT / "stream_index.parquet")
    frames = sampled_frames(stream_index)
    seqs = sorted(frames.seq.unique())
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]

    _, img_dirs = find_files(args.detrac_dir)
    missing = [s for s in seqs if s not in img_dirs]
    if missing:
        raise SystemExit(f"Image folders not found for {len(missing)} sequences, e.g. {missing[:3]}. Check --detrac-dir.")

    configs = [(Path(w).stem, w, s) for w in args.models for s in args.sizes]
    out_dirs = {}
    for name, _, size in configs:
        d = PROFILE_DIR / "raw" / f"{name}_{size}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[(name, size)] = d

    models = {}
    total_frames = int(frames[frames.seq.isin(seqs)].shape[0])
    done_frames, t_start = 0, time.time()
    pool = ThreadPoolExecutor(READ_WORKERS)

    for n_seq, seq in enumerate(seqs, 1):
        fr = frames[frames.seq == seq].frame.tolist()
        todo = [(name, w, size) for name, w, size in configs
                if not (out_dirs[(name, size)] / f"{seq}.parquet").exists()]
        if not todo:
            done_frames += len(fr)
            continue

        rows = {(name, size): [] for name, _, size in todo}
        t_read = t_gpu = 0.0
        for c in range(0, len(fr), CHUNK_FRAMES):
            chunk = fr[c:c + CHUNK_FRAMES]
            t0 = time.time()
            paths = [os.path.join(img_dirs[seq], f"img{f:05d}.jpg") for f in chunk]
            images = list(pool.map(cv2.imread, paths))
            bad = [p for p, im in zip(paths, images) if im is None]
            if bad:
                raise SystemExit(f"Could not read {len(bad)} images, e.g. {bad[0]}")
            t_read += time.time() - t0

            t0 = time.time()
            for name, w, size in todo:
                if w not in models:
                    models[w] = YOLO(w)
                dets = detect(models[w], images, size, device, half, args.batch)
                for f, frame_dets in zip(chunk, dets):
                    rows[(name, size)].extend([[seq, f, *d] for d in frame_dets])
            t_gpu += time.time() - t0

        for name, _, size in todo:
            save(rows[(name, size)], out_dirs[(name, size)] / f"{seq}.parquet")

        done_frames += len(fr)
        elapsed = time.time() - t_start
        rate = done_frames / elapsed if elapsed > 0 else 0
        eta = (total_frames - done_frames) / rate / 60 if rate > 0 else float("nan")
        print(f"[{n_seq}/{len(seqs)}] {seq}: {len(fr)} frames x {len(todo)} settings | "
              f"read {t_read:.0f}s, YOLO {t_gpu:.0f}s | elapsed {elapsed / 60:.1f} min, "
              f"ETA {eta:.0f} min", flush=True)

    pool.shutdown()
    print("All profiling jobs finished.")


if __name__ == "__main__":
    main()