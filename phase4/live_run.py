"""Phase 4, Part A: the live run (real YOLO on a real GPU).

Everything before this used a SIMULATOR: the cost of each setting came from a batched benchmark, and the
counts from profiling done beforehand. This script checks the simulator against reality.

For each method it replays the SAME 5-minute scenario on 6 TEST cameras in the dynamic environment (shared
GPU whose capacity drops, switching costs). Every 2-second step:
  1. the method chooses a setting (model, resolution, frame rate) for every camera, as in the simulator
  2. the frames each camera would really process in those 2 seconds are loaded from disk
  3. REAL YOLO runs on them on the GPU; frames with the same model and resolution from all cameras are
     batched together (what a real inference server does)
  4. the GPU INFERENCE time Ultralytics measures (what the GPU budget means, as in the simulator's benchmark)
     drives a REAL backlog and deadline check against the same fluctuating capacity; CPU pre/post-processing
     and the whole call's wall time are recorded separately
Saved per method: planned (simulator) vs measured GPU ms per step, simulated vs real lateness, counting error.
With --video, MAPPO's run is also saved as a demo video (6 camera tiles with detections and settings, plus
the GPU budget bar).

Methods: fixed_best, budget_sharing, tuned_margin (40%, the margin tuned on training cameras for this
budget), mappo (the main Phase 3 agent).

Outputs (phase4/outputs/live/): live_steps.csv (one row per method and step), live_summary.csv,
demo_mappo.mp4

Run (on the GPU server, from phase4/):
  EVAL_TARGET=reference python -u live_run.py --steps 10            (quick test, about 1-2 minutes)
  EVAL_TARGET=reference python -u live_run.py --video               (full: 150 steps = 5 minutes of video)
"""
import _paths  # noqa: F401
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import OUT4, PHASE1, RESULTS3
from config import (COUNT_CLASSES, COUNT_CONF, DATA_OUT, DEADLINE_MS, DETRAC_DIR, FRAME_H, FRAME_W, PROFILE_STEP,
                    SEED, SEGMENT_SECONDS)
from counting import keep_mask
from dynamic_env import BudgetSharingMargin, DynamicEnv
from policies import FixedBest
from rl_env import BudgetSharing
from traffic_env import SEGMENT_MS, SegmentTable

OUT = OUT4 / "live"
MAPPO_RUN = "mappo_cam4-16_nocount"


# ---------------- detector ----------------
class Detector:
    """Real YOLO (Ultralytics) on the GPU, one loaded model per model name; times every batched call."""

    def __init__(self, device):
        import torch
        self.torch, self.device, self.models = torch, device, {}
        self.half = device != "cpu"

    def _model(self, name):
        if name not in self.models:
            from ultralytics import YOLO
            local = [PHASE1 / f"{name}.pt", Path(f"{name}.pt")]
            weights = next((str(p) for p in local if p.exists()), f"{name}.pt")
            self.models[name] = YOLO(weights)
        return self.models[name]

    def warmup(self, names_sizes, image):
        for name, size in names_sizes:
            for _ in range(3):
                self._model(name)(image, imgsz=size, device=self.device, half=self.half, verbose=False)

    def run(self, name, size, images):
        """Returns (list of (xyxy, conf, cls) per image, times in ms).
        times = {"gpu": model inference (what the GPU budget means), "pre": CPU preprocessing,
                 "post": postprocessing (NMS), "wall": the whole call}; Ultralytics measures the parts."""
        m = self._model(name)
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = m(images, imgsz=size, conf=COUNT_CONF, classes=COUNT_CLASSES, device=self.device, half=self.half,
                verbose=False)
        out = [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()) for r in res]
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000
        sp = res[0].speed if res else {}                    # per-image averages over this batch
        n = len(images)
        return out, {"gpu": sp.get("inference", 0.0) * n, "pre": sp.get("preprocess", 0.0) * n,
                     "post": sp.get("postprocess", 0.0) * n, "wall": wall}


class FakeDetector:
    """For testing without a GPU: time proportional to the simulator's cost, no boxes."""

    def __init__(self, ms_per_frame):
        self.ms_per_frame = ms_per_frame

    def warmup(self, *a):
        pass

    def run(self, name, size, images):
        gpu = self.ms_per_frame[(name, size)] * len(images) * float(np.random.lognormal(0, 0.1))
        return [(np.zeros((0, 4)), np.zeros(0), np.zeros(0)) for _ in images], \
            {"gpu": gpu, "pre": 0.5 * len(images), "post": 0.3 * len(images), "wall": gpu + 0.8 * len(images)}


# ---------------- frames ----------------
class FrameSource:
    """The frames a setting really processes in one 2-second step of one camera."""

    def __init__(self, table, detrac_dir):
        idx = pd.read_parquet(DATA_OUT / "stream_index.parquet", columns=["stream", "t", "segment", "seq", "frame"])
        idx = idx[idx.t % PROFILE_STEP == 0]
        self.by_seg = {k: g for k, g in idx.groupby(["stream", "segment"])}
        self.tab = table
        self.img_dirs = {}
        try:
            from parse_detrac import find_files
            self.img_dirs = find_files(str(detrac_dir))[1]
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not index images ({e}); using blank frames")
        self.ignored = json.load(open(DATA_OUT / "ignored_regions.json"))

    def frames(self, row, skip):
        stream = str(self.tab.streams[self.tab.stream_of_seg[row]])
        g = self.by_seg.get((stream, int(self.tab.seg_index[row])))
        if g is None:
            return []
        g = g[(g.t // PROFILE_STEP) % skip == 0]
        return list(zip(g.seq, g.frame))

    def load(self, seq, frame):
        import cv2
        d = self.img_dirs.get(seq)
        img = cv2.imread(os.path.join(d, f"img{int(frame):05d}.jpg")) if d else None
        return img if img is not None else np.zeros((FRAME_H, FRAME_W, 3), np.uint8)


def parse_setting(name):
    """'yolov8s_640_k5' -> ('yolov8s', 640, 5)"""
    base, k = name.rsplit("_k", 1)
    model, size = base.rsplit("_", 1)
    return model, int(size), int(k)


# ---------------- demo video ----------------
class Video:
    TILE_W, TILE_H = 480, 270

    def __init__(self, path, fps=5):
        import cv2
        self.cv2 = cv2
        self.w, self.h = 3 * self.TILE_W, 2 * self.TILE_H + 110
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.w, self.h))
        self.last = {}

    def frame(self, cams, plan_ms, real_ms, capacity_ms, budget_ms, margin, late, t):
        cv2 = self.cv2
        canvas = np.full((self.h, self.w, 3), 24, np.uint8)
        for i, c in enumerate(cams):
            img = c["img"] if c["img"] is not None else self.last.get(i)
            if img is None:
                continue
            self.last[i] = img
            tile = img.copy()
            for (x1, y1, x2, y2) in c["boxes"]:
                cv2.rectangle(tile, (int(x1), int(y1)), (int(x2), int(y2)), (80, 220, 120), 2)
            tile = cv2.resize(tile, (self.TILE_W, self.TILE_H))
            cv2.rectangle(tile, (0, 0), (self.TILE_W, 30), (0, 0, 0), -1)
            cv2.putText(tile, f"Camera {i + 1}: {c['setting']}  |  {c['count']} vehicles", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y, x = (i // 3) * self.TILE_H, (i % 3) * self.TILE_W
            canvas[y:y + self.TILE_H, x:x + self.TILE_W] = tile
        y0, bar_x, bar_w = 2 * self.TILE_H + 20, 20, self.w - 40
        full = max(budget_ms, 1e-9)
        cv2.rectangle(canvas, (bar_x, y0), (bar_x + bar_w, y0 + 22), (70, 70, 70), -1)
        cap_x = bar_x + int(bar_w * min(capacity_ms / full, 1))
        use_x = bar_x + int(bar_w * min(real_ms / full, 1))
        cv2.rectangle(canvas, (bar_x, y0), (use_x, y0 + 22), (60, 60, 230) if late else (80, 180, 80), -1)
        cv2.line(canvas, (cap_x, y0 - 4), (cap_x, y0 + 26), (0, 200, 255), 2)
        txt = (f"t = {t * SEGMENT_SECONDS:>3d}s   GPU used {real_ms:6.1f} ms (planned {plan_ms:6.1f})   "
               f"available {capacity_ms:6.1f} of {budget_ms:6.1f} ms   safety margin {100 * margin:3.0f}%   "
               f"{'LATE' if late else 'on time'}")
        cv2.putText(canvas, txt, (bar_x, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, "MAPPO controller, real YOLO on GPU (orange line = GPU capacity available now)",
                    (bar_x, y0 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        self.writer.write(canvas)

    def close(self):
        self.writer.release()


# ---------------- one method ----------------
def run_method(name, make_policy, env, scenario, budget, det, src, steps, video=None):
    tab = env.tab
    pol = make_policy()
    pol.reset(env)
    obs = env.reset(scenario=scenario, budget_ms_per_cam_s=budget)
    real_backlog, rows = 0.0, []
    for t in range(steps):
        a = pol.act(obs, env)
        kw = {k: v for k, v in a.items() if k != "proposal"}
        obs, _, done, info = env.step(a["proposal"], **kw)
        executed, seg_rows = info["executed"], info["rows"]

        # Gather every camera's frames, batch by (model, resolution) across cameras, run real YOLO.
        jobs, groups = [], {}
        for i, (cfg, row) in enumerate(zip(executed, seg_rows)):
            model, size, k = parse_setting(str(tab.config_names[cfg]))
            for seq, fr in src.frames(row, k):
                groups.setdefault((model, size), []).append((i, seq, fr))
            jobs.append((model, size, k))
        real_ms, cpu_ms, wall_ms, last_det = 0.0, 0.0, 0.0, {}
        for (model, size), items in groups.items():
            images = [src.load(seq, fr) for _, seq, fr in items]
            dets, ms = det.run(model, size, images)
            real_ms += ms["gpu"]
            cpu_ms += ms["pre"] + ms["post"]
            wall_ms += ms["wall"]
            for (i, seq, fr), d, img in zip(items, dets, images):
                last_det[i] = (seq, d, img)
        # Models stay loaded and warm here, so a switch's warm-up is added with the simulator's estimate.
        real_ms += info["switch_ms"]
        capacity = info["budget_ms"]
        latency = (real_backlog + real_ms) / capacity * SEGMENT_MS
        real_late = latency > DEADLINE_MS
        real_backlog = max(0.0, real_backlog + real_ms - capacity)
        valid = info["valid"]
        err = float(np.mean(info["err"][valid])) if valid.any() else 0.0
        rows.append({"method": name, "step": t, "planned_ms": info["planned_ms"], "measured_ms": real_ms,
                     "cpu_pre_post_ms": cpu_ms, "wall_ms": wall_ms,
                     "capacity_ms": capacity, "sim_late": bool(info["deadline_miss"]), "real_late": bool(real_late),
                     "err": err, "switches": info.get("switches", 0)})
        if video is not None:
            cams = []
            for i in range(env.n):
                model, size, k = jobs[i]
                seq, d, img = last_det.get(i, (None, None, None))
                boxes, count = [], 0
                if d is not None and len(d[0]):
                    xyxy = d[0]
                    wh = np.column_stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
                    keep = keep_mask(wh, src.ignored.get(seq, []), FRAME_W, FRAME_H)
                    boxes, count = xyxy[keep], int(keep.sum())
                cams.append({"img": img, "boxes": boxes, "count": count,
                             "setting": f"{model} {size}px {12.5 / k:g} fps"})
            margin = pol.margins[-1] if getattr(pol, "margins", None) else 0.0
            for _ in range(2):
                video.frame(cams, info["planned_ms"], real_ms, capacity, env.base_budget, margin, real_late, t)
        if done:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150, help="2-second steps (150 = 5 minutes)")
    ap.add_argument("--budget", type=float, default=7.0, help="GPU ms per camera-second (3 tight, 7 medium, 15 generous)")
    ap.add_argument("--cameras", type=int, default=6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--detrac-dir", default=str(DETRAC_DIR))
    ap.add_argument("--video", action="store_true", help="save MAPPO's run as demo_mappo.mp4")
    ap.add_argument("--fake", action="store_true", help="no GPU/YOLO: simulated timings (for testing the script)")
    ap.add_argument("--methods", nargs="+", default=["fixed_best", "budget_sharing", "tuned_margin", "mappo"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    table = SegmentTable()
    env = DynamicEnv(table, split="test", seed=SEED)
    env.n = args.cameras
    scenario = env.sample_scenario(np.random.default_rng(SEED + 4242))
    src = FrameSource(table, args.detrac_dir)
    names = [parse_setting(str(c)) for c in table.config_names]
    if args.fake:
        ms = {(m, s): float(table.costs[j]) / (table.fps[j] * SEGMENT_SECONDS) for j, (m, s, k) in enumerate(names)}
        det = FakeDetector(ms)
    else:
        import logging
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)           # hides the repeated "'half' is deprecated" notice
        det = Detector(args.device)
        print("Warming up the models on the GPU ...", flush=True)
        det.warmup(sorted({(m, s) for m, s, _ in names}), np.zeros((FRAME_H, FRAME_W, 3), np.uint8))

    makers = {"fixed_best": FixedBest, "budget_sharing": BudgetSharing,
              "tuned_margin": lambda: BudgetSharingMargin(0.4)}
    if "mappo" in args.methods:
        from ma_env import MAPPOPolicy
        from mappo import load
        actor, norm, ckpt = load(RESULTS3 / MAPPO_RUN / "best.pt")
        cc = ckpt.get("info", {}).get("camera_count", True)
        makers["mappo"] = lambda: MAPPOPolicy(actor, norm, name="mappo", camera_count=cc)

    streams = [str(table.streams[s]) for s in scenario["streams"]]
    print(f"Scenario: {args.cameras} test cameras {streams}, budget {args.budget:g} ms/cam/s, {args.steps} steps", flush=True)
    all_rows = []
    for name in args.methods:
        t0 = time.time()
        video = Video(OUT / "demo_mappo.mp4") if (args.video and name == "mappo") else None
        rows = run_method(name, makers[name], env, scenario, args.budget, det, src, args.steps, video)
        if video is not None:
            video.close()
        all_rows += rows
        d = pd.DataFrame(rows)
        print(f"  {name:15s} error {d.err.mean():.4f} | GPU inference planned {d.planned_ms.mean():6.1f} ms/step, "
              f"measured {d.measured_ms.mean():6.1f} | CPU pre/post {d.cpu_pre_post_ms.mean():6.1f} | "
              f"late: simulator {100 * d.sim_late.mean():4.1f}%, real {100 * d.real_late.mean():4.1f}% | "
              f"{time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "live_steps.csv", index=False)
    scale = float(np.mean(table.err_prior))
    summ = df.groupby("method", sort=False).agg(err=("err", "mean"), planned_ms=("planned_ms", "mean"),
                                                measured_ms=("measured_ms", "mean"),
                                                cpu_pre_post_ms=("cpu_pre_post_ms", "mean"), wall_ms=("wall_ms", "mean"),
                                                sim_late=("sim_late", "mean"),
                                                real_late=("real_late", "mean")).reset_index()
    summ["measured_over_planned"] = summ.measured_ms / summ.planned_ms
    summ["score_real"] = summ.err / scale + summ.real_late
    summ["corr_planned_measured"] = [np.corrcoef(g.planned_ms, g.measured_ms)[0, 1] if g.planned_ms.std() > 0 else np.nan
                                     for _, g in df.groupby("method", sort=False)]
    summ.round(4).to_csv(OUT / "live_summary.csv", index=False)
    print("\n" + summ.round(3).to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
