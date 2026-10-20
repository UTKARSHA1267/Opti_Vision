"""Shared settings for Phase 1. Change paths here (or set the DETRAC_DIR environment variable)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folder that contains the UA-DETRAC images and XML annotations (any layout; scripts search inside it).
DETRAC_DIR = Path(os.environ.get("DETRAC_DIR", ROOT / "DATASET" / "DETRAC"))

# Where all generated files go.
OUT_DIR = ROOT / "outputs"
DATA_OUT = OUT_DIR / "data"

# UA-DETRAC video properties.
FPS = 25
FRAME_W, FRAME_H = 960, 540

# Decision interval for the RL agent: one decision every 2 seconds.
SEGMENT_SECONDS = 2
SEGMENT_FRAMES = FPS * SEGMENT_SECONDS  # 50 frames

# Frames within this distance of a join between two stitched sequences are excluded from evaluation.
BOUNDARY_MARGIN_FRAMES = SEGMENT_FRAMES

# Random seed used for suggested train/val/test splits.
SEED = 42

# ---------------- Part B: profiling ----------------
# YOLO models (small -> large) and input resolutions to profile.
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
IMG_SIZES = [320, 480, 640, 960]

# Profile every PROFILE_STEP-th frame (25 fps / 2 = 12.5 fps). This halves profiling time.
PROFILE_STEP = 2
PROFILE_FPS = FPS / PROFILE_STEP

# Frame-rate options for the agent: process every k-th PROFILED frame (12.5, 6.25, 2.5, 1.25 fps).
FRAME_SKIPS = [1, 2, 5, 10]

# COCO classes SAVED during profiling: car, motorcycle, bus, truck.
VEHICLE_CLASSES = [2, 3, 5, 7]
# COCO classes COUNTED as vehicles: car, bus, truck.
# Motorcycles are saved but not counted, because UA-DETRAC does not label two-wheelers.
COUNT_CLASSES = [2, 5, 7]

# Save every detection above SAVE_CONF (so the counting threshold can be changed later without re-running YOLO).
SAVE_CONF = 0.10
# Boxes overlapping by at least this IoU are treated as the SAME vehicle, even if YOLO gave them
# different classes (e.g. car + truck); only the most confident one is counted.
DEDUP_IOU = 0.7

# Default counting threshold (used by diagnose_counts.py).
COUNT_CONF = 0.25
# summarize_profiles.py picks the best threshold from this list for EACH (model, size), on the TRAIN split only.
CONF_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

# Images per YOLO call during profiling (lower it if the GPU runs out of memory).
BATCH_SIZE_DEFAULT = 32

# Cost benchmark: frames per batch (frames from many cameras processed together, like a real server).
BENCH_BATCH = 16
# Which measured cost the RL environment uses: "ms_batched" (GPU time per frame in batches) or "gflops".
COST_COLUMN = "ms_batched"

PROFILE_DIR = OUT_DIR / "profiles"

# ---------------- Part C: environment and baselines ----------------
ENV_DIR = OUT_DIR / "env"

# A segment is used for scoring only if at least this share of its frames has valid labels.
MIN_EVAL_FRACTION = 0.5

# Scenario: this many cameras play side by side for this many 2-second segments.
N_CAMERAS = 6
HORIZON_SEGMENTS = 150            # 5 minutes

# GPU budget levels: milliseconds of GPU time per camera per second of video (tight, medium, generous).
BUDGETS_MS_PER_CAM_S = [3.0, 7.0, 15.0]

# The scheduler plans to use at most this share of the budget (spare capacity for timing variation).
UTILIZATION_TARGET = 0.9
# Random variation of real GPU time around the measured cost (log-normal sigma).
COST_JITTER_SIGMA = 0.15
# Results of a segment must be ready within this many ms (one segment = 2000 ms).
DEADLINE_MS = 2000

# Reward = -(error + COST_W * budget share used + DEADLINE_W * deadline misses + OVERRIDE_W * override share)
REWARD_COST_WEIGHT = 0.05
REWARD_DEADLINE_WEIGHT = 1.0
REWARD_OVERRIDE_WEIGHT = 0.02

# Baselines
EVAL_SCENARIOS = 20
COUNT_RULE_THRESHOLDS = (5, 15)   # vehicles: below -> cheap, between -> medium, above -> expensive
REPROFILE_PERIOD = 15             # periodic re-tuning every 15 segments (30 s) per camera
REPROFILE_TAU = 0.10              # pick the cheapest setting within this difference from the best setting
REPROFILE_FRAMES = 5              # profiling runs each model/resolution on this many frames (a short snippet)
