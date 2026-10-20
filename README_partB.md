# Phase 1, Part B: Profiling (building the "notebook")

Goal: run YOLO with every model (n, s, m) and resolution (320, 480, 640, 960) on the sampled
frames, measure the real cost of each setting, and find the best settings.

| Script | What it does | Needs GPU? | Time |
|---|---|---|---|
| `profile_detections.py` | Runs YOLO, saves all detections | Yes | Several hours (resumable) |
| `benchmark_costs.py` | Measures real ms per frame for each setting | Yes | About 10–15 min |
| `summarize_profiles.py` | Counts, errors, costs, Pareto plot | No | A few minutes |

We profile every 2nd frame (12.5 fps), about 70,000 frames, times 12 model/resolution
combinations, which is roughly 840,000 YOLO runs.

---

## Option A: Kaggle (free GPU, recommended)

### 1. Prepare a small upload on your computer

Create a folder called `phase1_upload` with this structure:

```
phase1_upload\
└── phase1\
    ├── config.py, counting.py, parse_detrac.py, explore.py,
    │   group_locations.py, build_streams.py,
    │   profile_detections.py, benchmark_costs.py, summarize_profiles.py
    └── outputs\
        └── data\
            ├── stream_index.parquet
            ├── ignored_regions.json
            └── sequences.parquet
```

Do NOT include the `datasets` folder (the images are already on Kaggle) or the `venv` folder.
Zip `phase1_upload` (right-click → Send to → Compressed folder).

### 2. Upload it as a private Kaggle dataset

Kaggle → **Create → New Dataset** → upload the zip → title `rl-video-phase1` → keep it **Private** → Create.

### 3. Open a notebook with both datasets

1. Go to the UA-DETRAC dataset page → **Code → New Notebook**.
2. In the right panel: **Add Input** → search your `rl-video-phase1` → add it.
3. Settings (right panel):
   - **Accelerator:** GPU T4 x2 (or P100)
   - **Internet:** On (needed to install Ultralytics and download YOLO weights)
   - **Persistence:** Files, if the option is shown (keeps finished results between sessions)

### 4. Run these cells

**Cell 1: find the folders**
```python
!ls /kaggle/input
!ls /kaggle/input/*
```
Note the exact folder names; they may differ from the ones below.

**Cell 2: copy the code and install**
```python
!cp -r /kaggle/input/rl-video-phase1/phase1 /kaggle/working/
%cd /kaggle/working/phase1
!pip install -q ultralytics
DETRAC = "/kaggle/input/ua-detrac"   # change to the UA-DETRAC folder name from Cell 1
```

**Cell 3: quick test (1 sequence, 1 setting, about 1 minute)**
```python
!python profile_detections.py --detrac-dir {DETRAC} --max-seqs 1 --models yolov8n.pt --sizes 320
```
It should print one line ending in "... detections". If it says image folders not found, fix `DETRAC`.

**Cell 4: measure the real cost of each setting (about 10–15 min)**
```python
!python benchmark_costs.py --detrac-dir {DETRAC}
```

**Cell 5: full profiling (several hours)**
```python
!python profile_detections.py --detrac-dir {DETRAC}
```
Each finished sequence prints a progress line like `[154/1200] yolov8s 480px MVI_40131 ...`.

If the session stops (time limit, disconnection): open the notebook again, run Cells 2 and 5
again. Finished sequences are skipped automatically. If Persistence was not available, first
download your partial results (Cell 7) before the session ends, and upload them back next time.

**Cell 6: summary (a few minutes)**
```python
!python summarize_profiles.py
```

**Cell 7: download the results**
```python
!zip -qr /kaggle/working/profiles.zip outputs/profiles
```
Then download `profiles.zip` from the Output section in the right panel, and unzip it into
`D:\rl_video_project\phase1\` so you get `phase1\outputs\profiles\...` on your computer.

---

## Option B: Your own NVIDIA GPU

1. Install the CUDA version of PyTorch using the command from pytorch.org (choose your CUDA version).
2. `pip install ultralytics`
3. Run in order, no `--detrac-dir` needed:
   ```
   python profile_detections.py --max-seqs 1 --models yolov8n.pt --sizes 320
   python benchmark_costs.py
   python profile_detections.py
   python summarize_profiles.py
   ```

If the GPU runs out of memory, add `--batch 8` to `profile_detections.py`.

---

## What you get (outputs/profiles/)

| File | Contents |
|---|---|
| `raw/<model>_<size>/<seq>.parquet` | Every detection (conf at least 0.10) |
| `costs.csv` | Real ms per frame for each model and resolution |
| `profile_counts.parquet` | Vehicle count of every setting on every profiled frame (used by the RL environment) |
| `configs_summary.csv` | Error and GPU cost of all 48 settings, for train, val and test |
| `pareto_configs.yaml` | The pruned list of good settings the RL agent will choose from |
| `pareto.png` | Error vs compute plot: your first key figure |

## Notes

- Detections are saved at confidence 0.10 or above, but counted at 0.25 or above
  (`COUNT_CONF` in `config.py`). You can change the counting threshold later without re-running YOLO.
- The Pareto choice uses the TRAIN split only. Val and test are never used to make decisions.
- Report costs with the GPU name (`costs.csv` records it), e.g. "measured on an NVIDIA T4".
- Pretrained YOLO uses COCO classes: car, motorcycle, bus, truck are counted as vehicles.

## Done when

- [ ] All profiling jobs finished
- [ ] `costs.csv` exists
- [ ] `summarize_profiles.py` printed the table and saved `pareto.png`
