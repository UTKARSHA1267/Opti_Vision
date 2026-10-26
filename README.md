# Opti_Vision: Budget-Aware Multi-Agent RL for City-Scale Traffic Video Analytics

**One GPU, many traffic cameras, a fixed compute budget.** Every 2 seconds, Opti_Vision decides which YOLO setting
(model, resolution, frame rate) each camera gets, so the city counts vehicles as accurately as possible without
results arriving late. A **MAPPO** policy trained on 4–16 cameras runs systems of up to **48 cameras without
retraining**, and on unseen test cameras beats the fixed configurations used in practice by **14–26%**.

---

## The problem

A city runs object detection on hundreds of camera feeds with limited GPUs. Running the best detector on every
camera is too expensive; running the cheapest one everywhere misses vehicles. Deployments today usually pick
**one fixed setting for all cameras**. But traffic changes over time and between cameras, and the GPU is often
shared with other jobs, so its available capacity drops without warning.

**Goal:** choose a setting per camera, every 2 seconds, to minimise counting error and late results under a shared,
fluctuating GPU budget.

## Results (unseen test cameras)

Score = counting error ÷ typical error + share of late results (lower is better). Percentages are paired
differences over 30 scenarios with 95% bootstrap confidence intervals; ranges cover three budgets
(3, 7, 15 GPU-ms per camera-second).

**MAPPO (main agent) vs baselines, shared GPU with capacity drops:**

| MAPPO vs | 6 cameras | 12 cameras | 24 cameras | 48 cameras |
|---|---|---|---|---|
| Fixed best setting (deployment practice) | −14 to −26% | −17 to −26% | −14 to −26% | −16 to −26% |
| Count-based rule | −13 to −26% | −13 to −26% | −13 to −23% | −11 to −27% |
| Chameleon-style periodic re-profiling | −42 to −53% | −37 to −51% | −36 to −49% | −35 to −48% |
| Budget sharing (optimisation, no margin) | −5 to −32% | −5 to −31% | −5 to −32% | −6 to −33% |
| Safety margin hand-tuned per size and budget | +0.5 to +2.0% | +0.9 to +2.4% | +0.5 to +4.0% | +0.5 to +3.1% |

All differences against the traditional methods and budget sharing are statistically significant.
MAPPO matches a hand-tuned safety margin within 0.5–4% **without any tuning**, using one policy for every system size.

**Single-agent PPO (6 cameras):** −15 to −26% vs fixed best, −5 to −30% vs budget sharing, on par with the tuned
margin (within 0.8%). Three training seeds agreed closely.

**Real-GPU check** (6 test cameras, 5 minutes, real YOLO on the GPU, medium budget). Live batches are smaller
than the batch-64 benchmark the simulator uses, so real inference took 1.8× longer on average; GPU capacity is
calibrated by that one factor, the same for every method:

| Method | Late results on the real GPU | Real GPU inference per 2-s step |
|---|---|---|
| Fixed best | 68% | 91.6 ms |
| Budget sharing | 33% | 86.9 ms |
| Tuned margin | 10% | 67.9 ms |
| **MAPPO** | **17%** | **70.1 ms** |

On real hardware MAPPO cuts late results from 68% to 17% versus the fixed setting and uses 23% less GPU,
with the same ranking of methods as the simulator.

| | |
|---|---|
| ![Test results](phase4/outputs/figures/fig_test.png) | ![Pareto front](phase4/outputs/figures/fig_pareto.png) |
| ![Scaling](phase4/outputs/figures/fig_scaling.png) | ![Training](phase4/outputs/figures/fig_training.png) |
| ![Real GPU](phase4/outputs/figures/fig_live_calibrated.png) | |

## RL formulation

**Environment.** 6 (or N) cameras stream UA-DETRAC traffic video. Each 2-second step, every camera gets one of
**16 Pareto-optimal YOLO settings** (chosen from 48 = 3 models × 4 resolutions × 4 frame rates, from 1.2 to
36.4 GPU-ms per video-second). A scheduler enforces the GPU budget. In the **dynamic world**, another job takes
half the GPU at random (on average every 30 s) and the system only sees capacity after the fact; switching a
camera's model or resolution costs 2 warm-up frames. Counting error is measured against the most expensive
setting (as in Chameleon and VideoStorm), computed from real YOLO outputs profiled on 70,000 frames.

**Single-agent PPO (Phase 2).**
- *State* (30 values): per camera, the last count, its change, its 10-second mean and its current setting; globally,
  the planning budget, budget used, backlog, nominal budget, measured GPU availability, time since it changed.
- *Action:* one safety margin from {0, 10, 20, 30, 40, 50}% per step; the remaining budget is allocated by a
  Lagrangian optimiser (budget sharing).
- *Reward:* r = −(error ÷ typical error) − 1·(late result).

**MAPPO (Phase 3).** One agent per camera, one shared actor network (parameter sharing), centralised critic,
team reward.
- *Agent observation* (12): own camera (count, change, 10-s mean, setting, relative priority) + 7 system values.
- *Critic state* (17): system values + mean and max of all cameras' features (works for any number of cameras).
- *Action* (discrete, per agent): importance ∈ {low, normal, high} × margin vote ∈ {0, …, 50%}; the margin is the
  **average** of votes, and importance weights each camera in the allocation.
- *Reward:* the same team reward for every agent.
- *Training:* PPO with GAE (γ = 0.95, λ = 0.95, clip 0.2, entropy 0.01), 8 parallel environments, 2 M steps,
  4–16 cameras per episode. Independent PPO (local critics) is the ablation.

**Baselines:** fixed best setting, count-based rule, Chameleon-style periodic re-profiling, budget sharing, fixed
safety margins (10–50%), a margin tuned per budget and size on training cameras, a state-aware margin rule, and
cheating oracles (past, smoothed, full) as upper bounds.

## What we learned

- **Budget sharing** (mixing settings across cameras to use the whole budget) cuts error by 15–25% vs one fixed
  setting with no learning at all. It became the strong baseline RL had to beat.
- Under a **shared, fluctuating GPU**, budget sharing is late up to 88% of the time; RL learns how much safety
  margin to keep, cutting late results to ~5–15%.
- Two design fixes made MAPPO work: **averaging margin votes** instead of taking the median (credit assignment),
  and **removing the camera-count input**, whose values at 48 cameras were never seen in training.
- Passive clues (confidence, brightness, motion) did **not** predict where compute is needed on new cameras;
  learned predictors memorised cameras. Probing the expensive model was either too costly or too noisy.
- With identical training, MAPPO and independent PPO perform similarly; the central critic helped only at the
  medium budget.

## Repository structure

```
phase1/   data preparation, YOLO profiling, simulator, baselines
phase2/   observable features, bounds, probing, single-agent PPO, dynamic environment, final evaluation
phase3/   MAPPO (own PyTorch implementation), independent PPO, scaling, test evaluation
phase4/   real-GPU check, figures for the report and README
```

| Key file | Purpose |
|---|---|
| `phase1/config.py` | All shared settings (cameras, budgets, models, evaluation target) |
| `phase1/traffic_env.py` | Multi-camera environment over the profiled segment table |
| `phase2/dynamic_env.py` | Shared-GPU dynamics, switching costs, margin baselines |
| `phase2/rl_env.py`, `train_ppo.py` | Gymnasium environment and single-agent PPO (Stable-Baselines3) |
| `phase2/final_eval.py` | Phase 2 test evaluation with paired confidence intervals |
| `phase3/ma_env.py`, `mappo.py`, `train_mappo.py` | Multi-agent environment and MAPPO |
| `phase3/evaluate_mappo.py` | Evaluation at any number of cameras |
| `phase4/live_run.py` | Replays a test scenario with real YOLO on the GPU and measures GPU time |
| `phase4/calibrate_live.py` | Calibrates capacity to the measured live speed and recomputes lateness |
| `phase4/make_figures.py` | All figures from saved results |

## Reproduce

Environment: Python 3.11, PyTorch 2.6 (CUDA 12.4), Ultralytics 8.4, Stable-Baselines3 2.6, Gymnasium 1.1,
LightGBM 4.7. Data: [UA-DETRAC](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig) under
`phase1/DATASET/DETRAC/`. All commands use `EVAL_TARGET=reference`.

```bash
# Setup
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Phase 1: data, profiling (GPU, ~1 h), simulator
cd phase1
python parse_detrac.py && python build_streams.py
python profile_detections.py && python benchmark_costs.py && python summarize_profiles.py
python build_segment_table.py && python evaluate_baselines.py

# Phase 2: single-agent PPO in the dynamic world (~25 min per seed)
cd ../phase2
python train_ppo.py --dynamic --mode margin --timesteps 3000000
python final_eval.py

# Phase 3: MAPPO (~27 min) and test evaluation at 6-48 cameras (~2 h)
cd ../phase3
python train_mappo.py --cameras 4 16 --no-camera-count
python train_mappo.py --critic local
python evaluate_mappo.py --split test --cameras 6 12 24 48

# Phase 4: real-GPU check (~5 min) and figures
cd ../phase4
python live_run.py --device 0
python calibrate_live.py outputs/live/live_steps.csv
python make_figures.py
```

## Limitations

- Main results come from a simulator built on real YOLO outputs and batch-64 GPU costs. The real-GPU check covers
  one 5-minute scenario, with capacity calibrated to measured speed; CPU preprocessing in our Python pipeline,
  not the GPU, limited end-to-end latency.
- One training seed per Phase 3 agent (Phase 2 used three, which agreed closely).
- Larger systems reuse cameras from the same 51 UA-DETRAC streams.
- UA-DETRAC has no two-wheeler labels, so only cars, buses and trucks are counted.

## References

- Jiang et al., *Chameleon: Scalable Adaptation of Video Analytics*, SIGCOMM 2018.
- Zhang et al., *Live Video Analytics at Scale with Approximation and Delay-Tolerance* (VideoStorm), NSDI 2017.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO), NeurIPS 2022.
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Wen et al., *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*, CVIU 2020.
