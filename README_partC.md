# Phase 1, Part C: Environment, scheduler, baselines, oracle

No GPU needed. Everything runs on the saved YOLO results from Part B.

## Files

| File | What it does |
|---|---|
| `build_segment_table.py` | Precomputes, for every 2-second segment and each of the 16 Pareto settings: counting error, the count the system would see, difference from the best setting, and GPU cost. Output: `outputs/env/segment_table.npz` |
| `scheduler.py` | Enforces the GPU budget: lowers settings one step at a time where it hurts accuracy least per millisecond saved. Keeps 10% spare capacity, subtracts backlog, respects per-camera floors |
| `traffic_env.py` | The playground: N cameras play side by side; each 2-second step, a policy picks settings; the scheduler enforces the budget; GPU time varies randomly; a queue tracks backlog, delay and deadline misses; returns error, cost and reward |
| `policies.py` | Baselines (fixed best, max quality, count rule, periodic re-tuning) and the hindsight oracle |
| `evaluate_baselines.py` | Runs every method on the same scenarios at 3 budget levels; saves a table and plot |

## Settings (config.py, Part C section)

| Setting | Default | Meaning |
|---|---|---|
| `N_CAMERAS` | 6 | Cameras per scenario |
| `HORIZON_SEGMENTS` | 150 | Segments per scenario (5 minutes) |
| `BUDGETS_MS_PER_CAM_S` | 3, 7, 15 | GPU ms per camera per second of video (tight, medium, generous) |
| `UTILIZATION_TARGET` | 0.9 | Scheduler plans at most 90% of the budget |
| `COST_JITTER_SIGMA` | 0.15 | Random variation of real GPU time |
| `DEADLINE_MS` | 2000 | Results must be ready within one segment |
| `EVAL_SCENARIOS` | 20 | Scenarios per method and budget |

## Run

```
rlv
python scheduler.py                                       # self-test
python -u build_segment_table.py
python -u evaluate_baselines.py 2>&1 | tee baselines_log.txt
```

`evaluate_baselines.py` uses the VALIDATION split. Run `--split test` only for the final report.

## Outputs (outputs/env/)

| File | Contents |
|---|---|
| `segment_table.npz` | The precomputed table the environment reads |
| `baselines_val.csv` | Error, worst-camera error, GPU used, deadline misses, overrides per method and budget |
| `baselines_val.png` | Error vs compute, one panel per budget |

## Reading the results

- `err`: average counting error over all cameras and scored segments (lower is better)
- `worst_cam_err`: error of the worst camera (fairness)
- `gpu_ms_per_cam_s`: GPU time actually used per camera per second of video
- `err_vs_fixed_%`, `compute_vs_fixed_%`: change compared with the fixed-best baseline
- `deadline_miss_%`: share of segments whose results came late
- `override_%`: share of camera decisions the scheduler had to lower
- The **oracle** cheats (it knows the true errors). Its gap to fixed-best is the most any
  allocation method could gain, and the target for the RL agent.
