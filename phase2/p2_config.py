"""Phase 2 settings (Phase 1 settings stay in phase1/config.py)."""

# A probe runs the camera's current model/resolution AND the reference setting on this many frames
# of the segment, and compares their counts.
PROBE_FRAMES = 3

# The estimator uses a probe result for at most this many segments after it was taken.
PROBE_MAX_AGE = 6

# The estimator predicts the error averaged over the next AHEAD segments (less 2-second luck).
AHEAD = 3

# Image clues are expressed relative to the camera's own previous REL_WINDOW segments,
# so the estimator cannot just recognise which camera it is looking at.
REL_WINDOW = 5

# Training data: information states sampled per target segment, and the share with no probe yet.
SAMPLES_PER_SEGMENT = 8
NO_PROBE_SHARE = 0.2
NO_HISTORY_SHARE = 0.05

# Probing policies evaluated: average number of probes per segment (0.25 = one probe every 4 segments).
PROBE_RATES = [0, 0.25, 0.5, 1, 2]

# ---- Estimator v3 (simple, probe-based) ----
# Probes always compare the CHEAPEST setting's model/resolution with the reference.
# The estimator averages a camera's probes over the last LONG_WINDOW segments (60 s) for its long-term term.
LONG_WINDOW = 30
# Probe "fading": weight of a probe that is `age` segments old = exp(-(age - 1) / tau); tau chosen from:
FADE_TAUS = [1, 2, 3, 5, 8, 13]
# Probing intervals (in segments) used to simulate realistic probe histories for training.
TRAIN_PROBE_INTERVALS = [1, 2, 4, 8, 16]

# ---- RL (Part E) ----
# Budgets are drawn uniformly from this range for every training episode (ms per camera-second).
RL_BUDGET_RANGE = (3.0, 15.0)
# Importance action a in [-1, 1] becomes a weight exp(RL_WEIGHT_SCALE * a) on that camera's errors.
RL_WEIGHT_SCALE = 1.5
# The reserve action r in [-1, 1] holds back max(r, 0) x this share of the budget (r <= 0: no reserve).
RL_MAX_RESERVE = 0.6
# Reward = -(error / mean prior error) - RL_DEADLINE_PENALTY * (deadline missed)
RL_DEADLINE_PENALTY = 1.0
# Length of the rolling count average in the observation (segments).
RL_COUNT_WINDOW = 5
# PPO training
RL_TIMESTEPS = 300_000
RL_N_ENVS = 8

# ---- Dynamic environment (shared GPU, switching costs) ----
# GPU availability switches between full and DYN_LOW_SHARE of the budget (another job using the GPU).
DYN_LOW_SHARE = 0.5
# Probability per segment that availability switches (1/15 = on average every 30 s).
DYN_SWITCH_PROB = 1 / 15
# Small random variation of availability (log-normal sigma).
DYN_NOISE = 0.05
# Changing a camera's model/resolution costs this many extra frames of the new model (warm-up).
SWITCH_COST_FRAMES = 2
# Safety margin of the "budget sharing with margin" baseline.
SAFETY_MARGIN = 0.2
# Margins evaluated for the fixed-margin baseline, and for the state-aware rule (reserve only at full capacity).
SAFETY_MARGINS = [0.1, 0.2, 0.3, 0.4, 0.5]
STATE_AWARE_MARGINS = [0.3, 0.5]

# ---- RL margin controller (action = one safety margin per segment) ----
MARGIN_CHOICES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# Training scenarios used to tune the fixed margin per budget (the "tuned margin" baseline).
TUNE_SCENARIOS = 12
