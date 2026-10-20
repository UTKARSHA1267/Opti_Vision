"""Phase 2 settings (Phase 1 settings stay in phase1/config.py)."""

# A probe runs the camera's current model/resolution AND the reference setting on this many frames
# of the segment, and compares their counts.
PROBE_FRAMES = 1

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