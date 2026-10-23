"""Phase 3 settings (MAPPO)."""

# Each camera agent picks an importance level (-> weight exp(RL_WEIGHT_SCALE * level) on its errors)
IMPORTANCE_LEVELS = [-1.0, 0.0, 1.0]

# Training
MAPPO_TIMESTEPS = 2_000_000      # environment steps (each step = all cameras decide once)
MAPPO_N_ENVS = 8                 # parallel environments
MAPPO_EPOCHS = 10
MAPPO_MINIBATCHES = 4
MAPPO_LR = 3e-4
MAPPO_GAMMA = 0.95
MAPPO_LAMBDA = 0.95
MAPPO_CLIP = 0.2
MAPPO_ENTROPY = 0.01
MAPPO_HIDDEN = 128
MAPPO_REWARD_SCALE = 0.1         # rewards are multiplied by this for training stability
MAPPO_EVAL_TIMES = 10            # quick validation checks during training

# Number of cameras per training episode (inclusive range). Part A: fixed 6. Part C (scaling): e.g. 4-8.
MAPPO_CAMERA_RANGE = (6, 6)
