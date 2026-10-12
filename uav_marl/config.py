"""
Configuration for Multi-UAV Cooperative Path Planning experiments.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Environment ──
N_AGENTS = 3               # Number of UAVs
STATE_DIM = 6              # per agent: x, y, vx, vy (2D) + relative goal
ACTION_DIM = 2             # per agent: ax, ay
OBS_DIM = STATE_DIM + 4    # own state + 2 nearest obstacles (rel pos + size)

MAP_SIZE = 500.0           # m × m
N_OBSTACLES = 8            # Number of obstacles
OBSTACLE_RADIUS = 25.0     # m
D_SAFE = 20.0              # Minimum safe distance (m)
D_COMM = 200.0             # Communication range (m)
V_MAX = 15.0               # Max velocity (m/s)
A_MAX = 5.0                # Max acceleration (m/s²)
DT = 0.5                   # Time step (s)
MAX_EPISODE_STEPS = 200

# ── Reward weights ──
W_GOAL = 1.0
W_OBS = 10.0
W_FORM = 0.5
W_ENERGY = 0.01
W_COMM = 2.0
W_STEP = 0.01              # Small penalty per step

# ── Training ──
GAMMA = 0.99
TAU = 0.005                # Soft update
BATCH_SIZE = 128           # Reduced for CPU speed
REPLAY_SIZE = int(5e5)
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4          # Must match actor_lr to prevent critic overfitting
MAX_EPISODES = 1500        # Enough for Zone-4 paper
EXPLORE_NOISE = 0.1
TARGET_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2
EVAL_INTERVAL = 100
WARMUP_STEPS = 5000        # Collect 5k steps (~8 eps) before training starts

# ── Algorithm-specific ──
BETA_MUAV_MTD3 = 0.5       # Mutual estimation weight
BETA_SOFT_MUAV_STD3 = 5.0  # Softmax temperature (higher = closer to max)
K_SAMPLES_MUAV_STD3 = 5   # Action samples for softmax (more samples = better coverage)

# ── Network ──
HIDDEN_DIM = 128            # Halved for CPU speed
