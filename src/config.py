import os

# --- General ---
SEED = 42
DEVICE = "cuda"  # "cuda" or "cpu" (auto-switch happens in code)
SAVE_DIR = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Environment ---
GRID_SIZE = 20          # cells per side
CELL_PIXELS = 20        # pixel per cell (for rendering)
SNAKE_INIT_LEN = 3
RENDER_TRAIN = False
FPS_TRAIN = 60
FPS_PLAY = 15

# --- DQN Hyperparams ---
STATE_SIZE = 11         # fixed by env's get_state()
ACTION_SIZE = 3         # straight, left, right
HIDDEN_SIZE = 256
LR = 1e-3
GAMMA = 0.99

# Epsilon-greedy
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100_000  # linear decay steps

# Replay buffer + training schedule
BUFFER_CAPACITY = 100_000
BATCH_SIZE = 512
LEARN_START = 10_000       # start updates after enough experience
TARGET_UPDATE_EVERY = 1_000
TRAIN_STEPS_PER_ENV_STEP = 1

# Episodes / steps
MAX_EPISODES = 1_000
MAX_STEPS_PER_EPISODE = GRID_SIZE * GRID_SIZE * 4  # safety cap

# Rewards
REWARD_FOOD = 10.0
REWARD_DEAD = -10.0
REWARD_STEP = -0.01       # small penalty to encourage shorter paths
REWARD_TOWARD_FOOD = 0.05 # small reward if we moved closer to food

# Checkpointing
SAVE_EVERY_EPISODES = 25
BEST_MODEL_FILENAME = "best_model.pth"
LAST_MODEL_FILENAME = "last_model.pth"
