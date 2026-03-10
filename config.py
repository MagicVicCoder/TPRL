import torch
import os
from datetime import datetime

# --- Global Settings ---
# Use a cache directory within your project or home directory
os.environ["HF_HOME"] = os.path.expanduser("~/MagicVic/data/huggingface_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# --- MLLM and Dataset Configuration ---
# Supported models:
# - "Qwen/Qwen2.5-VL-3B-Instruct"
# - "llava-hf/llava-1.5-7b-hf"
# - "llava-hf/llava-1.5-13b-hf"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# ScreenSpot-Pro bbox 任务
DATASET_NAME = "Voxel51/ScreenSpot-Pro"
DATASET_SPLIT = "train"  # ScreenSpot-Pro 通常只提供 train/validation 等划分
TRAIN_TEST_SPLIT_RATIO = 0.8

# --- Pruning Settings ---
PRUNING_TARGET_RATIO = 0.5  # Target ratio of tokens to keep after pruning (e.g., 0.5 means prune 50%)

# --- MLP Pruner Settings ---
PRUNING_MLP_HIDDEN_DIM = 256 # Hidden dimension for the MLP used to predict pruning scores
PRUNING_MLP_DROPOUT = 0.1   # Dropout rate for the MLP
PRUNING_LEARNING_RATE = 1e-4 # Learning rate for training the MLP pruner (if applicable)
PRUNING_NUM_EPOCHS = 10      # Number of epochs to train the pruner (if applicable)
PRUNING_BATCH_SIZE = 4       # Batch size for processing samples during pruning/evaluation

# --- Evaluation Settings ---
EVAL_MODE = "full"  # 可选值："full", "budget", "none"
EVAL_BATCH_SIZE = 4 # Batch size for evaluation
BBOX_SUCCESS_IOU = 0.5  # IoU threshold for considering bbox prediction correct

# --- RL Pruner Settings ---
USE_AUTOENCODER = False  # Whether to use autoencoder for state compression
RL_LATENT_DIM = 256  # Latent dimension for autoencoder (if used)
RL_NHEAD = 8  # Number of attention heads
RL_NUM_LAYERS = 2  # Number of transformer layers
RL_HIDDEN_DIM = 512  # Hidden dimension for MLP heads
RL_DROPOUT = 0.1  # Dropout rate
RL_THRESHOLD = 0.5  # Retention probability threshold for inference
RL_STEP_DISCOUNT = 0.5  # Step-wise discount factor (lambda_disc)

# --- Autoencoder Training Settings ---
AE_NUM_EPOCHS = 10
AE_LEARNING_RATE = 1e-4
AE_WEIGHT_DECAY = 0.01
AE_BATCH_SIZE = 256

# --- Learning from Demonstrations (LfD) Settings ---
LFD_NUM_DEMOS = 50000  # Number of demonstrations to generate
LFD_NUM_STEPS = 3  # Number of pruning steps per trajectory
LFD_NUM_EPOCHS = 5  # Number of training epochs
LFD_LEARNING_RATE = 5e-5  # Learning rate
LFD_BATCH_SIZE = 128  # Batch size
LFD_CHECKPOINT_PATH = None  # Path to load LfD checkpoint (set after training)

# --- PPO Training Settings ---
PPO_NUM_EPOCHS = 200  # Number of training epochs
PPO_ROLLOUT_BATCH_SIZE = 64  # Number of rollouts per epoch
PPO_MINI_BATCH_SIZE = 16  # Mini-batch size for PPO updates
PPO_MAX_STEPS = 3  # Maximum pruning steps per rollout
PPO_LR_ACTOR = 1e-5  # Learning rate for actor (policy)
PPO_LR_CRITIC = 3e-5  # Learning rate for critic (value)
PPO_GAMMA = 0.99  # Discount factor
PPO_LAM = 0.95  # GAE lambda
PPO_CLIP_EPSILON = 0.2  # PPO clip ratio
PPO_VALUE_COEF = 0.5  # Value loss coefficient (c1)
PPO_ENTROPY_COEF = 0.01  # Entropy coefficient (c2)
PPO_EPOCHS = 4  # Number of PPO update epochs per rollout
PPO_REWARD_ALPHA = 1.0  # Task reward weight
PPO_REWARD_BETA = 0.1  # Efficiency reward weight
