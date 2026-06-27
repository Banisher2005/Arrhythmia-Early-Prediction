"""
Project Configuration

Central configuration for the Arrhythmia Early Prediction project.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR = PROJECT_ROOT / "models_saved"

GRAPHS_DIR = PROJECT_ROOT / "graphs"

# ==========================================================
# MIT-BIH Dataset
# ==========================================================

DEFAULT_RECORD = 100

MIT_BIH_RECORDS = [
    100, 101, 102, 103, 104,
    105, 106, 107, 108, 109,
    111, 112, 113, 114, 115,
    116, 117, 118, 119, 121,
    122, 123, 124, 200, 201,
    202, 203, 205, 207, 208,
    209, 210, 212, 213, 214,
    215, 217, 219, 220, 221,
    222, 223, 228, 230, 231,
    232, 233, 234,
]

SAMPLING_RATE = 360

WINDOW_BEFORE = 90

WINDOW_AFTER = 90

SEGMENT_LENGTH = WINDOW_BEFORE + WINDOW_AFTER

NUM_CLASSES = 5

CLASS_NAMES = (
    "N",
    "S",
    "V",
    "F",
    "Q",
)

# ==========================================================
# Dataset Split
# ==========================================================

RANDOM_SEED = 42

VALIDATION_RATIO = 0.20

# ==========================================================
# Training
# ==========================================================

BATCH_SIZE = 256

NUM_WORKERS = 4

NUM_EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PIN_MEMORY = True

DROP_LAST = False

# ==========================================================
# Model
# ==========================================================

INPUT_CHANNELS = 1

INPUT_LENGTH = SEGMENT_LENGTH

# ==========================================================
# Checkpointing
# ==========================================================

BEST_MODEL_NAME = "best_model.pt"

LAST_MODEL_NAME = "last_model.pth"

CHECKPOINT_DIR = MODELS_DIR

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"