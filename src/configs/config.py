"""
Project Configuration

Central configuration for the Arrhythmia Early Prediction project.
"""

from pathlib import Path

# ==============================================================================
# Project Directories
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
REPORTS_DIR = RESULTS_DIR / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models_saved"

# ==============================================================================
# Dataset Configuration
# ==============================================================================

DEFAULT_RECORD = 100

ECG_LEAD = "MLII"

SAMPLING_RATE = 360

WINDOW_BEFORE = 90
WINDOW_AFTER = 90

SEGMENT_LENGTH = WINDOW_BEFORE + WINDOW_AFTER

# ==============================================================================
# Model Configuration
# ==============================================================================

RANDOM_STATE = 42

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==============================================================================
# Create Required Directories
# ==============================================================================

for directory in (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    GRAPHS_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)