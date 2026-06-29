"""Configuration for the Streamlit ECG application."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


STREAMLIT_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMLIT_APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import (  # noqa: E402
    BEST_MODEL_NAME,
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SAMPLING_RATE,
    SEGMENT_LENGTH,
)


@dataclass(frozen=True)
class StreamlitSettings:
    """Default runtime settings for the clinical demo app."""

    device: str = "Auto"
    confidence_threshold: float = 0.50
    batch_size: int = 32
    theme: str = "System"


APP_TITLE = "AMSRAN-GF ECG Analysis Workbench"
APP_SUBTITLE = "Research-grade arrhythmia classification and explainability"
SUPPORTED_UPLOAD_TYPES = ("csv", "txt", "dat", "npy")
DEFAULT_SETTINGS = StreamlitSettings()
MODEL_PATH = MODELS_DIR / BEST_MODEL_NAME
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"
EXPLAINABILITY_OUTPUT_DIR = RESULTS_DIR / "explainability" / "streamlit_report"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "mitbih_dataset.npz"
BENCHMARK_PATHS = (
    RESULTS_DIR / "explainability" / "benchmark" / "benchmark.json",
    RESULTS_DIR / "explainability_benchmark.json",
    RESULTS_DIR / "explainability" / "benchmark.json",
)
