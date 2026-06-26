"""
Dataset Creation

Creates and saves the processed heartbeat dataset from the
MIT-BIH Arrhythmia Database.

Author:
Abhinav Kumar

Project:
Arrhythmia Early Prediction using Deep Learning
"""

import numpy as np

from src.configs.config import (
    DEFAULT_RECORD,
    PROCESSED_DATA_DIR,
)

from src.preprocessing.filter_signal import preprocess_signal
from src.preprocessing.load_annotations import load_annotations
from src.preprocessing.load_dataset import load_ecg_signal
from src.preprocessing.segment_beats import segment_beats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_dataset(
    beats: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray,
) -> None:
    """
    Save processed dataset.

    Parameters
    ----------
    beats : np.ndarray
        Segmented heartbeat signals.

    labels : np.ndarray
        Beat labels.

    record_ids : np.ndarray
        Record IDs corresponding to each beat.
    """

    np.save(
        PROCESSED_DATA_DIR / "beats.npy",
        beats,
    )

    np.save(
        PROCESSED_DATA_DIR / "labels.npy",
        labels,
    )

    np.save(
        PROCESSED_DATA_DIR / "record_ids.npy",
        record_ids,
    )

    logger.info(
        "Dataset saved successfully."
    )

    logger.info(
        "Saved to: %s",
        PROCESSED_DATA_DIR,
    )


def main() -> None:
    """
    Create dataset for a single ECG record.
    """

    logger.info("Creating dataset...")

    signal = load_ecg_signal(DEFAULT_RECORD)

    signal = preprocess_signal(signal)

    annotations = load_annotations(DEFAULT_RECORD)

    dataset = segment_beats(
        signal=signal,
        annotations=annotations,
        record_id=DEFAULT_RECORD,
    )

    save_dataset(
        dataset.beats,
        dataset.labels,
        dataset.record_ids,
    )

    logger.info(
        "Dataset Shape : %s",
        dataset.beats.shape,
    )

    logger.info(
        "Labels Shape : %s",
        dataset.labels.shape,
    )


if __name__ == "__main__":
    main()