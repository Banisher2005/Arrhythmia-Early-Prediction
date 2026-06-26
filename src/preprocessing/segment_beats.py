"""
Heartbeat Segmentation

Extract heartbeat-centred ECG segments from filtered ECG signals.
"""

import numpy as np

from src.configs.config import (
    DEFAULT_RECORD,
    WINDOW_BEFORE,
    WINDOW_AFTER,
)

from src.preprocessing.filter_signal import preprocess_signal
from src.preprocessing.load_annotations import load_annotations
from src.preprocessing.load_dataset import load_ecg_signal

from src.utils.logger import get_logger
from src.utils.types import (
    AnnotationData,
    BeatDataset,
)

logger = get_logger(__name__)


def segment_beats(
    signal: np.ndarray,
    annotations: AnnotationData,
    record_id: int,
    window_before: int = WINDOW_BEFORE,
    window_after: int = WINDOW_AFTER,
) -> BeatDataset:
    """
    Segment ECG into heartbeat-centred windows.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.

    annotations : AnnotationData
        Heartbeat annotation information.

    record_id : int
        MIT-BIH record number.

    Returns
    -------
    BeatDataset
        Segmented heartbeat dataset.
    """

    beats = []
    labels = []
    record_ids = []

    window_size = window_before + window_after

    for position, label in zip(
        annotations.positions,
        annotations.labels,
    ):

        start = position - window_before
        end = position + window_after

        if start < 0:
            continue

        if end > len(signal):
            continue

        beat = signal[start:end]

        if beat.shape[0] != window_size:
            continue

        beats.append(beat)
        labels.append(label)
        record_ids.append(record_id)

    dataset = BeatDataset(
        beats=np.asarray(beats, dtype=np.float32),
        labels=np.asarray(labels),
        record_ids=np.asarray(record_ids, dtype=np.int32),
    )

    logger.info(
        "Extracted %d heartbeat segments.",
        len(dataset.beats),
    )

    return dataset


def main() -> None:
    """
    Module test.
    """

    signal = load_ecg_signal(DEFAULT_RECORD)

    signal = preprocess_signal(signal)

    annotations = load_annotations(DEFAULT_RECORD)

    dataset = segment_beats(
        signal=signal,
        annotations=annotations,
        record_id=DEFAULT_RECORD,
    )

    logger.info(
        "Beat dataset shape: %s",
        dataset.beats.shape,
    )

    logger.info(
        "Labels shape: %s",
        dataset.labels.shape,
    )

    logger.info(
        "Record IDs shape: %s",
        dataset.record_ids.shape,
    )


if __name__ == "__main__":
    main()