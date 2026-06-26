"""
Dataset Creation

Creates the complete MIT-BIH heartbeat dataset and stores it
as a single compressed dataset.

Author:
Abhinav Kumar

Project:
Arrhythmia Early Prediction using Deep Learning
"""

import numpy as np

from src.configs.config import (
    MIT_BIH_RECORDS,
    PROCESSED_DATA_DIR,
    SAMPLING_RATE,
    WINDOW_AFTER,
    WINDOW_BEFORE,
)

from src.preprocessing.filter_signal import preprocess_signal
from src.preprocessing.load_annotations import load_annotations
from src.preprocessing.load_dataset import load_ecg_signal

from src.utils.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------
# AAMI Mapping
# -------------------------------------------------------------

AAMI_MAPPING = {

    # Normal

    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # Supraventricular

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # Ventricular

    "V": "V",
    "E": "V",

    # Fusion

    "F": "F",

    # Unknown

    "/": "Q",
    "f": "Q",
    "Q": "Q",

}


CLASS_TO_INT = {

    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,

}


def segment_record(signal, annotations):

    beats = []
    labels = []
    encoded = []
    positions = []

    skipped = 0

    for sample, label in zip(
        annotations.positions,
        annotations.labels,
    ):

        if label not in AAMI_MAPPING:
            skipped += 1
            continue

        start = sample - WINDOW_BEFORE
        end = sample + WINDOW_AFTER

        if start < 0:
            continue

        if end > len(signal):
            continue

        beat = signal[start:end]

        if len(beat) != WINDOW_BEFORE + WINDOW_AFTER:
            continue

        beats.append(beat)

        labels.append(AAMI_MAPPING[label])

        encoded.append(
            CLASS_TO_INT[
                AAMI_MAPPING[label]
            ]
        )

        positions.append(sample)

    return (

        np.asarray(beats, dtype=np.float32),

        np.asarray(labels),

        np.asarray(encoded, dtype=np.int64),

        np.asarray(positions, dtype=np.int32),

        skipped,

    )


def main():

    logger.info("=" * 70)
    logger.info("Creating Complete MIT-BIH Dataset")
    logger.info("=" * 70)

    all_beats = []
    all_labels = []
    all_encoded = []
    all_records = []
    all_positions = []

    skipped_total = 0

    processed_records = 0

    for record in MIT_BIH_RECORDS:

        logger.info(
            "Processing Record %d",
            record,
        )

        signal = load_ecg_signal(record)

        signal = preprocess_signal(signal)

        annotations = load_annotations(record)

        beats, labels, encoded, positions, skipped = (
            segment_record(
                signal,
                annotations,
            )
        )

        all_beats.append(beats)

        all_labels.append(labels)

        all_encoded.append(encoded)

        all_positions.append(positions)

        all_records.append(

            np.full(

                len(beats),

                record,

                dtype=np.int32,

            )

        )

        skipped_total += skipped

        processed_records += 1

        logger.info(
            "Extracted %d beats",
            len(beats),
        )

    beats = np.concatenate(all_beats)

    labels = np.concatenate(all_labels)

    encoded = np.concatenate(all_encoded)

    record_ids = np.concatenate(all_records)

    sample_positions = np.concatenate(
        all_positions
    )

    metadata = {

        "dataset": "MIT-BIH Arrhythmia",

        "sampling_rate": SAMPLING_RATE,

        "window_before": WINDOW_BEFORE,

        "window_after": WINDOW_AFTER,

        "segment_length":
        WINDOW_BEFORE + WINDOW_AFTER,

        "records_used":
        MIT_BIH_RECORDS,

        "total_records":
        processed_records,

        "total_beats":
        len(beats),

        "classes":
        ["N", "S", "V", "F", "Q"],

    }

    np.savez_compressed(

        PROCESSED_DATA_DIR / "mitbih_dataset.npz",

        beats=beats,

        labels=labels,

        encoded_labels=encoded,

        record_ids=record_ids,

        sample_positions=sample_positions,

        metadata=metadata,

    )

    logger.info("=" * 70)

    logger.info("Dataset Created Successfully")

    logger.info("=" * 70)

    logger.info(
        "Processed Records : %d",
        processed_records,
    )

    logger.info(
        "Total Beats : %d",
        len(beats),
    )

    logger.info(
        "Skipped Labels : %d",
        skipped_total,
    )

    logger.info(
        "Saved to %s",
        PROCESSED_DATA_DIR /
        "mitbih_dataset.npz",
    )


if __name__ == "__main__":
    main()