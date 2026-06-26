"""
Official MIT-BIH Inter-Patient Dataset Splitter

Implements the de Chazal DS1 / DS2 protocol.

DS1 -> Train + Validation
DS2 -> Test

This protocol prevents patient leakage and is widely adopted in
ECG heartbeat classification literature.
"""

from pathlib import Path

import numpy as np

from src.configs.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

from src.configs.config import (
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    VALIDATION_RATIO,
)

# ----------------------------------------------------------
# Official DS1 (Training Pool)
# ----------------------------------------------------------

DS1_RECORDS = np.array(
    [
        101,
        106,
        108,
        109,
        112,
        114,
        115,
        116,
        118,
        119,
        122,
        124,
        201,
        203,
        205,
        207,
        208,
        209,
        215,
        220,
        223,
        230,
    ],
    dtype=np.int32,
)

# ----------------------------------------------------------
# Official DS2 (Testing)
# ----------------------------------------------------------

DS2_RECORDS = np.array(
    [
        100,
        103,
        105,
        111,
        113,
        117,
        121,
        123,
        200,
        202,
        210,
        212,
        213,
        214,
        219,
        221,
        222,
        228,
        231,
        232,
        233,
        234,
    ],
    dtype=np.int32,
)


def split_dataset() -> None:
    """
    Create the official inter-patient split.

    DS1:
        Train + Validation

    DS2:
        Test
    """

    dataset_path = PROCESSED_DATA_DIR / "mitbih_dataset.npz"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    logger.info("Loading processed dataset...")

    data = np.load(
        dataset_path,
        allow_pickle=True,
    )

    record_ids = data["record_ids"]

    dataset_records = set(np.unique(record_ids).tolist())

    expected_records = set(
        DS1_RECORDS.tolist() + DS2_RECORDS.tolist()
    )

    missing = expected_records - dataset_records

    if missing:
        raise ValueError(
            f"Missing records in dataset: {sorted(missing)}"
        )

    rng = np.random.default_rng(RANDOM_SEED)

    shuffled_ds1 = DS1_RECORDS.copy()
    rng.shuffle(shuffled_ds1)

    validation_size = int(
        len(shuffled_ds1) * VALIDATION_RATIO
    )

    validation_records = np.sort(
        shuffled_ds1[:validation_size]
    )

    training_records = np.sort(
        shuffled_ds1[validation_size:]
    )

    testing_records = np.sort(
        DS2_RECORDS
    )

    train_mask = np.isin(
        record_ids,
        training_records,
    )

    validation_mask = np.isin(
        record_ids,
        validation_records,
    )

    test_mask = np.isin(
        record_ids,
        testing_records,
    )

    train_indices = np.where(train_mask)[0]
    validation_indices = np.where(validation_mask)[0]
    test_indices = np.where(test_mask)[0]

    output_path = (
        PROCESSED_DATA_DIR /
        "patient_split.npz"
    )

    np.savez_compressed(
        output_path,
        train_indices=train_indices,
        valid_indices=validation_indices,
        test_indices=test_indices,
        train_records=training_records,
        valid_records=validation_records,
        test_records=testing_records,
    )

    logger.info("=" * 70)
    logger.info("Official DS1 / DS2 split created")
    logger.info("=" * 70)

    logger.info(
        "Training Records (%d): %s",
        len(training_records),
        training_records.tolist(),
    )

    logger.info(
        "Validation Records (%d): %s",
        len(validation_records),
        validation_records.tolist(),
    )

    logger.info(
        "Testing Records (%d): %s",
        len(testing_records),
        testing_records.tolist(),
    )

    logger.info("")

    logger.info(
        "Training Beats   : %d",
        len(train_indices),
    )

    logger.info(
        "Validation Beats : %d",
        len(validation_indices),
    )

    logger.info(
        "Testing Beats    : %d",
        len(test_indices),
    )

    logger.info("")

    logger.info(
        "Saved split to %s",
        output_path,
    )


def main() -> None:
    """
    Entry point.
    """

    split_dataset()


if __name__ == "__main__":
    main()