"""
Dataset Split Verification

Verifies the official patient-wise dataset split.
"""

import numpy as np

from src.configs.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def verify_split() -> None:
    """
    Verify dataset split integrity.
    """

    dataset = np.load(
        PROCESSED_DATA_DIR / "mitbih_dataset.npz",
        allow_pickle=True,
    )

    split = np.load(
        PROCESSED_DATA_DIR / "patient_split.npz",
        allow_pickle=True,
    )

    train_indices = split["train_indices"]
    valid_indices = split["valid_indices"]
    test_indices = split["test_indices"]

    train_records = set(split["train_records"].tolist())
    valid_records = set(split["valid_records"].tolist())
    test_records = set(split["test_records"].tolist())

    logger.info("=" * 70)
    logger.info("Verifying Dataset Split")
    logger.info("=" * 70)

    # ------------------------------------------------------
    # Verify no patient leakage
    # ------------------------------------------------------

    assert train_records.isdisjoint(valid_records)
    assert train_records.isdisjoint(test_records)
    assert valid_records.isdisjoint(test_records)

    logger.info("✓ No patient leakage.")

    # ------------------------------------------------------
    # Verify no duplicated samples
    # ------------------------------------------------------

    combined_indices = np.concatenate(
        (
            train_indices,
            valid_indices,
            test_indices,
        )
    )

    assert len(np.unique(combined_indices)) == len(combined_indices)

    logger.info("✓ No duplicated samples.")

    # ------------------------------------------------------
    # Verify all selected samples belong to valid records
    # ------------------------------------------------------

    record_ids = dataset["record_ids"][combined_indices]

    valid_records_set = (
        train_records
        | valid_records
        | test_records
    )

    invalid_records = set(record_ids.tolist()) - valid_records_set

    assert len(invalid_records) == 0

    logger.info("✓ All selected samples belong to valid records.")

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    logger.info("")

    logger.info(
        "Train Samples : %d",
        len(train_indices),
    )

    logger.info(
        "Valid Samples : %d",
        len(valid_indices),
    )

    logger.info(
        "Test Samples  : %d",
        len(test_indices),
    )

    logger.info(
        "Selected Samples : %d",
        len(combined_indices),
    )

    logger.info(
        "Dataset Samples  : %d",
        len(dataset["beats"]),
    )

    logger.info("")

    excluded = len(dataset["beats"]) - len(combined_indices)

    logger.info(
        "Excluded Samples (Official DS1/DS2): %d",
        excluded,
    )

    logger.info("=" * 70)
    logger.info("Verification Successful")
    logger.info("=" * 70)


def main() -> None:
    """
    Entry point.
    """

    verify_split()


if __name__ == "__main__":
    main()