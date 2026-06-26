"""
ECG Annotation Loader

Loads heartbeat annotations from the MIT-BIH Arrhythmia Database.
"""

from pathlib import Path

import numpy as np

from src.configs.config import (
    DEFAULT_RECORD,
    RAW_DATA_DIR,
)
from src.utils.logger import get_logger
from src.utils.types import AnnotationData

logger = get_logger(__name__)


def load_annotations(
    record_id: int = DEFAULT_RECORD,
) -> AnnotationData:
    """
    Load heartbeat annotations for a MIT-BIH ECG record.

    Parameters
    ----------
    record_id : int
        MIT-BIH record number.

    Returns
    -------
    AnnotationData
        Heartbeat positions and labels.
    """

    file_path = RAW_DATA_DIR / f"{record_id}annotations.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {file_path}"
        )

    logger.info(
        "Loading annotations for record %d...",
        record_id,
    )

    positions = []
    labels = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        # Skip header
        next(file, None)

        for line_number, line in enumerate(file, start=2):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Expected format:
            # Time Sample Symbol Sub Chan Num Aux
            #
            # Example:
            # 0:00.050 18 N 0 0 0
            #
            # Record 214 also contains:
            # 5:14.703 113293 " 0 0 0 TS
            #
            # Using split() avoids all CSV quoting issues.

            if len(parts) < 3:
                logger.warning(
                    "Skipping malformed line %d in %s",
                    line_number,
                    file_path.name,
                )
                continue

            try:
                sample = int(parts[1])
            except ValueError:
                logger.warning(
                    "Invalid sample index on line %d in %s",
                    line_number,
                    file_path.name,
                )
                continue

            symbol = parts[2]

            positions.append(sample)
            labels.append(symbol)

    positions = np.asarray(
        positions,
        dtype=np.int32,
    )

    labels = np.asarray(
        labels,
        dtype=str,
    )

    logger.info(
        "Loaded %d heartbeat annotations.",
        len(positions),
    )

    return AnnotationData(
        positions=positions,
        labels=labels,
    )


def summarize_annotations(
    annotations: AnnotationData,
) -> None:
    """
    Display annotation statistics.

    Parameters
    ----------
    annotations : AnnotationData
        Annotation information.
    """

    unique_labels, counts = np.unique(
        annotations.labels,
        return_counts=True,
    )

    logger.info("Heartbeat Distribution")

    for label, count in zip(
        unique_labels,
        counts,
    ):
        logger.info(
            "%s : %d",
            label,
            count,
        )


def main() -> None:
    """
    Example usage.
    """

    annotations = load_annotations()

    logger.info(
        "Total beats: %d",
        len(annotations.positions),
    )

    logger.info(
        "First five positions: %s",
        annotations.positions[:5],
    )

    logger.info(
        "First five labels: %s",
        annotations.labels[:5],
    )

    summarize_annotations(
        annotations,
    )


if __name__ == "__main__":
    main()