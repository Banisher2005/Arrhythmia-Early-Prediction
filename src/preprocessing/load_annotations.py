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


def parse_annotation_lines(
    lines,
) -> AnnotationData:
    """
    Parse MIT-BIH annotation text lines into sample positions and labels.
    """

    positions = []
    labels = []

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 3:
            logger.warning(
                "Skipping malformed annotation line %d",
                line_number,
            )
            continue

        try:
            sample = int(parts[1])
        except ValueError:
            if line_number == 1:
                continue
            logger.warning(
                "Invalid sample index on annotation line %d",
                line_number,
            )
            continue

        positions.append(sample)
        labels.append(parts[2])

    return AnnotationData(
        positions=np.asarray(positions, dtype=np.int32),
        labels=np.asarray(labels, dtype=str),
    )


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

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        annotations = parse_annotation_lines(file)

    logger.info(
        "Loaded %d heartbeat annotations.",
        len(annotations.positions),
    )

    return annotations


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
