"""
ECG Annotation Loader

Loads heartbeat annotations from the MIT-BIH Arrhythmia Database.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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

    annotation_df = pd.read_csv(
        file_path,
        sep=r"\s+",
    )

    positions = annotation_df["Sample"].to_numpy(
        dtype=np.int32
    )

    labels = annotation_df["#"].to_numpy()

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

    summarize_annotations(annotations)


if __name__ == "__main__":
    main()