"""
ECG Annotation Loader

Loads heartbeat annotations from the MIT-BIH Arrhythmia Database.

Author:
Abhinav Kumar

Project:
Arrhythmia Early Prediction using Deep Learning
"""

from pathlib import Path

import pandas as pd

from src.configs.config import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_annotations(record_id: int) -> pd.DataFrame:
    """
    Load annotation file corresponding to an ECG record.

    Parameters
    ----------
    record_id : int
        MIT-BIH record number.

    Returns
    -------
    pandas.DataFrame
        Annotation dataframe.

    Raises
    ------
    FileNotFoundError
        If the annotation file does not exist.
    """

    file_path = RAW_DATA_DIR / f"{record_id}annotations.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {file_path}"
        )

    logger.info("Loading annotations for record %s...", record_id)

    annotations = pd.read_csv(
        file_path,
        sep=r"\s+",
    )

    logger.info(
        "Loaded %d heartbeat annotations.",
        len(annotations),
    )

    return annotations


def get_beat_positions(
    annotations: pd.DataFrame,
) -> pd.Series:
    """
    Extract heartbeat sample locations.

    Parameters
    ----------
    annotations : pandas.DataFrame

    Returns
    -------
    pandas.Series
    """

    return annotations["Sample"]


def get_beat_labels(
    annotations: pd.DataFrame,
) -> pd.Series:
    """
    Extract heartbeat labels.

    Parameters
    ----------
    annotations : pandas.DataFrame

    Returns
    -------
    pandas.Series
    """

    return annotations["#"]


def summarize_annotations(
    annotations: pd.DataFrame,
) -> None:
    """
    Display annotation statistics.

    Parameters
    ----------
    annotations : pandas.DataFrame
    """

    logger.info("Annotation shape : %s", annotations.shape)

    logger.info(
        "Unique beat symbols : %s",
        annotations["#"].unique(),
    )

    logger.info(
        "Beat distribution:\n%s",
        annotations["#"].value_counts(),
    )


def main() -> None:

    record_id = 100

    annotations = load_annotations(record_id)

    summarize_annotations(annotations)

    beat_positions = get_beat_positions(annotations)

    beat_labels = get_beat_labels(annotations)

    logger.info(
        "First five beat positions:\n%s",
        beat_positions.head(),
    )

    logger.info(
        "First five beat labels:\n%s",
        beat_labels.head(),
    )


if __name__ == "__main__":
    main()