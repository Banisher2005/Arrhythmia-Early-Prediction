"""
ECG Dataset Loader

Loads ECG recordings from the MIT-BIH Arrhythmia Database.

Author:
Abhinav Kumar

Project:
Arrhythmia Early Prediction using Deep Learning
"""

from pathlib import Path

import pandas as pd

from src.configs.config import ECG_LEAD, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_ecg_record(record_id: int) -> pd.DataFrame:
    """
    Load a MIT-BIH ECG record.

    Parameters
    ----------
    record_id : int
        MIT-BIH record number (e.g., 100, 101, 102).

    Returns
    -------
    pandas.DataFrame
        ECG recording containing all available leads.

    Raises
    ------
    FileNotFoundError
        If the requested ECG record does not exist.
    """

    file_path = RAW_DATA_DIR / f"{record_id}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"ECG record not found: {file_path}"
        )

    logger.info("Loading ECG record %s...", record_id)

    ecg = pd.read_csv(file_path)

    # Remove unwanted quotes and spaces from column names
    ecg.columns = (
        ecg.columns
        .str.replace("'", "", regex=False)
        .str.strip()
    )

    logger.info(
        "ECG record %s loaded successfully (%d samples).",
        record_id,
        len(ecg),
    )

    return ecg


def get_ecg_signal(
    ecg: pd.DataFrame,
    lead: str = ECG_LEAD
) -> pd.Series:
    """
    Extract a specific ECG lead.

    Parameters
    ----------
    ecg : pandas.DataFrame
        Loaded ECG recording.

    lead : str
        ECG lead to extract.

    Returns
    -------
    pandas.Series
        ECG signal.
    """

    if lead not in ecg.columns:
        raise ValueError(
            f"Lead '{lead}' not found in ECG recording."
        )

    logger.info("Using ECG lead: %s", lead)

    return ecg[lead]


def main() -> None:
    """
    Example usage.
    """

    record_id = 100

    ecg = load_ecg_record(record_id)

    signal = get_ecg_signal(ecg)

    logger.info("Dataset shape: %s", ecg.shape)

    logger.info("Signal length: %d", len(signal))


if __name__ == "__main__":
    main()