"""
ECG Dataset Loader

Loads ECG recordings from the MIT-BIH Arrhythmia Database.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.configs.config import (
    DEFAULT_RECORD,
    ECG_LEAD,
    RAW_DATA_DIR,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_ecg_dataframe(record_id: int = DEFAULT_RECORD) -> pd.DataFrame:
    """
    Load a MIT-BIH ECG record as a pandas DataFrame.

    Parameters
    ----------
    record_id : int
        MIT-BIH record number.

    Returns
    -------
    pandas.DataFrame
        ECG recording.
    """

    file_path = RAW_DATA_DIR / f"{record_id}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"ECG record not found: {file_path}"
        )

    logger.info("Loading ECG record %d...", record_id)

    ecg = pd.read_csv(file_path)

    ecg.columns = (
        ecg.columns
        .str.replace("'", "", regex=False)
        .str.strip()
    )

    logger.info(
        "Loaded ECG record %d (%d samples).",
        record_id,
        len(ecg),
    )

    return ecg


def load_ecg_signal(
    record_id: int = DEFAULT_RECORD,
    lead: str = ECG_LEAD,
) -> np.ndarray:
    """
    Load an ECG signal.

    If the requested lead is unavailable, automatically use
    the first available ECG lead.
    """

    ecg = load_ecg_dataframe(record_id)

    if lead not in ecg.columns:

        logger.warning(
            "Lead '%s' not found for record %d. Using '%s' instead.",
            lead,
            record_id,
            ecg.columns[1],
        )

        lead = ecg.columns[1]

    signal = ecg[lead].to_numpy(dtype=np.float32)

    logger.info(
        "Loaded lead '%s' with %d samples.",
        lead,
        len(signal),
    )

    return signal


def main() -> None:
    """
    Example usage.
    """

    signal = load_ecg_signal()

    logger.info(
        "Signal shape: %s",
        signal.shape,
    )


if __name__ == "__main__":
    main()