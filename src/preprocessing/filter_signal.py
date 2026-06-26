"""
ECG Signal Filtering

Applies band-pass filtering and normalization to ECG signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from src.configs.config import (
    SAMPLING_RATE,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def butter_bandpass(
    lowcut: float = 0.5,
    highcut: float = 40.0,
    sampling_rate: int = SAMPLING_RATE,
    order: int = 4,
):
    """
    Create Butterworth band-pass filter coefficients.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency.

    highcut : float
        Upper cutoff frequency.

    sampling_rate : int
        ECG sampling frequency.

    order : int
        Filter order.

    Returns
    -------
    tuple
        Filter coefficients (b, a).
    """

    nyquist = 0.5 * sampling_rate

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="bandpass",
    )

    return b, a


def filter_ecg(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> np.ndarray:
    """
    Apply Butterworth band-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal.

    Returns
    -------
    np.ndarray
        Filtered ECG signal.
    """

    b, a = butter_bandpass(
        lowcut=lowcut,
        highcut=highcut,
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal,
    )

    logger.info("Signal filtering completed.")

    return filtered_signal.astype(np.float32)


def normalize_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Normalize ECG signal to the range [0, 1].

    Parameters
    ----------
    signal : np.ndarray
        ECG signal.

    Returns
    -------
    np.ndarray
        Normalized ECG signal.
    """

    minimum = np.min(signal)
    maximum = np.max(signal)

    normalized_signal = (
        signal - minimum
    ) / (
        maximum - minimum
    )

    logger.info("Signal normalization completed.")

    return normalized_signal.astype(np.float32)


def preprocess_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal.

    Returns
    -------
    np.ndarray
        Filtered and normalized ECG.
    """

    filtered_signal = filter_ecg(signal)

    normalized_signal = normalize_signal(
        filtered_signal
    )

    return normalized_signal


def main() -> None:
    """
    Module test.
    """

    from src.preprocessing.load_dataset import load_ecg_signal

    signal = load_ecg_signal()

    processed_signal = preprocess_signal(signal)

    logger.info(
        "Processed signal shape: %s",
        processed_signal.shape,
    )


if __name__ == "__main__":
    main()