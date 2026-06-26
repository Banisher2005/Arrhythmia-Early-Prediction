"""
ECG Signal Filtering

Applies signal preprocessing techniques to ECG recordings
from the MIT-BIH Arrhythmia Database.

Author:
Abhinav Kumar

Project:
Arrhythmia Early Prediction using Deep Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from src.configs.config import (
    SAMPLING_RATE,
    ECG_LEAD,
)

from src.preprocessing.load_dataset import (
    load_ecg_record,
    get_ecg_signal,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def butter_bandpass(
    lowcut: float,
    highcut: float,
    sampling_rate: int,
    order: int = 4,
):
    """
    Create a Butterworth bandpass filter.
    """

    nyquist = 0.5 * sampling_rate

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band",
    )

    return b, a


def filter_ecg(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filtering.
    """

    b, a = butter_bandpass(
        lowcut,
        highcut,
        SAMPLING_RATE,
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal,
    )

    return filtered_signal


def normalize_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Normalize ECG signal between 0 and 1.
    """

    signal = signal.astype(np.float32)

    signal = (
        signal - signal.min()
    ) / (
        signal.max() - signal.min()
    )

    return signal


def plot_signals(
    original: np.ndarray,
    filtered: np.ndarray,
    samples: int = 2000,
) -> None:
    """
    Compare original and filtered ECG.
    """

    plt.figure(figsize=(15, 5))

    plt.plot(
        original[:samples],
        label="Original ECG",
        alpha=0.7,
    )

    plt.plot(
        filtered[:samples],
        label="Filtered ECG",
        linewidth=2,
    )

    plt.title("ECG Signal Filtering")

    plt.xlabel("Sample")

    plt.ylabel("Amplitude")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def main() -> None:

    record_id = 100

    ecg = load_ecg_record(record_id)

    signal = get_ecg_signal(
        ecg,
        ECG_LEAD,
    ).to_numpy()

    logger.info("Filtering ECG signal...")

    filtered_signal = filter_ecg(signal)

    normalized_signal = normalize_signal(
        filtered_signal
    )

    logger.info(
        "Original signal shape : %s",
        signal.shape,
    )

    logger.info(
        "Filtered signal shape : %s",
        filtered_signal.shape,
    )

    logger.info(
        "Normalized signal range : %.3f to %.3f",
        normalized_signal.min(),
        normalized_signal.max(),
    )

    plot_signals(
        signal,
        filtered_signal,
    )


if __name__ == "__main__":
    main()