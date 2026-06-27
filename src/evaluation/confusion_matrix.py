"""
Confusion Matrix

Generates and saves a publication-quality confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

CLASS_NAMES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]

SAVE_DIRECTORY = Path("results")

SAVE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
    save_path: Path = SAVE_DIRECTORY / "confusion_matrix.png",
) -> None:
    """
    Plot and save confusion matrix.
    """

    if normalize:

        cm = confusion_matrix(
            y_true,
            y_pred,
            normalize="true",
        )

        title = "Normalized Confusion Matrix"

    else:

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        title = "Confusion Matrix"

    fig, ax = plt.subplots(
        figsize=(8, 8),
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=True,
        values_format=".2f" if normalize else "d",
    )

    ax.set_title(
        title,
        fontsize=16,
        pad=20,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Confusion matrix saved to %s",
        save_path,
    )


def main() -> None:
    """
    Example usage.
    """

    logger.info("=" * 70)
    logger.info("Testing Confusion Matrix")
    logger.info("=" * 70)

    rng = np.random.default_rng(42)

    y_true = rng.integers(
        0,
        5,
        1000,
    )

    y_pred = rng.integers(
        0,
        5,
        1000,
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        normalize=True,
        save_path=SAVE_DIRECTORY / "confusion_matrix_normalized.png",
    )

    logger.info("=" * 70)
    logger.info("Confusion Matrix Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()