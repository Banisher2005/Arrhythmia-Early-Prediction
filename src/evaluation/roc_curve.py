"""
ROC Curve

Generates publication-quality ROC curves and AUC scores
for multi-class ECG heartbeat classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

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

N_CLASSES = len(CLASS_NAMES)

SAVE_DIRECTORY = Path("results")
SAVE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================


def compute_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict:
    """
    Compute ROC curves and AUC scores.

    Parameters
    ----------
    y_true : ndarray
        Shape (N,)

    y_score : ndarray
        Shape (N, num_classes)

    Returns
    -------
    dict
    """

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(N_CLASSES),
    )

    fpr = {}
    tpr = {}
    roc_auc = {}

    # ------------------------------------------------------

    for i in range(N_CLASSES):

        fpr[i], tpr[i], _ = roc_curve(
            y_true_bin[:, i],
            y_score[:, i],
        )

        roc_auc[i] = auc(
            fpr[i],
            tpr[i],
        )

    # ------------------------------------------------------
    # Micro Average
    # ------------------------------------------------------

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(),
        y_score.ravel(),
    )

    roc_auc["micro"] = auc(
        fpr["micro"],
        tpr["micro"],
    )

    # ------------------------------------------------------
    # Macro Average
    # ------------------------------------------------------

    roc_auc["macro"] = roc_auc_score(
        y_true_bin,
        y_score,
        average="macro",
        multi_class="ovr",
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
    }


def plot_roc_curve(
    roc_results: Dict,
    save_path: Path = SAVE_DIRECTORY / "roc_curve.png",
) -> None:
    """
    Plot ROC curves.
    """

    plt.figure(
        figsize=(9, 8),
    )

    for i, cls in enumerate(CLASS_NAMES):

        plt.plot(
            roc_results["fpr"][i],
            roc_results["tpr"][i],
            linewidth=2,
            label=(
                f"{cls}"
                f" (AUC={roc_results['auc'][i]:.3f})"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlim([0, 1])

    plt.ylim([0, 1.05])

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend(
        loc="lower right",
    )

    plt.grid(
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "ROC curve saved to %s",
        save_path,
    )


def print_auc(
    roc_results: Dict,
) -> None:
    """
    Print AUC values.
    """

    logger.info("=" * 70)
    logger.info("ROC AUC Scores")
    logger.info("=" * 70)

    for i, cls in enumerate(CLASS_NAMES):

        logger.info(
            "%s : %.4f",
            cls,
            roc_results["auc"][i],
        )

    logger.info("")

    logger.info(
        "Micro Average : %.4f",
        roc_results["auc"]["micro"],
    )

    logger.info(
        "Macro Average : %.4f",
        roc_results["auc"]["macro"],
    )

    logger.info("=" * 70)


def main() -> None:
    """
    Example usage.
    """

    logger.info("=" * 70)
    logger.info("Testing ROC Curve")
    logger.info("=" * 70)

    rng = np.random.default_rng(42)

    y_true = rng.integers(
        0,
        N_CLASSES,
        1000,
    )

    scores = rng.random(
        (
            1000,
            N_CLASSES,
        )
    )

    scores /= scores.sum(
        axis=1,
        keepdims=True,
    )

    roc_results = compute_auc(
        y_true,
        scores,
    )

    print_auc(
        roc_results,
    )

    plot_roc_curve(
        roc_results,
    )

    logger.info("ROC Curve Test Passed")


if __name__ == "__main__":
    main()