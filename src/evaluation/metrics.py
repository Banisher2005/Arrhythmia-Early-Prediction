"""
Evaluation Metrics

Computes classification metrics for ECG heartbeat classification.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# MIT-BIH Classes
# ==========================================================

CLASS_NAMES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]

# ==========================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth labels.

    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    Dict
        Dictionary containing all metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    precision_class, recall_class, f1_class, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            labels=range(len(CLASS_NAMES)),
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "per_class": {},
    }

    for idx, name in enumerate(CLASS_NAMES):

        metrics["per_class"][name] = {
            "precision": float(precision_class[idx]),
            "recall": float(recall_class[idx]),
            "f1": float(f1_class[idx]),
            "support": int(support[idx]),
        }

    return metrics


def print_metrics(
    metrics: Dict,
) -> None:
    """
    Log all metrics.
    """

    logger.info("=" * 70)
    logger.info("Evaluation Metrics")
    logger.info("=" * 70)

    logger.info(
        "Accuracy           : %.4f",
        metrics["accuracy"],
    )

    logger.info(
        "Macro Precision    : %.4f",
        metrics["macro_precision"],
    )

    logger.info(
        "Macro Recall       : %.4f",
        metrics["macro_recall"],
    )

    logger.info(
        "Macro F1           : %.4f",
        metrics["macro_f1"],
    )

    logger.info(
        "Weighted Precision : %.4f",
        metrics["weighted_precision"],
    )

    logger.info(
        "Weighted Recall    : %.4f",
        metrics["weighted_recall"],
    )

    logger.info(
        "Weighted F1        : %.4f",
        metrics["weighted_f1"],
    )

    logger.info("")
    logger.info("Per-Class Metrics")
    logger.info("-" * 70)

    for name in CLASS_NAMES:

        cls = metrics["per_class"][name]

        logger.info(
            "%s | Precision %.4f | Recall %.4f | F1 %.4f | Support %d",
            name,
            cls["precision"],
            cls["recall"],
            cls["f1"],
            cls["support"],
        )

    logger.info("=" * 70)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """
    Return sklearn classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )


def main() -> None:
    """
    Example usage.
    """

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

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    print_metrics(
        metrics,
    )

    logger.info("")
    logger.info("Classification Report")
    logger.info("-" * 70)

    report = get_classification_report(
        y_true,
        y_pred,
    )

    print(report)


if __name__ == "__main__":
    main()