"""
Class Weight Computation

Computes class weights from the TRAINING dataset only.

These weights are used to compensate for severe class imbalance
during model training.
"""

from typing import Dict

import numpy as np
import torch

from src.data.heartbeat_dataset import HeartbeatDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_class_weights() -> torch.Tensor:
    """
    Compute inverse-frequency class weights using only the
    training dataset.

    Returns
    -------
    torch.Tensor
        Tensor containing one weight per class.
    """

    dataset = HeartbeatDataset(
        split="train",
    )

    labels = dataset.labels

    classes, counts = np.unique(
        labels,
        return_counts=True,
    )

    total_samples = len(labels)

    num_classes = len(classes)

    weights = (
        total_samples
        / (num_classes * counts)
    )

    weights = weights.astype(np.float32)

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    logger.info("=" * 70)
    logger.info("Training Class Distribution")
    logger.info("=" * 70)

    for cls, count, weight in zip(
        classes,
        counts,
        weights,
    ):
        logger.info(
            "Class %d | Samples: %-7d | Weight: %.4f",
            cls,
            count,
            weight,
        )

    logger.info("=" * 70)

    return class_weights


def get_class_distribution() -> Dict[int, int]:
    """
    Return training class distribution.

    Returns
    -------
    Dict[int, int]
        Mapping from class index to sample count.
    """

    dataset = HeartbeatDataset(
        split="train",
    )

    classes, counts = np.unique(
        dataset.labels,
        return_counts=True,
    )

    return dict(
        zip(
            classes.tolist(),
            counts.tolist(),
        )
    )


def main() -> None:
    """
    Example usage.
    """

    weights = compute_class_weights()

    logger.info(
        "Weights Tensor: %s",
        weights,
    )


if __name__ == "__main__":
    main()