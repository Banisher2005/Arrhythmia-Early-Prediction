"""
Class Weight Computation

Computes normalized class weights for imbalanced heartbeat classification.
"""

from typing import Dict

import numpy as np
import torch

from src.data.heartbeat_dataset import HeartbeatDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_WEIGHT = 25.0


def compute_class_weights() -> torch.Tensor:
    """
    Compute normalized class weights.

    Returns
    -------
    torch.Tensor
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

    weights = total_samples / (num_classes * counts)

    # Normalize so average weight = 1
    weights = weights / weights.mean()

    # Prevent extreme values
    weights = np.clip(
        weights,
        0.10,
        MAX_WEIGHT,
    )

    weights = weights.astype(np.float32)

    logger.info("=" * 70)
    logger.info("Normalized Training Class Weights")
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

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def get_class_distribution() -> Dict[int, int]:

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

    weights = compute_class_weights()

    logger.info(
        "Weights Tensor: %s",
        weights,
    )


if __name__ == "__main__":
    main()