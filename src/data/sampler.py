"""
Weighted Random Sampler

Balances heartbeat classes during training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.heartbeat_dataset import HeartbeatDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_weighted_sampler() -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for the training dataset.
    """

    dataset = HeartbeatDataset(
        split="train",
    )

    labels = np.asarray(
        dataset.labels,
        dtype=np.int64,
    )

    classes, counts = np.unique(
        labels,
        return_counts=True,
    )

    logger.info("=" * 70)
    logger.info("Creating Weighted Random Sampler")
    logger.info("=" * 70)

    # ==========================================================
    # Square-root inverse frequency
    # Less aggressive than 1 / frequency
    # ==========================================================

    class_weights = (
        1.0 /
        np.sqrt(
            counts.astype(np.float64)
        )
    )

    # Normalize
    class_weights = (
        class_weights /
        class_weights.mean()
    )

    # Assign weight to every sample
    sample_weights = class_weights[
        labels
    ]

    for cls, count, weight in zip(
        classes,
        counts,
        class_weights,
    ):
        logger.info(
            "Class %d | Samples: %-7d | Sampling Weight: %.4f",
            cls,
            count,
            weight,
        )

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    logger.info("")

    logger.info(
        "Sampler Size : %d",
        len(sample_weights),
    )

    logger.info(
        "Replacement : %s",
        sampler.replacement,
    )

    logger.info("=" * 70)

    return sampler


def main() -> None:
    """
    Example usage.
    """

    sampler = create_weighted_sampler()

    logger.info(
        "Sampler Type : %s",
        sampler.__class__.__name__,
    )

    logger.info(
        "Num Samples : %d",
        sampler.num_samples,
    )


if __name__ == "__main__":
    main()