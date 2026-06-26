"""
Loss Functions

Implements Cross Entropy Loss and Focal Loss for ECG classification.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.class_weights import compute_class_weights
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


def get_cross_entropy_loss(
    weighted: bool = True,
) -> nn.Module:
    """
    Create CrossEntropyLoss.
    """

    if weighted:

        weights = compute_class_weights()

        logger.info(
            "Using weighted CrossEntropyLoss."
        )

        return nn.CrossEntropyLoss(
            weight=weights,
        )

    logger.info(
        "Using standard CrossEntropyLoss."
    )

    return nn.CrossEntropyLoss()


def get_focal_loss(
    gamma: float = 2.0,
    weighted: bool = True,
) -> FocalLoss:
    """
    Create Focal Loss.
    """

    alpha = None

    if weighted:
        alpha = compute_class_weights()

        logger.info(
            "Using weighted Focal Loss."
        )

    else:

        logger.info(
            "Using standard Focal Loss."
        )

    return FocalLoss(
        alpha=alpha,
        gamma=gamma,
    )


def main() -> None:
    """
    Example usage.
    """

    logger.info("=" * 70)
    logger.info("Testing Loss Functions")
    logger.info("=" * 70)

    focal = get_focal_loss()

    cross_entropy = get_cross_entropy_loss()

    logits = torch.randn(
        16,
        5,
    )

    labels = torch.randint(
        0,
        5,
        (
            16,
        ),
    )

    focal_loss = focal(
        logits,
        labels,
    )

    ce_loss = cross_entropy(
        logits,
        labels,
    )

    logger.info(
        "Focal Loss        : %.6f",
        focal_loss.item(),
    )

    logger.info(
        "Cross Entropy Loss: %.6f",
        ce_loss.item(),
    )

    logger.info("=" * 70)
    logger.info("Loss Function Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()