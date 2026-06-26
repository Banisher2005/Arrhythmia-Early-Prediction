"""
Optimizer Factory

Creates optimizers for AMSRAN-GF.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from src.models.amsran_gf import AMSRAN_GF
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_optimizer(
    model: AMSRAN_GF,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    """
    Create AdamW optimizer.

    Parameters
    ----------
    model : AMSRAN_GF
        Neural network.

    learning_rate : float
        Initial learning rate.

    weight_decay : float
        L2 regularization.

    Returns
    -------
    torch.optim.Optimizer
    """

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    logger.info(
        "Optimizer : AdamW"
    )

    logger.info(
        "Learning Rate : %.6f",
        learning_rate,
    )

    logger.info(
        "Weight Decay : %.6f",
        weight_decay,
    )

    return optimizer


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:
    """
    Count model parameters.

    Returns
    -------
    total_parameters
    trainable_parameters
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def main() -> None:
    """
    Example usage.
    """

    logger.info("=" * 70)
    logger.info("Testing Optimizer")
    logger.info("=" * 70)

    model = AMSRAN_GF()

    optimizer = get_optimizer(model)

    total, trainable = count_parameters(
        model,
    )

    logger.info(
        "Total Parameters     : %d",
        total,
    )

    logger.info(
        "Trainable Parameters : %d",
        trainable,
    )

    logger.info(
        "Optimizer Type : %s",
        optimizer.__class__.__name__,
    )

    logger.info("=" * 70)
    logger.info("Optimizer Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()