"""
Learning Rate Scheduler

Creates learning rate schedulers for AMSRAN-GF.
"""

from __future__ import annotations

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_scheduler(
    optimizer: optim.Optimizer,
    epochs: int = 100,
    eta_min: float = 1e-6,
) -> CosineAnnealingLR:
    """
    Create cosine annealing scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        Model optimizer.

    epochs : int
        Total training epochs.

    eta_min : float
        Minimum learning rate.

    Returns
    -------
    CosineAnnealingLR
    """

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )

    logger.info(
        "Scheduler : CosineAnnealingLR"
    )

    logger.info(
        "Epochs : %d",
        epochs,
    )

    logger.info(
        "Minimum LR : %.8f",
        eta_min,
    )

    return scheduler


def main() -> None:
    """
    Example usage.
    """

    from src.models.amsran_gf import AMSRAN_GF
    from src.training.optimizer import get_optimizer

    logger.info("=" * 70)
    logger.info("Testing Scheduler")
    logger.info("=" * 70)

    model = AMSRAN_GF()

    optimizer = get_optimizer(model)

    scheduler = get_scheduler(
        optimizer,
        epochs=100,
    )

    logger.info(
        "Initial LR : %.6f",
        optimizer.param_groups[0]["lr"],
    )

    # Dummy optimization step
    optimizer.zero_grad()

    dummy_loss = sum(
        p.sum() * 0
        for p in model.parameters()
    )

    dummy_loss.backward()

    optimizer.step()

    scheduler.step()

    logger.info(
        "LR After One Step : %.6f",
        optimizer.param_groups[0]["lr"],
    )

    logger.info("=" * 70)
    logger.info("Scheduler Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()