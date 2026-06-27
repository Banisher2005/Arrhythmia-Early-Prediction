"""
Train AMSRAN-GF

Main training script.
"""

from pathlib import Path

import torch

from src.data.dataloaders import create_dataloaders
from src.models.amsran_gf import AMSRAN_GF
from src.training.losses import get_focal_loss
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

SAVE_DIRECTORY = Path("models_saved")

HISTORY_FILE = SAVE_DIRECTORY / "training_history.json"

# ==========================================================


def main() -> None:

    logger.info("=" * 70)
    logger.info("AMSRAN-GF Training")
    logger.info("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "Device : %s",
        device,
    )

    # ======================================================
    # Data
    # ======================================================

    dataloaders = create_dataloaders()

    # ======================================================
    # Model
    # ======================================================

    model = AMSRAN_GF()

    # ======================================================
    # Loss
    # ======================================================

    criterion = get_focal_loss(
        weighted=True,
        gamma=2.0,
    )

    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = get_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ======================================================
    # Scheduler
    # ======================================================

    scheduler = get_scheduler(
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    # ======================================================
    # Trainer
    # ======================================================

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        save_dir=SAVE_DIRECTORY,
    )

    # ======================================================
    # Train
    # ======================================================

    history = trainer.fit()

    trainer.save_history(
        HISTORY_FILE,
    )

    # ======================================================
    # Load Best Model
    # ======================================================

    trainer.load_best_model()

    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluating Best Model")
    logger.info("=" * 70)

    results = trainer.evaluate(
        dataloaders["test"],
    )

    logger.info(
        "Final Test Accuracy : %.2f%%",
        results["accuracy"],
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Training Complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()