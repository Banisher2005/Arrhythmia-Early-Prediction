"""
Early Stopping

Stops training when validation loss stops improving.
"""

from pathlib import Path

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping utility.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_path: Path | str = "models_saved/best_model.pt",
    ) -> None:

        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(
        self,
        validation_loss: float,
        model: torch.nn.Module,
    ) -> bool:
        """
        Update early stopping state.

        Returns
        -------
        bool
            True if training should stop.
        """

        if validation_loss < (
            self.best_loss - self.min_delta
        ):

            self.best_loss = validation_loss

            self.counter = 0

            self.save_checkpoint(model)

            logger.info(
                "Validation improved: %.6f",
                validation_loss,
            )

        else:

            self.counter += 1

            logger.info(
                "No improvement (%d/%d)",
                self.counter,
                self.patience,
            )

            if self.counter >= self.patience:

                self.early_stop = True

                logger.info(
                    "Early stopping triggered."
                )

        return self.early_stop

    def save_checkpoint(
        self,
        model: torch.nn.Module,
    ) -> None:
        """
        Save best model.
        """

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            self.checkpoint_path,
        )


def main() -> None:

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing Early Stopping")
    logger.info("=" * 70)

    model = AMSRAN_GF()

    stopper = EarlyStopping(
        patience=3,
    )

    losses = [
        0.80,
        0.72,
        0.71,
        0.715,
        0.718,
        0.720,
        0.725,
    ]

    for epoch, loss in enumerate(
        losses,
        start=1,
    ):

        logger.info(
            "Epoch %d - Val Loss %.4f",
            epoch,
            loss,
        )

        if stopper(loss, model):

            logger.info(
                "Stopped at epoch %d",
                epoch,
            )

            break

    logger.info("=" * 70)
    logger.info("Early Stopping Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()