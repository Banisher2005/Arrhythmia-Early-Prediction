"""
Training Engine

Handles training and validation for AMSRAN-GF.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.training.early_stopping import EarlyStopping
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs: int = 100,
        save_dir: str | Path = "models_saved",
    ) -> None:

        self.model = model.to(device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.epochs = epochs

        self.scaler = GradScaler(
    "cuda",
    enabled=self.device.type == "cuda",
)

        self.save_dir = Path(save_dir)

        self.early_stopping = EarlyStopping(
        patience=15,
        checkpoint_path=self.save_dir / "best_model.pt",
        )

        self.history = {
            "train_loss": [],
            "valid_loss": [],
            "train_accuracy": [],
            "valid_accuracy": [],
        }

    def train_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )

        for batch in progress:

            signal = batch["signal"].to(
                self.device,
                non_blocking=True,
            )

            label = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad()

            with autocast(
    "cuda",
    enabled=self.device.type == "cuda",
):

                outputs = self.model(signal)

                logits = outputs["logits"]

                loss = self.criterion(
                    logits,
                    label,
                )

            self.scaler.scale(loss).backward()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            running_loss += loss.item()

            prediction = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                prediction == label
            ).sum().item()

            total += label.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        loss = running_loss / len(
            self.train_loader
        )

        accuracy = (
            100.0 * correct / total
        )

        return loss, accuracy

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(
            self.valid_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress:

            signal = batch["signal"].to(
                self.device,
                non_blocking=True,
            )

            label = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(
                signal
            )

            logits = outputs["logits"]

            loss = self.criterion(
                logits,
                label,
            )

            running_loss += loss.item()

            prediction = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                prediction == label
            ).sum().item()

            total += label.size(0)

        loss = running_loss / len(
            self.valid_loader
        )

        accuracy = (
            100.0 * correct / total
        )

        return loss, accuracy

    def fit(self):

        logger.info("=" * 70)
        logger.info("Training Started")
        logger.info("=" * 70)

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            logger.info(
                "Epoch %d/%d",
                epoch,
                self.epochs,
            )

            train_loss, train_acc = (
                self.train_epoch()
            )

            valid_loss, valid_acc = (
                self.validate()
            )

            self.scheduler.step()

            self.history[
                "train_loss"
            ].append(train_loss)

            self.history[
                "valid_loss"
            ].append(valid_loss)

            self.history[
                "train_accuracy"
            ].append(train_acc)

            self.history[
                "valid_accuracy"
            ].append(valid_acc)

            logger.info(
                "Train Loss : %.4f | Train Acc : %.2f%%",
                train_loss,
                train_acc,
            )

            logger.info(
                "Valid Loss : %.4f | Valid Acc : %.2f%%",
                valid_loss,
                valid_acc,
            )

            logger.info(
                "Learning Rate : %.8f",
                self.optimizer.param_groups[0]["lr"],
            )

            stop = self.early_stopping(
                valid_loss,
                self.model,
            )

            if stop:

                logger.info(
                    "Early stopping."
                )

                break

            logger.info("-" * 70)

        logger.info("=" * 70)
        logger.info("Training Finished")
        logger.info("=" * 70)

        return self.history

    def save_history(
        self,
        filepath: str | Path,
    ) -> None:
        """
        Save training history.
        """

        import json

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            filepath,
            "w",
        ) as file:
            json.dump(
                self.history,
                file,
                indent=4,
            )

        logger.info(
            "Training history saved to %s",
            filepath,
        )

    def load_best_model(
        self,
    ) -> None:
        """
        Load the best saved model.
        """

        checkpoint = (
            self.save_dir
            / "best_model.pt"
        )

        if not checkpoint.exists():

            raise FileNotFoundError(
                checkpoint
            )

        self.model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=self.device,
            )
        )

        logger.info(
            "Loaded best model from %s",
            checkpoint,
        )

    def evaluate(
        self,
        test_loader,
    ):
        """
        Evaluate model.
        """

        self.model.eval()

        correct = 0

        total = 0

        predictions = []

        labels = []

        with torch.no_grad():

            progress = tqdm(
                test_loader,
                desc="Testing",
                leave=False,
            )

            for batch in progress:

                signal = batch[
                    "signal"
                ].to(
                    self.device,
                    non_blocking=True,
                )

                target = batch[
                    "label"
                ].to(
                    self.device,
                    non_blocking=True,
                )

                outputs = self.model(
                    signal
                )

                logits = outputs[
                    "logits"
                ]

                prediction = torch.argmax(
                    logits,
                    dim=1,
                )

                correct += (
                    prediction == target
                ).sum().item()

                total += target.size(0)

                predictions.extend(
                    prediction.cpu().numpy()
                )

                labels.extend(
                    target.cpu().numpy()
                )

        accuracy = (
            100.0 * correct / total
        )

        logger.info(
            "Test Accuracy : %.2f%%",
            accuracy,
        )

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "labels": labels,
        }

    def get_history(
        self,
    ):
        """
        Return training history.
        """

        return self.history

    def get_model(
        self,
    ):
        """
        Return trained model.
        """

        return self.model

    def get_optimizer(
        self,
    ):
        """
        Return optimizer.
        """

        return self.optimizer

    def get_scheduler(
        self,
    ):
        """
        Return scheduler.
        """

        return self.scheduler


def main() -> None:
    """
    Example usage.
    """

    import torch

    from src.data.dataloaders import create_dataloaders
    from src.models.amsran_gf import AMSRAN_GF
    from src.training.losses import get_focal_loss
    from src.training.optimizer import get_optimizer
    from src.training.scheduler import get_scheduler

    logger.info("=" * 70)
    logger.info("Testing Trainer")
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

    dataloaders = create_dataloaders()

    model = AMSRAN_GF()

    criterion = get_focal_loss()

    optimizer = get_optimizer(
        model,
    )

    scheduler = get_scheduler(
        optimizer,
        epochs=5,
    )

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=1,
    )

    history = trainer.fit()

    logger.info(
        "History Keys : %s",
        list(history.keys()),
    )

    logger.info("=" * 70)
    logger.info("Trainer Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()