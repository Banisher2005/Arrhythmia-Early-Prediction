"""
Training Curves

Plots training and validation loss/accuracy curves from the saved
training history.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Paths
# ==========================================================

HISTORY_PATH = Path("models_saved/training_history.json")
OUTPUT_PATH = Path("results/training_curves.png")

# ==========================================================


def plot_training_curves() -> None:
    """
    Plot training history.
    """

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    valid_loss = history["valid_loss"]

    train_acc = history["train_accuracy"]
    valid_acc = history["valid_accuracy"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        train_loss,
        linewidth=2,
        label="Train",
    )

    plt.plot(
        epochs,
        valid_loss,
        linewidth=2,
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        train_acc,
        linewidth=2,
        label="Train",
    )

    plt.plot(
        epochs,
        valid_acc,
        linewidth=2,
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Training curves saved to %s",
        OUTPUT_PATH,
    )


def main() -> None:
    """
    Example usage.
    """

    logger.info("=" * 70)
    logger.info("Generating Training Curves")
    logger.info("=" * 70)

    plot_training_curves()

    logger.info("=" * 70)
    logger.info("Training Curves Generated Successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()