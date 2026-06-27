"""
PyTorch DataLoaders

Creates optimized DataLoaders for the MIT-BIH heartbeat dataset.
"""

from typing import Dict

from torch.utils.data import DataLoader

from src.configs import config
from src.data.heartbeat_dataset import HeartbeatDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

PERSISTENT_WORKERS = config.NUM_WORKERS > 0

# ==========================================================


def create_dataloader(
    split: str,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = config.NUM_WORKERS,
) -> DataLoader:
    """
    Create a DataLoader for a dataset split.

    Parameters
    ----------
    split : str
        Dataset split.
        One of:
            - train
            - valid
            - test

    batch_size : int
        Mini-batch size.

    shuffle : bool
        Whether to shuffle the dataset.

    num_workers : int
        Number of worker processes.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader.
    """

    dataset = HeartbeatDataset(split=split)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=config.DROP_LAST,
    )

    logger.info(
        "%s DataLoader created (%d samples).",
        split.capitalize(),
        len(dataset),
    )

    return loader


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation and test datasets.
    """

    return {
        "train": create_dataloader(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "valid": create_dataloader(
            split="valid",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": create_dataloader(
            split="test",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def main() -> None:
    """
    Example usage.
    """

    dataloaders = create_dataloaders()

    train_loader = dataloaders["train"]

    logger.info("")

    logger.info(
        "Number of training batches : %d",
        len(train_loader),
    )

    batch = next(iter(train_loader))

    logger.info(
        "Signal Batch Shape : %s",
        tuple(batch["signal"].shape),
    )

    logger.info(
        "Label Batch Shape : %s",
        tuple(batch["label"].shape),
    )

    logger.info(
        "First Record ID : %d",
        batch["record_id"][0],
    )

    logger.info(
        "First Label : %d",
        batch["label"][0].item(),
    )


if __name__ == "__main__":
    main()