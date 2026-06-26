"""
PyTorch DataLoaders

Creates optimized DataLoaders for the MIT-BIH heartbeat dataset.
"""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.heartbeat_dataset import HeartbeatDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 256

NUM_WORKERS = 4

PIN_MEMORY = torch.cuda.is_available()

PERSISTENT_WORKERS = NUM_WORKERS > 0

# ==========================================================


def create_dataloader(
    split: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """
    Create a DataLoader for a dataset split.

    Parameters
    ----------
    split : str
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
    """

    dataset = HeartbeatDataset(split=split)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=False,
    )

    logger.info(
        "%s DataLoader created (%d samples).",
        split.capitalize(),
        len(dataset),
    )

    return loader


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Dict[str, DataLoader]:
    """
    Create train, validation and test DataLoaders.
    """

    train_loader = create_dataloader(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_loader = create_dataloader(
        split="valid",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
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