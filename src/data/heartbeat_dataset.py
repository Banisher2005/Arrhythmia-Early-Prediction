"""
PyTorch Heartbeat Dataset

Provides a clean Dataset interface for loading heartbeat segments from
the processed MIT-BIH dataset.

This Dataset is shared by every model in the project.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.configs.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HeartbeatDataset(Dataset):
    """
    PyTorch Dataset for heartbeat classification.

    Each sample is returned as a dictionary to make downstream
    training, evaluation, explainability and visualization easier.
    """

    def __init__(
        self,
        split: str,
        dataset_path: Optional[Path] = None,
        split_path: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Dataset split.
            One of:
                - train
                - valid
                - test

        dataset_path : Path, optional
            Path to mitbih_dataset.npz.

        split_path : Path, optional
            Path to patient_split.npz.
        """

        if split not in ("train", "valid", "test"):
            raise ValueError(
                "split must be one of "
                "('train', 'valid', 'test')."
            )

        self.split = split

        if dataset_path is None:
            dataset_path = (
                PROCESSED_DATA_DIR /
                "mitbih_dataset.npz"
            )

        if split_path is None:
            split_path = (
                PROCESSED_DATA_DIR /
                "patient_split.npz"
            )

        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        if not split_path.exists():
            raise FileNotFoundError(split_path)

        logger.info(
            "Loading %s dataset...",
            split,
        )

        dataset = np.load(
            dataset_path,
            allow_pickle=True,
        )

        split_data = np.load(
            split_path,
            allow_pickle=True,
        )

        indices = split_data[f"{split}_indices"]

        self.beats = dataset["beats"][indices].astype(np.float32)

        self.labels = dataset[
            "encoded_labels"
        ][indices].astype(np.int64)

        self.record_ids = dataset[
            "record_ids"
        ][indices].astype(np.int32)

        self.sample_positions = dataset[
            "sample_positions"
        ][indices].astype(np.int32)

        self.class_names = dataset["labels"][indices]

        logger.info(
            "%s dataset loaded with %d samples.",
            split.capitalize(),
            len(self.beats),
        )

    def __len__(self) -> int:
        """
        Return dataset size.
        """

        return len(self.beats)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        """
        Return one heartbeat sample.
        """

        signal = torch.from_numpy(
            self.beats[index]
        ).unsqueeze(0)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        sample = {
            "signal": signal,
            "label": label,
            "label_name": self.class_names[index],
            "record_id": int(
                self.record_ids[index]
            ),
            "sample_position": int(
                self.sample_positions[index]
            ),
        }

        return sample


def main() -> None:
    """
    Example usage.
    """

    dataset = HeartbeatDataset(
        split="train",
    )

    logger.info(
        "Dataset size: %d",
        len(dataset),
    )

    sample = dataset[0]

    logger.info(
        "Signal Shape : %s",
        tuple(sample["signal"].shape),
    )

    logger.info(
        "Label        : %d",
        sample["label"].item(),
    )

    logger.info(
        "Label Name   : %s",
        sample["label_name"],
    )

    logger.info(
        "Record ID    : %d",
        sample["record_id"],
    )

    logger.info(
        "Sample Pos   : %d",
        sample["sample_position"],
    )


if __name__ == "__main__":
    main()