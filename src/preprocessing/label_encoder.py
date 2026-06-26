"""
Label Encoder

Converts MIT-BIH annotation symbols into the
standard AAMI EC57 classes.

Author:
Abhinav Kumar
"""

import numpy as np

from src.configs.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------
# AAMI Mapping
# -----------------------------

AAMI_MAPPING = {
    # Normal
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # Supraventricular
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # Ventricular
    "V": "V",
    "E": "V",

    # Fusion
    "F": "F",

    # Unknown / Paced
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


CLASS_TO_INT = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}


from collections import Counter

def encode_labels(labels):

    encoded = []
    class_names = []

    skipped = []

    for label in labels:

        if label not in AAMI_MAPPING:
            skipped.append(label)
            continue

        aami = AAMI_MAPPING[label]

        class_names.append(aami)
        encoded.append(CLASS_TO_INT[aami])

    logger.info("Skipped labels: %d", len(skipped))

    if skipped:
        logger.info("Skipped annotation types:")
        for symbol, count in Counter(skipped).items():
            logger.info("%s : %d", symbol, count)

    return (
        np.array(class_names),
        np.array(encoded, dtype=np.int64),
    )


def print_distribution(class_names):

    logger.info("=" * 50)
    logger.info("Class Distribution")
    logger.info("=" * 50)

    unique, counts = np.unique(class_names, return_counts=True)

    for c, n in zip(unique, counts):
        logger.info("%s : %d", c, n)


def main():

    labels = np.load(
        PROCESSED_DATA_DIR / "labels.npy",
        allow_pickle=True,
    )

    class_names, encoded = encode_labels(labels)

    np.save(
        PROCESSED_DATA_DIR / "aami_labels.npy",
        class_names,
    )

    np.save(
        PROCESSED_DATA_DIR / "encoded_labels.npy",
        encoded,
    )

    print_distribution(class_names)

    logger.info("Encoded labels saved successfully.")


if __name__ == "__main__":
    main()