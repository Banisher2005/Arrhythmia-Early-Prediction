"""
Project Data Types

Common data structures used throughout the project.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AnnotationData:
    """
    Stores heartbeat annotation information.
    """

    positions: np.ndarray
    labels: np.ndarray


@dataclass(slots=True)
class BeatDataset:
    """
    Stores segmented heartbeat data.
    """

    beats: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray