"""
Base Attribution Framework

Shared utilities for all explainability methods.

Supported

- GradCAM
- GradCAM++
- Saliency
- SmoothGrad
- Integrated Gradients
- Future attribution methods

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttributionMethod(ABC):
    """
    Base class for all attribution methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ):

        self.model = model
        self.model.eval()

        if device is None:
            device = next(model.parameters()).device

        self.device = device

    ###########################################################################

    @staticmethod
    def normalize(
        attribution: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize attribution to [0,1]
        """

        attribution = attribution.detach()

        attribution = attribution - attribution.min()

        maximum = attribution.max()

        if maximum > 0:
            attribution = attribution / maximum

        return attribution

    ###########################################################################

    @staticmethod
    def absolute(
        attribution: torch.Tensor,
    ) -> torch.Tensor:

        return attribution.abs()

    ###########################################################################

    @staticmethod
    def interpolate(
        attribution: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Linear interpolation.
        """

        if attribution.ndim == 1:
            attribution = attribution.unsqueeze(0)

        attribution = attribution.unsqueeze(1)

        attribution = F.interpolate(
            attribution,
            size=target_length,
            mode="linear",
            align_corners=False,
        )

        attribution = attribution.squeeze(1)

        return attribution

    ###########################################################################

    @staticmethod
    def prediction(
        outputs,
    ):

        probabilities = outputs["probabilities"]

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

        confidence = probabilities[
            0,
            prediction,
        ].item()

        return (
            prediction.item(),
            confidence,
        )

    ###########################################################################

    @staticmethod
    def plot(
        ecg: np.ndarray,
        attribution: np.ndarray,
        title: str,
        cmap: str = "jet",
        save_path: Optional[str] = None,
        show: bool = True,
    ):

        x = np.arange(
            len(ecg)
        )

        plt.figure(
            figsize=(14,4)
        )

        plt.plot(
            x,
            ecg,
            color="black",
            linewidth=1.5,
            zorder=3,
        )

        scatter = plt.scatter(
            x,
            ecg,
            c=attribution,
            cmap=cmap,
            s=18,
            zorder=5,
        )

        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.title(title)

        plt.grid(True)

        plt.colorbar(
            scatter,
            label="Importance",
        )

        plt.tight_layout()

        if save_path is not None:

            save_path = Path(save_path)

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            logger.info(
                "Saved %s",
                save_path,
            )

        if show:
            plt.show()
        else:
            plt.close()

    ###########################################################################

    @staticmethod
    def overlay(
        ecg: np.ndarray,
        attribution: np.ndarray,
        cmap: str = "jet",
        alpha: float = 0.7,
    ):

        fig = plt.figure(
            figsize=(14,4)
        )

        plt.plot(
            ecg,
            color="black",
            linewidth=1.4,
        )

        plt.scatter(
            np.arange(
                len(ecg)
            ),
            ecg,
            c=attribution,
            cmap=cmap,
            alpha=alpha,
            s=18,
        )

        plt.grid(True)

        plt.tight_layout()

        return fig

    ###########################################################################

    @staticmethod
    def to_numpy(
        tensor: torch.Tensor,
    ) -> np.ndarray:

        return (
            tensor
            .detach()
            .cpu()
            .numpy()
        )

    ###########################################################################

    @staticmethod
    def prepare_input(
        ecg: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:

        ecg = ecg.to(device)

        ecg.requires_grad_(True)

        return ecg

    ###########################################################################

    @staticmethod
    def zero_grad(
        model,
    ):

        model.zero_grad(
            set_to_none=True
        )

    ###########################################################################

    @abstractmethod
    def generate(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
    ):
        """
        Must return

        attribution

        prediction

        confidence
        """

        raise NotImplementedError