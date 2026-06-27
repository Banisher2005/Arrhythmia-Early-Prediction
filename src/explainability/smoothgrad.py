"""
SmoothGrad

Research-grade SmoothGrad implementation for ECG explainability.

Reference
---------
Smilkov et al.
SmoothGrad: removing noise by adding noise.

Adapted for 1D ECG signals.

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.explainability.attribution import AttributionMethod
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SmoothGrad(AttributionMethod):
    """
    SmoothGrad implementation.

    Computes

        Saliency(x + noise)

    over multiple noisy samples and averages the
    resulting gradients.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        samples: int = 50,
        noise_level: float = 0.15,
    ):

        super().__init__(
            model=model,
            device=device,
        )

        self.samples = samples
        self.noise_level = noise_level

    ###########################################################################

    def _generate_noise(
        self,
        ecg: torch.Tensor,
    ) -> torch.Tensor:

        std = (
            ecg.max() - ecg.min()
        ) * self.noise_level

        return torch.randn_like(
            ecg
        ) * std

    ###########################################################################

    def generate(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
    ):

        ecg = self.prepare_input(
            ecg,
            self.device,
        )

        accumulated_gradient = torch.zeros_like(
            ecg
        )

        with torch.no_grad():

            outputs = self.model(
                ecg
            )

            prediction, confidence = self.prediction(
                outputs
            )

        if target_class is None:
            target_class = prediction

        for _ in range(
            self.samples
        ):

            noisy = (
                ecg.detach()
                + self._generate_noise(
                    ecg
                )
            )

            noisy.requires_grad_(True)

            self.zero_grad(
                self.model
            )

            outputs = self.model(
                noisy
            )

            score = outputs[
                "logits"
            ][
                :,
                target_class,
            ]

            score.backward()

            accumulated_gradient += (
                noisy.grad.abs()
            )

        attribution = (
            accumulated_gradient
            / self.samples
        )

        attribution = (
            attribution.squeeze()
        )

        attribution = self.normalize(
            attribution
        )

        return (
            attribution.cpu().numpy(),
            prediction,
            confidence,
        )

    ###########################################################################

    def explain(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):

        attribution, prediction, confidence = (
            self.generate(
                ecg,
                target_class,
            )
        )

        waveform = (
            ecg.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        self.plot(
            ecg=waveform,
            attribution=attribution,
            title=(
                f"SmoothGrad | "
                f"Prediction={prediction} | "
                f"Confidence={confidence:.4f}"
            ),
            cmap="hot",
            save_path=save_path,
            show=show,
        )

        return {
            "smoothgrad": attribution,
            "prediction": prediction,
            "confidence": confidence,
        }

    ###########################################################################

    def batch_generate(
        self,
        batch: torch.Tensor,
        target_class: Optional[int] = None,
    ):

        maps = []

        predictions = []

        confidences = []

        for sample in batch:

            saliency, pred, conf = (
                self.generate(
                    sample.unsqueeze(0),
                    target_class,
                )
            )

            maps.append(
                saliency
            )

            predictions.append(
                pred
            )

            confidences.append(
                conf
            )

        return (
            np.asarray(
                maps
            ),
            predictions,
            confidences,
        )
            ###########################################################################
    # Save
    ###########################################################################

    def save(
        self,
        ecg: np.ndarray,
        attribution: np.ndarray,
        save_path: str,
    ):

        self.plot(
            ecg=ecg,
            attribution=attribution,
            title="SmoothGrad",
            cmap="hot",
            save_path=save_path,
            show=False,
        )

    ###########################################################################
    # Overlay
    ###########################################################################

    def overlay(
        self,
        ecg: np.ndarray,
        attribution: np.ndarray,
        alpha: float = 0.70,
    ):

        fig = plt.figure(
            figsize=(14, 4)
        )

        plt.plot(
            ecg,
            color="black",
            linewidth=1.5,
            zorder=2,
        )

        plt.scatter(
            np.arange(
                len(ecg)
            ),
            ecg,
            c=attribution,
            cmap="hot",
            s=18,
            alpha=alpha,
            zorder=3,
        )

        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.title("SmoothGrad")

        plt.grid(True)

        plt.tight_layout()

        return fig


###############################################################################
# Standalone Test
###############################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing SmoothGrad")
    logger.info("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AMSRAN_GF().to(device)

    model.eval()

    ecg = torch.randn(
        1,
        1,
        180,
        device=device,
    )

    explainer = SmoothGrad(
        model=model,
        device=device,
        samples=50,
        noise_level=0.15,
    )

    result = explainer.explain(
        ecg=ecg,
        save_path=(
            "results/smoothgrad/"
            "example_smoothgrad.png"
        ),
        show=False,
    )

    logger.info(
        "Prediction : %d",
        result["prediction"],
    )

    logger.info(
        "Confidence : %.4f",
        result["confidence"],
    )

    logger.info(
        "SmoothGrad Shape : %s",
        result["smoothgrad"].shape,
    )

    logger.info("=" * 70)
    logger.info("SmoothGrad Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()