"""
Grad-CAM++

Research-grade 1D Grad-CAM++ implementation for AMSRAN-GF.

Reference
---------
Chattopadhyay et al.
Grad-CAM++: Improved Visual Explanations for Deep
Convolutional Networks

Adapted for 1D ECG signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.explainability.attribution import AttributionMethod
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAMPlusPlus(AttributionMethod):
    """
    Grad-CAM++ implementation for 1D ECG.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ):

        super().__init__(
            model=model,
            device=device,
        )

    ###########################################################################

    @staticmethod
    def _compute_alpha(
        gradients: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Grad-CAM++ alpha coefficients.
        """

        grad2 = gradients.pow(2)

        grad3 = gradients.pow(3)

        denominator = (
            2 * grad2
            + activations * grad3.sum(
                dim=2,
                keepdim=True,
            )
        )

        denominator = torch.where(
            denominator != 0,
            denominator,
            torch.ones_like(
                denominator
            ),
        )

        alpha = grad2 / denominator

        return alpha

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

        self.zero_grad(
            self.model
        )

        outputs = self.model(
            ecg
        )

        logits = outputs["logits"]

        probabilities = outputs[
            "probabilities"
        ]

        feature_map = outputs[
            "cnn_feature_map"
        ]

        feature_map.retain_grad()

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

        if target_class is None:
            target_class = prediction.item()

        score = logits[
            :,
            target_class,
        ]

        score.backward(
            retain_graph=True
        )

        gradients = feature_map.grad

        if gradients is None:

            raise RuntimeError(
                "Unable to compute Grad-CAM++ gradients."
            )

        alpha = self._compute_alpha(
            gradients,
            feature_map,
        )

        positive_gradients = F.relu(
            gradients
        )

        weights = (
            alpha
            * positive_gradients
        ).sum(
            dim=2,
            keepdim=True,
        )

        cam = (
            weights
            * feature_map
        ).sum(
            dim=1
        )

        cam = F.relu(
            cam
        )

        cam = self.normalize(
            cam
        )

        cam = self.interpolate(
            cam,
            ecg.shape[-1],
        )

        confidence = probabilities[
            0,
            target_class,
        ].item()

        return (
            cam.squeeze().cpu().numpy(),
            prediction.item(),
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

        cam, prediction, confidence = self.generate(
            ecg,
            target_class,
        )

        waveform = (
            ecg.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        self.plot(
            ecg=waveform,
            attribution=cam,
            title=(
                f"Grad-CAM++ | "
                f"Prediction={prediction} | "
                f"Confidence={confidence:.4f}"
            ),
            cmap="jet",
            save_path=save_path,
            show=show,
        )

        return {
            "gradcam_pp": cam,
            "prediction": prediction,
            "confidence": confidence,
        }

    ###########################################################################

    def batch_generate(
        self,
        batch: torch.Tensor,
        target_class: Optional[int] = None,
    ):

        cams = []

        predictions = []

        confidences = []

        for sample in batch:

            cam, pred, conf = self.generate(
                sample.unsqueeze(0),
                target_class,
            )

            cams.append(
                cam
            )

            predictions.append(
                pred
            )

            confidences.append(
                conf
            )

        return (
            np.asarray(
                cams
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
            title="Grad-CAM++",
            cmap="jet",
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
            cmap="jet",
            s=18,
            alpha=alpha,
            zorder=3,
        )

        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.title("Grad-CAM++")

        plt.grid(True)

        plt.tight_layout()

        return fig


###############################################################################
# Standalone Test
###############################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing Grad-CAM++")
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

    explainer = GradCAMPlusPlus(
        model=model,
        device=device,
    )

    result = explainer.explain(
        ecg=ecg,
        save_path=(
            "results/gradcam_pp/"
            "example_gradcam_pp.png"
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
        "CAM Shape : %s",
        result["gradcam_pp"].shape,
    )

    logger.info("=" * 70)
    logger.info("Grad-CAM++ Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()