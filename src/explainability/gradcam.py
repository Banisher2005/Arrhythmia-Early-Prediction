"""
Grad-CAM for AMSRAN-GF

Implements 1D Grad-CAM for ECG heartbeat classification.

Paper
-----
Selvaraju et al.
Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization.

Adapted for 1D ECG signals.

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM1D:
    """
    1D Grad-CAM implementation.

    Expected model output:

    {
        "logits",
        "probabilities",
        "cnn_feature_map",
        ...
    }

    cnn_feature_map must have shape

        (B,C,L)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str = "cnn_feature_map",
        device: Optional[torch.device] = None,
    ):

        self.model = model
        self.model.eval()

        self.target_layer = target_layer

        if device is None:
            device = next(model.parameters()).device

        self.device = device

    ###########################################################################
    # Utilities
    ###########################################################################

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:

        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    @staticmethod
    def _upsample(
        cam: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False,
        )

        return cam.squeeze(1)

    ###########################################################################
    # Core GradCAM
    ###########################################################################

    def generate(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        ecg

            Shape

            (1,1,L)

        target_class

            Optional class index.

            If None, uses predicted class.

        Returns
        -------
        cam

            (L,)

        prediction

        confidence
        """

        ecg = ecg.to(self.device)

        self.model.zero_grad()

        outputs = self.model(ecg)

        logits = outputs["logits"]

        probabilities = outputs["probabilities"]

        feature_map = outputs["cnn_feature_map"]

        feature_map.retain_grad()

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

        if target_class is None:
            target_class = prediction.item()

        score = logits[:, target_class]

        score.backward(retain_graph=True)

        gradients = feature_map.grad

        if gradients is None:
            raise RuntimeError(
                "Gradients are None.\n"
                "Ensure cnn_feature_map participates "
                "in autograd."
            )

        #
        # Global Average Pooling
        #

        weights = gradients.mean(
            dim=2,
            keepdim=True,
        )

        #
        # Weighted Feature Maps
        #

        cam = (weights * feature_map).sum(
            dim=1,
        )

        cam = F.relu(cam)

        cam = self._normalize(cam)

        cam = self._upsample(
            cam,
            ecg.shape[-1],
        )

        cam = cam.squeeze().detach().cpu()

        confidence = probabilities[
            0,
            target_class,
        ].item()

        return (
            cam.numpy(),
            prediction.item(),
            confidence,
        )

    ###########################################################################
    # Visualization
    ###########################################################################

    def plot(
        self,
        ecg: np.ndarray,
        cam: np.ndarray,
        title: str = "Grad-CAM",
        save_path: Optional[str] = None,
    ):

        x = np.arange(len(ecg))

        fig = plt.figure(
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
            c=cam,
            cmap="jet",
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

            Path(
                save_path
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            logger.info(
                "Saved GradCAM -> %s",
                save_path,
            )

        plt.show()

    ###########################################################################
    # Overlay
    ###########################################################################

    def overlay(
        self,
        ecg: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.65,
        cmap: str = "jet",
    ):

        fig = plt.figure(
            figsize=(14,4)
        )

        plt.plot(
            ecg,
            linewidth=1.4,
            color="black",
        )

        plt.scatter(
            np.arange(len(ecg)),
            ecg,
            c=cam,
            cmap=cmap,
            alpha=alpha,
            s=16,
        )

        plt.grid(True)

        plt.tight_layout()

        return fig

    ###########################################################################
    # Batch GradCAM
    ###########################################################################

    def batch_generate(
        self,
        ecg_batch: torch.Tensor,
    ):

        cams = []

        predictions = []

        confidences = []

        for sample in ecg_batch:

            cam, pred, conf = self.generate(
                sample.unsqueeze(0)
            )

            cams.append(cam)

            predictions.append(pred)

            confidences.append(conf)

        return (
            np.asarray(cams),
            predictions,
            confidences,
        )
    ###########################################################################
    # Save Heatmap
    ###########################################################################

    def save(
        self,
        ecg: np.ndarray,
        cam: np.ndarray,
        save_path: str,
        title: str = "Grad-CAM",
    ):
        """
        Save Grad-CAM visualization.
        """

        self.plot(
            ecg=ecg,
            cam=cam,
            title=title,
            save_path=save_path,
        )

    ###########################################################################
    # Explain
    ###########################################################################

    def explain(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        High-level API.

        Parameters
        ----------
        ecg
            ECG tensor of shape (1, 1, L)

        target_class
            Optional target class.

        save_path
            Optional image output path.

        show
            Whether to display the figure.

        Returns
        -------
        dict
        """

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

        if show or save_path is not None:

            x = np.arange(len(waveform))

            plt.figure(figsize=(14, 4))

            plt.plot(
                x,
                waveform,
                color="black",
                linewidth=1.5,
                zorder=3,
            )

            scatter = plt.scatter(
                x,
                waveform,
                c=cam,
                cmap="jet",
                s=18,
                zorder=4,
            )

            plt.xlabel("Samples")
            plt.ylabel("Amplitude")

            plt.title(
                f"Grad-CAM | Prediction={prediction} | Confidence={confidence:.4f}"
            )

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
                    "Grad-CAM saved to %s",
                    save_path,
                )

            if show:
                plt.show()
            else:
                plt.close()

        return {
            "cam": cam,
            "prediction": prediction,
            "confidence": confidence,
        }


###########################################################################
# Standalone Example
###########################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing Grad-CAM")
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

    gradcam = GradCAM1D(
        model=model,
        device=device,
    )

    results = gradcam.explain(
        ecg=ecg,
        save_path="results/gradcam/example_gradcam.png",
        show=False,
    )

    logger.info(
        "Prediction : %d",
        results["prediction"],
    )

    logger.info(
        "Confidence : %.4f",
        results["confidence"],
    )

    logger.info(
        "CAM Shape : %s",
        results["cam"].shape,
    )

    logger.info("=" * 70)
    logger.info("Grad-CAM Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()