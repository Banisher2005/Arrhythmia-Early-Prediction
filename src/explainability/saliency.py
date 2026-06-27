"""
Saliency Maps for AMSRAN-GF

Computes gradient-based saliency maps for ECG heartbeat
classification.

Reference
---------
Simonyan et al.
Deep Inside Convolutional Networks:
Visualising Image Classification Models
(Adapted for 1D ECG)

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

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SaliencyMap:
    """
    Gradient-based Saliency Map.

    Computes

        ∂score / ∂input

    and visualizes the absolute gradient as the importance
    of every ECG sample.
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
    def _normalize(
        saliency: torch.Tensor,
    ) -> torch.Tensor:

        saliency = saliency - saliency.min()

        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return saliency

    ###########################################################################

    def generate(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
    ):
        """
        Generate Saliency Map.

        Parameters
        ----------
        ecg

            Shape

            (1,1,L)

        target_class

            Optional class.

        Returns
        -------
        saliency

        prediction

        confidence
        """

        ecg = ecg.to(self.device)

        ecg.requires_grad_(True)

        self.model.zero_grad()

        outputs = self.model(ecg)

        logits = outputs["logits"]

        probabilities = outputs["probabilities"]

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

        if target_class is None:
            target_class = prediction.item()

        score = logits[:, target_class]

        score.backward()

        saliency = ecg.grad.detach().abs()

        saliency = saliency.squeeze()

        saliency = self._normalize(
            saliency
        )

        confidence = probabilities[
            0,
            target_class,
        ].item()

        return (
            saliency.cpu().numpy(),
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

        saliency, prediction, confidence = self.generate(
            ecg,
            target_class,
        )

        waveform = (
            ecg.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        x = np.arange(
            len(waveform)
        )

        plt.figure(
            figsize=(14,4)
        )

        plt.plot(
            waveform,
            color="black",
            linewidth=1.5,
            zorder=3,
        )

        scatter = plt.scatter(
            x,
            waveform,
            c=saliency,
            cmap="hot",
            s=18,
            zorder=5,
        )

        plt.title(
            f"Saliency Map | Prediction={prediction} | Confidence={confidence:.4f}"
        )

        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

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
                "Saved Saliency Map -> %s",
                save_path,
            )

        if show:
            plt.show()
        else:
            plt.close()

        return {
            "saliency": saliency,
            "prediction": prediction,
            "confidence": confidence,
        }


###########################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing Saliency Maps")
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

    saliency = SaliencyMap(
        model=model,
        device=device,
    )

    result = saliency.explain(
        ecg=ecg,
        save_path="results/saliency/example_saliency.png",
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
        "Saliency Shape : %s",
        result["saliency"].shape,
    )

    logger.info("=" * 70)
    logger.info("Saliency Map Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()