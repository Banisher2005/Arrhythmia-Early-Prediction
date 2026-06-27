"""
Research-Grade Integrated Gradients

Supports

- Zero Baseline
- Mean Baseline
- Median Baseline
- Random Baseline
- Custom Baseline
- Multi-Baseline Averaging

Reference
---------
Sundararajan et al.
Axiomatic Attribution for Deep Networks
(ICML 2017)

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.explainability.attribution import AttributionMethod
from src.utils.logger import get_logger

logger = get_logger(__name__)


BaselineType = Literal[
    "zero",
    "mean",
    "median",
    "random",
    "custom",
]


class IntegratedGradients(AttributionMethod):

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        steps: int = 100,
        baseline_type: BaselineType = "zero",
        average_random_baselines: int = 8,
    ):

        super().__init__(
            model=model,
            device=device,
        )

        self.steps = steps
        self.baseline_type = baseline_type
        self.average_random_baselines = average_random_baselines

    ###########################################################################

    def zero_baseline(
        self,
        ecg: torch.Tensor,
    ):

        return torch.zeros_like(ecg)

    ###########################################################################

    def mean_baseline(
        self,
        ecg: torch.Tensor,
    ):

        mean = ecg.mean(
            dim=-1,
            keepdim=True,
        )

        return mean.expand_as(ecg)

    ###########################################################################

    def median_baseline(
        self,
        ecg: torch.Tensor,
    ):

        median = ecg.median(
            dim=-1,
            keepdim=True,
        ).values

        return median.expand_as(ecg)

    ###########################################################################

    def random_baseline(
        self,
        ecg: torch.Tensor,
    ):

        low = ecg.min()

        high = ecg.max()

        return torch.rand_like(
            ecg
        ) * (
            high - low
        ) + low

    ###########################################################################

    def build_baseline(
        self,
        ecg: torch.Tensor,
        custom_baseline: Optional[
            torch.Tensor
        ] = None,
    ):

        if self.baseline_type == "zero":
            return self.zero_baseline(ecg)

        if self.baseline_type == "mean":
            return self.mean_baseline(ecg)

        if self.baseline_type == "median":
            return self.median_baseline(ecg)

        if self.baseline_type == "random":
            return self.random_baseline(ecg)

        if self.baseline_type == "custom":

            if custom_baseline is None:
                raise ValueError(
                    "Custom baseline required."
                )

            return custom_baseline.to(
                self.device
            )

        raise ValueError(
            "Unknown baseline."
        )

    ###########################################################################

    def integrated_gradient(
        self,
        ecg: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
    ):

        accumulated_gradients = torch.zeros_like(
            ecg
        )

        alphas = torch.linspace(
            0,
            1,
            self.steps,
            device=self.device,
        )

        for alpha in alphas:

            sample = baseline + alpha * (
                ecg - baseline
            )

            sample.requires_grad_(True)

            self.zero_grad(
                self.model
            )

            outputs = self.model(
                sample
            )

            score = outputs[
                "logits"
            ][
                :,
                target_class,
            ]

            score.backward()

            accumulated_gradients += (
                sample.grad
            )

        avg_grad = (
            accumulated_gradients
            / self.steps
        )

        attribution = (
            ecg - baseline
        ) * avg_grad

        return attribution

    ###########################################################################

    def generate(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
        custom_baseline: Optional[
            torch.Tensor
        ] = None,
    ):
        ecg = self.prepare_input(
            ecg,
            self.device,
        )

        with torch.no_grad():

            outputs = self.model(
                ecg
            )

            prediction, confidence = (
                self.prediction(
                    outputs
                )
            )

        if target_class is None:
            target_class = prediction

        #######################################################################
        # Random Baseline Averaging
        #######################################################################

        if self.baseline_type == "random":

            attribution = torch.zeros_like(
                ecg
            )

            for _ in range(
                self.average_random_baselines
            ):

                baseline = self.random_baseline(
                    ecg
                )

                attribution += self.integrated_gradient(
                    ecg,
                    baseline,
                    target_class,
                )

            attribution /= (
                self.average_random_baselines
            )

        #######################################################################
        # Other Baselines
        #######################################################################

        else:

            baseline = self.build_baseline(
                ecg,
                custom_baseline,
            )

            attribution = self.integrated_gradient(
                ecg,
                baseline,
                target_class,
            )

        attribution = (
            attribution.squeeze()
            .detach()
            .abs()
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
        custom_baseline: Optional[
            torch.Tensor
        ] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):

        attribution, prediction, confidence = (
            self.generate(
                ecg=ecg,
                target_class=target_class,
                custom_baseline=custom_baseline,
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
                f"Integrated Gradients | "
                f"Prediction={prediction} | "
                f"Confidence={confidence:.4f}"
            ),
            cmap="viridis",
            save_path=save_path,
            show=show,
        )

        return {
            "integrated_gradients": attribution,
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

            ig, pred, conf = self.generate(
                sample.unsqueeze(0),
                target_class=target_class,
            )

            maps.append(
                ig
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


###############################################################################
# Standalone Test
###############################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing Research Integrated Gradients")
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

    explainer = IntegratedGradients(
        model=model,
        device=device,
        baseline_type="random",
        steps=100,
        average_random_baselines=8,
    )

    result = explainer.explain(
        ecg=ecg,
        save_path=(
            "results/integrated_gradients/"
            "research_example.png"
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
        "Attribution Shape : %s",
        result["integrated_gradients"].shape,
    )

    logger.info("=" * 70)
    logger.info("Research Integrated Gradients Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()    
        
        