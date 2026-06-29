"""
Explainability Report Generator

Creates a complete explainability report for a single ECG.

Outputs
-------
results/
└── explainability/
    └── sample_xxx/
        ├── report.json
        ├── waveform.png
        ├── gradcam++.png
        ├── smoothgrad.png
        ├── integrated_gradients.png
        ├── saliency.png
        ├── attention.png
        ├── gate_weights.png
        └── probability.png

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.explainability.explainer import Explainer
from src.explainability.visualizer import ExplainabilityVisualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityReport:

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        device: Optional[torch.device] = None,
    ):

        self.model = model
        self.model.eval()

        self.class_names = class_names

        self.device = (
            device
            if device is not None
            else next(model.parameters()).device
        )

        self.visualizer = ExplainabilityVisualizer()

    ###########################################################################

    def _top_predictions(
        self,
        probabilities: np.ndarray,
        top_k: int = 3,
    ):

        indices = np.argsort(
            probabilities
        )[::-1][:top_k]

        results = []

        for idx in indices:

            results.append(
                {
                    "class": self.class_names[idx],
                    "probability": float(
                        probabilities[idx]
                    ),
                }
            )

        return results

    ###########################################################################

    def generate(
        self,
        ecg: torch.Tensor,
        output_directory: str,
        ground_truth: Optional[str] = None,
    ):

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        #######################################################################
        # Model Prediction
        #######################################################################

        with torch.no_grad():

            outputs = self.model(
                ecg.to(
                    self.device
                )
            )

        probabilities = (
            outputs["probabilities"]
            .squeeze()
            .cpu()
            .numpy()
        )

        prediction = int(
            np.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[
                prediction
            ]
        )

        #######################################################################
        # Explainability Methods
        #######################################################################

        gradcam = Explainer(
            self.model,
            method="gradcam++",
            device=self.device,
        )

        smoothgrad = Explainer(
            self.model,
            method="smoothgrad",
            device=self.device,
        )

        saliency = Explainer(
            self.model,
            method="saliency",
            device=self.device,
        )

        ig = Explainer(
            self.model,
            method="integrated_gradients",
            device=self.device,
        )

        gradcam_result = gradcam.explain(
            ecg,
            save_path=str(
                output_directory
                / "gradcam++.png"
            ),
            show=False,
        )

        smoothgrad_result = smoothgrad.explain(
            ecg,
            save_path=str(
                output_directory
                / "smoothgrad.png"
            ),
            show=False,
        )

        saliency_result = saliency.explain(
            ecg,
            save_path=str(
                output_directory
                / "saliency.png"
            ),
            show=False,
        )

        ig_result = ig.explain(
            ecg,
            save_path=str(
                output_directory
                / "integrated_gradients.png"
            ),
            show=False,
        )
        #######################################################################
        # Attention Visualization
        #######################################################################

        attention = (
            outputs["attention_weights"]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        self.visualizer.waveform_with_attention(
            ecg=ecg.squeeze().cpu().numpy(),
            attention=attention,
            prediction=self.class_names[prediction],
            confidence=confidence,
            save_path=str(
                output_directory
                / "attention.png"
            ),
            show=False,
        )

        #######################################################################
        # Adaptive Gate Visualization
        #######################################################################

        gate_weights = []

        for gate in outputs["gate_weights"]:

            gate_weights.append(
                gate.squeeze()
                .detach()
                .cpu()
                .numpy()
            )

        gate_weights = np.asarray(
            gate_weights
        )

        self.visualizer.gate_weight_plot(
            gate_weights=gate_weights,
            save_path=str(
                output_directory
                / "gate_weights.png"
            ),
            show=False,
        )

        #######################################################################
        # Probability Plot
        #######################################################################

        self.visualizer.probability_plot(
            probabilities=probabilities,
            class_names=self.class_names,
            save_path=str(
                output_directory
                / "probabilities.png"
            ),
            show=False,
        )

        #######################################################################
        # Combined Comparison Figure
        #######################################################################

        explanation_maps = {

            "GradCAM++":
                gradcam_result[
                    "gradcam_pp"
                ],

            "SmoothGrad":
                smoothgrad_result[
                    "smoothgrad"
                ],

            "Integrated Gradients":
                ig_result[
                    "integrated_gradients"
                ],

            "Saliency":
                saliency_result[
                    "saliency"
                ],

        }

        self.visualizer.comparison_grid(
            ecg=ecg.squeeze().cpu().numpy(),
            explanations=explanation_maps,
            prediction=self.class_names[
                prediction
            ],
            confidence=confidence,
            ground_truth=ground_truth,
            save_path=str(
                output_directory
                / "comparison.png"
            ),
            show=False,
        )

        #######################################################################
        # JSON Report
        #######################################################################

        report = {

            "prediction": self.class_names[
                prediction
            ],

            "prediction_index":
                prediction,

            "confidence":
                confidence,

            "ground_truth":
                ground_truth,

            "top_predictions":
                self._top_predictions(
                    probabilities,
                    top_k=3,
                ),

            "generated_files": {

                "gradcam++":
                    "gradcam++.png",

                "smoothgrad":
                    "smoothgrad.png",

                "saliency":
                    "saliency.png",

                "integrated_gradients":
                    "integrated_gradients.png",

                "attention":
                    "attention.png",

                "gate_weights":
                    "gate_weights.png",

                "comparison":
                    "comparison.png",

                "probabilities":
                    "probabilities.png",

            }

        }

        with open(

            output_directory
            / "report.json",

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        logger.info(
            "Explainability report saved to %s",
            output_directory,
        )

        return report
###############################################################################
# Standalone Test
###############################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing Explainability Report Generator")
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

    class_names = [
        "N",
        "S",
        "V",
        "F",
        "Q",
    ]

    generator = ExplainabilityReport(
        model=model,
        class_names=class_names,
        device=device,
    )

    report = generator.generate(
        ecg=ecg,
        output_directory=(
            "results/explainability/"
            "example"
        ),
        ground_truth="N",
    )

    logger.info(
        "Prediction : %s",
        report["prediction"],
    )

    logger.info(
        "Confidence : %.4f",
        report["confidence"],
    )

    logger.info(
        "Top Predictions:"
    )

    for item in report["top_predictions"]:

        logger.info(
            "%s : %.4f",
            item["class"],
            item["probability"],
        )

    logger.info("=" * 70)
    logger.info("Explainability Report Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()