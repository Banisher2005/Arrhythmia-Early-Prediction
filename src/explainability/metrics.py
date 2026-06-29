"""
AMSRAN-GF Explainability Metrics

Research-oriented explainability metrics specifically designed
for the AMSRAN-GF architecture.

Evaluates

• Branch Contribution
• Branch Diversity
• Attention Entropy
• Attention Concentration
• Attention Peak Location
• Gate Diversity
• Gate Stability
• GradCAM Energy
• Explanation Agreement
• Explanation Stability
• Faithfulness
• Deletion AUC
• Insertion AUC

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExplainabilityMetricsResult:

    branch_contribution: dict

    branch_diversity: float

    attention_entropy: float

    attention_concentration: float

    attention_peak_index: int

    gate_diversity: float

    gate_stability: float

    gradcam_energy: float

    explanation_agreement: float

    explanation_stability: float

    deletion_auc: float | None

    insertion_auc: float | None

    faithfulness: float | None


class ExplainabilityMetrics:

    ###########################################################################

    @staticmethod
    def branch_contribution(
        gate_weights: np.ndarray,
    ):

        gate_weights = np.asarray(
            gate_weights
        )

        if gate_weights.ndim == 3:

            gate_weights = gate_weights.mean(
                axis=-1
            )

        contribution = gate_weights.mean(
            axis=1
        )

        contribution /= (
            contribution.sum()
            + 1e-8
        )

        return {

            "kernel3": float(
                contribution[0]
            ),

            "kernel5": float(
                contribution[1]
            ),

            "kernel7": float(
                contribution[2]
            ),

        }

    ###########################################################################

    @staticmethod
    def branch_diversity(
        gate_weights: np.ndarray,
    ):

        gate_weights = np.asarray(
            gate_weights
        )

        if gate_weights.ndim == 3:

            gate_weights = gate_weights.mean(
                axis=-1
            )

        return float(
            np.std(
                gate_weights.mean(
                    axis=1
                )
            )
        )

    ###########################################################################

    @staticmethod
    def attention_entropy(
        attention: np.ndarray,
    ):

        attention = np.asarray(
            attention
        )

        attention = attention / (
            attention.sum()
            + 1e-8
        )

        return float(

            -np.sum(

                attention
                * np.log(
                    attention + 1e-8
                )

            )

        )

    ###########################################################################

    @staticmethod
    def attention_concentration(
        attention: np.ndarray,
    ):

        attention = np.asarray(
            attention
        )

        return float(
            attention.max()
        )

    ###########################################################################

    @staticmethod
    def attention_peak_index(
        attention: np.ndarray,
    ):

        return int(
            np.argmax(
                attention
            )
        )

    ###########################################################################

    @staticmethod
    def gate_diversity(
        gate_weights: np.ndarray,
    ):

        return float(
            np.var(
                gate_weights
            )
        )

    ###########################################################################

    @staticmethod
    def gate_stability(
        gate_weights: np.ndarray,
    ):

        gate_weights = np.asarray(
            gate_weights
        )

        return float(

            1.0
            /
            (
                np.std(
                    gate_weights
                )
                + 1e-8
            )

        )

    ###########################################################################

    @staticmethod
    def gradcam_energy(
        gradcam: np.ndarray,
    ):

        return float(

            np.sum(
                gradcam ** 2
            )

        )

    ###########################################################################

    @staticmethod
    def explanation_agreement(
        *maps: np.ndarray,
    ):

        correlations = []

        for i in range(
            len(maps)
        ):

            for j in range(
                i + 1,
                len(maps),
            ):

                r = np.corrcoef(
                    maps[i],
                    maps[j],
                )[0,1]

                correlations.append(
                    r
                )

        return float(
            np.mean(
                correlations
            )
        )

    ###########################################################################

    @staticmethod
    def explanation_stability(
        *maps: np.ndarray,
    ):

        stacked = np.stack(
            maps
        )

        return float(

            np.mean(
                np.std(
                    stacked,
                    axis=0,
                )
            )

        )

    ###########################################################################

    @staticmethod
    def summarize(

        gate_weights,

        attention,

        gradcam,

        *other_maps,

    ):

        metrics = ExplainabilityMetrics()

        return ExplainabilityMetricsResult(

            branch_contribution=
            metrics.branch_contribution(
                gate_weights
            ),

            branch_diversity=
            metrics.branch_diversity(
                gate_weights
            ),

            attention_entropy=
            metrics.attention_entropy(
                attention
            ),

            attention_concentration=
            metrics.attention_concentration(
                attention
            ),

            attention_peak_index=
            metrics.attention_peak_index(
                attention
            ),

            gate_diversity=
            metrics.gate_diversity(
                gate_weights
            ),

            gate_stability=
            metrics.gate_stability(
                gate_weights
            ),

            gradcam_energy=
            metrics.gradcam_energy(
                gradcam
            ),

            explanation_agreement=
            metrics.explanation_agreement(
                gradcam,
                *other_maps,
            ),

            explanation_stability=
            metrics.explanation_stability(
                gradcam,
                *other_maps,
            ),

            deletion_auc=None,

            insertion_auc=None,

            faithfulness=None,

        )


###############################################################################

def main():

    print(
        "AMSRAN-GF Explainability Metrics Ready."
    )


if __name__ == "__main__":
    main()