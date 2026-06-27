"""
Explainability Visualizer

Unified visualization module for all explainability methods.

Supports

- Grad-CAM
- Grad-CAM++
- Saliency
- SmoothGrad
- Integrated Gradients

Can generate publication-quality figures.

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class ExplainabilityVisualizer:

    def __init__(
        self,
        dpi: int = 300,
    ):

        self.dpi = dpi

    ###########################################################################

    @staticmethod
    def _prepare_axis(
        ax,
        title: str,
    ):

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Samples"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(True)

    ###########################################################################

    def single_plot(
        self,
        ecg: np.ndarray,
        attribution: np.ndarray,
        title: str,
        cmap: str = "jet",
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):

        fig, ax = plt.subplots(
            figsize=(14,4)
        )

        x = np.arange(
            len(ecg)
        )

        ax.plot(
            x,
            ecg,
            color="black",
            linewidth=1.5,
            zorder=2,
        )

        scatter = ax.scatter(
            x,
            ecg,
            c=attribution,
            cmap=cmap,
            s=18,
            zorder=3,
        )

        self._prepare_axis(
            ax,
            title,
        )

        if prediction is not None:

            ax.text(
                0.01,
                0.98,
                f"Prediction : {prediction}\nConfidence : {confidence:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                ),
            )

        plt.colorbar(
            scatter,
            ax=ax,
            label="Importance",
        )

        plt.tight_layout()

        if save_path is not None:

            save_path = Path(
                save_path
            )

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=self.dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()

    ###########################################################################

    def compare(
        self,
        ecg: np.ndarray,
        maps: dict[str, np.ndarray],
        prediction: str,
        confidence: float,
        save_path: Optional[str] = None,
        show: bool = True,
    ):

        methods = list(
            maps.keys()
        )

        fig, axes = plt.subplots(
            len(methods),
            1,
            figsize=(
                14,
                3 * len(methods),
            ),
            sharex=True,
        )

        if len(methods) == 1:
            axes = [axes]

        x = np.arange(
            len(ecg)
        )

        for ax, method in zip(
            axes,
            methods,
        ):

            scatter = ax.scatter(
                x,
                ecg,
                c=maps[method],
                cmap="jet",
                s=18,
            )

            ax.plot(
                x,
                ecg,
                color="black",
                linewidth=1.4,
            )

            ax.set_title(
                method
            )

            ax.grid(True)

            plt.colorbar(
                scatter,
                ax=ax,
            )

        axes[0].text(
            0.01,
            1.15,
            f"Prediction : {prediction}    Confidence : {confidence:.4f}",
            transform=axes[0].transAxes,
            fontsize=11,
        )

        plt.tight_layout()
        if save_path is not None:

            save_path = Path(
                save_path
            )

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=self.dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()

    ###########################################################################

    def comparison_grid(
        self,
        ecg: np.ndarray,
        explanations: dict[str, np.ndarray],
        prediction: str,
        confidence: float,
        ground_truth: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Publication-quality comparison grid.
        """

        methods = list(
            explanations.keys()
        )

        n = len(methods)

        cols = 2

        rows = int(
            np.ceil(
                n / cols
            )
        )

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(
                15,
                4 * rows,
            ),
            sharex=True,
        )

        axes = np.asarray(
            axes
        ).reshape(-1)

        x = np.arange(
            len(ecg)
        )

        for idx, method in enumerate(
            methods
        ):

            ax = axes[idx]

            scatter = ax.scatter(
                x,
                ecg,
                c=explanations[
                    method
                ],
                cmap="jet",
                s=18,
                zorder=3,
            )

            ax.plot(
                x,
                ecg,
                color="black",
                linewidth=1.4,
                zorder=2,
            )

            self._prepare_axis(
                ax,
                method,
            )

            plt.colorbar(
                scatter,
                ax=ax,
            )

        for idx in range(
            len(methods),
            len(axes),
        ):

            axes[idx].axis(
                "off"
            )

        title = (
            f"Prediction : {prediction}"
            f"    Confidence : {confidence:.4f}"
        )

        if ground_truth is not None:

            title += (
                f"    Ground Truth : {ground_truth}"
            )

        fig.suptitle(
            title,
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        plt.subplots_adjust(
            top=0.90,
        )

        if save_path is not None:

            save_path = Path(
                save_path
            )

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=self.dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()

    ###########################################################################

    def probability_plot(
        self,
        probabilities: np.ndarray,
        class_names: list[str],
        save_path: Optional[str] = None,
        show: bool = True,
    ):

        fig = plt.figure(
            figsize=(8,4)
        )

        plt.bar(
            class_names,
            probabilities,
        )

        plt.ylim(
            0,
            1,
        )

        plt.ylabel(
            "Probability"
        )

        plt.title(
            "Prediction Probabilities"
        )

        plt.grid(
            axis="y"
        )

        plt.tight_layout()

        if save_path is not None:

            save_path = Path(
                save_path
            )

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=self.dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()
    ###########################################################################

    def waveform_with_attention(
        self,
        ecg: np.ndarray,
        attention: np.ndarray,
        prediction: str,
        confidence: float,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Visualize temporal attention weights.
        """

        x = np.arange(len(ecg))

        fig, ax1 = plt.subplots(
            figsize=(14, 4)
        )

        ax1.plot(
            x,
            ecg,
            color="black",
            linewidth=1.5,
            label="ECG",
        )

        ax1.set_xlabel("Samples")
        ax1.set_ylabel("Amplitude")

        ax2 = ax1.twinx()

        ax2.plot(
            x,
            attention,
            color="red",
            linewidth=2,
            alpha=0.75,
            label="Attention",
        )

        ax2.set_ylabel("Attention Weight")

        plt.title(
            f"Temporal Attention\nPrediction={prediction} | Confidence={confidence:.4f}"
        )

        fig.tight_layout()

        if save_path is not None:

            save_path = Path(save_path)

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=self.dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()

    ###########################################################################

    def gate_weight_plot(
        self,
        gate_weights: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Visualize Adaptive Gate branch contributions.
        """

        if gate_weights.ndim == 3:
            gate_weights = gate_weights.mean(axis=-1)

        gate_mean = gate_weights.mean(axis=0)

        labels = [
            "Kernel-3",
            "Kernel-5",
            "Kernel-7",
        ]

        plt.figure(
            figsize=(6, 4)
        )

        plt.bar(
            labels,
            gate_mean,
        )

        plt.ylabel(
            "Average Weight"
        )

        plt.title(
            "Adaptive Gate Branch Importance"
        )

        plt.grid(
            axis="y"
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
                dpi=self.dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()


###############################################################################
# Standalone Test
###############################################################################

def main():

    visualizer = ExplainabilityVisualizer()

    ecg = np.sin(
        np.linspace(
            0,
            8 * np.pi,
            180,
        )
    )

    explanation = np.random.rand(
        180,
    )

    visualizer.single_plot(
        ecg=ecg,
        attribution=explanation,
        title="Grad-CAM++",
        prediction="V",
        confidence=0.9812,
        show=False,
    )

    visualizer.comparison_grid(
        ecg=ecg,
        explanations={
            "Grad-CAM++": explanation,
            "SmoothGrad": np.random.rand(180),
            "Integrated Gradients": np.random.rand(180),
            "Saliency": np.random.rand(180),
        },
        prediction="V",
        confidence=0.9812,
        ground_truth="V",
        show=False,
    )

    visualizer.waveform_with_attention(
        ecg=ecg,
        attention=np.random.rand(180),
        prediction="V",
        confidence=0.9812,
        show=False,
    )

    visualizer.gate_weight_plot(
        gate_weights=np.random.rand(3, 180),
        show=False,
    )


if __name__ == "__main__":
    main()