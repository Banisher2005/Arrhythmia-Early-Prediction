"""
Explainability Utilities

Common utility functions for ECG explainability methods including
Grad-CAM, Saliency Maps, and Integrated Gradients.

Features
--------
- ECG waveform plotting
- Attribution normalization
- Heatmap overlay
- Figure saving
- Probability bar plotting
- Directory management

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Directory Utilities
# ==========================================================

def ensure_directory(path: Path | str) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : Path or str
        Directory path.

    Returns
    -------
    Path
        Path object.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==========================================================
# Normalization
# ==========================================================

def normalize_attribution(
    attribution: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize attribution values to [0, 1].

    Parameters
    ----------
    attribution : np.ndarray
        Raw attribution map.

    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    np.ndarray
        Normalized attribution.
    """

    attribution = np.asarray(attribution, dtype=np.float32)

    attribution = np.abs(attribution)

    attribution -= attribution.min()

    attribution /= (attribution.max() + eps)

    return attribution


# ==========================================================
# ECG Plot
# ==========================================================

def plot_ecg(
    ecg: np.ndarray,
    title: str = "ECG Signal",
    figsize: tuple = (12, 4),
):
    """
    Plot ECG waveform.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.

    title : str
        Plot title.

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        ecg,
        linewidth=1.5,
    )

    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")

    ax.grid(True)

    fig.tight_layout()

    return fig


# ==========================================================
# Attribution Overlay
# ==========================================================

def plot_attribution(
    ecg: np.ndarray,
    attribution: np.ndarray,
    title: str = "Explainability",
    cmap: str = "jet",
    alpha: float = 0.75,
    figsize: tuple = (14, 4),
):
    """
    Overlay attribution scores on ECG waveform.

    Parameters
    ----------
    ecg : np.ndarray
        ECG waveform.

    attribution : np.ndarray
        Attribution values.

    title : str
        Figure title.

    cmap : str
        Colormap.

    alpha : float
        Transparency.

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """

    ecg = np.asarray(ecg).flatten()

    attribution = normalize_attribution(attribution)

    x = np.arange(len(ecg))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        x,
        ecg,
        color="black",
        linewidth=1.5,
        zorder=3,
    )

    ax.scatter(
        x,
        ecg,
        c=attribution,
        cmap=cmap,
        s=16,
        alpha=alpha,
        zorder=4,
    )

    cbar = plt.colorbar(
        ax.collections[0],
        ax=ax,
        pad=0.02,
    )

    cbar.set_label("Importance")

    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    ax.grid(True)

    fig.tight_layout()

    return fig


# ==========================================================
# Probability Plot
# ==========================================================

def plot_probabilities(
    probabilities: np.ndarray,
    class_names: list[str],
    title: str = "Prediction Probabilities",
):
    """
    Plot class probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Softmax probabilities.

    class_names : list[str]
        Class labels.

    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """

    probabilities = np.asarray(probabilities)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        class_names,
        probabilities,
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Probability")

    ax.set_title(title)

    ax.grid(axis="y")

    fig.tight_layout()

    return fig


# ==========================================================
# Figure Saving
# ==========================================================

def save_figure(
    figure,
    save_path: Path | str,
    dpi: int = 300,
):
    """
    Save matplotlib figure.

    Parameters
    ----------
    figure
        Matplotlib figure.

    save_path : Path or str
        Output image.

    dpi : int
        Image DPI.
    """

    save_path = Path(save_path)

    ensure_directory(save_path.parent)

    figure.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Saved figure -> %s",
        save_path,
    )


# ==========================================================
# Combined Convenience Function
# ==========================================================

def save_explainability_figure(
    ecg: np.ndarray,
    attribution: np.ndarray,
    output_path: Path | str,
    title: str,
):
    """
    Create and save an explainability visualization.

    Parameters
    ----------
    ecg : np.ndarray
        ECG waveform.

    attribution : np.ndarray
        Attribution scores.

    output_path : Path or str
        Destination image.

    title : str
        Figure title.
    """

    fig = plot_attribution(
        ecg=ecg,
        attribution=attribution,
        title=title,
    )

    save_figure(
        fig,
        output_path,
    )


# ==========================================================
# Module Test
# ==========================================================

def main():
    """
    Example utility usage.
    """

    logger.info("=" * 70)
    logger.info("Testing Explainability Utilities")
    logger.info("=" * 70)

    x = np.linspace(0, 6 * np.pi, 256)

    ecg = (
        np.sin(x)
        + 0.2 * np.sin(5 * x)
    )

    attribution = np.random.rand(256)

    output = (
        Path("results")
        / "gradcam"
        / "utility_test.png"
    )

    save_explainability_figure(
        ecg=ecg,
        attribution=attribution,
        output_path=output,
        title="Utility Test",
    )

    logger.info("Utility test completed successfully.")


if __name__ == "__main__":
    main()