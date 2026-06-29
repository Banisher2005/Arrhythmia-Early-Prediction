"""
Inference

Runs inference on the trained AMSRAN-GF model and generates
all evaluation metrics for the test dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataloaders import create_dataloaders
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import (
    compute_metrics,
    get_classification_report,
    print_metrics,
)
from src.evaluation.roc_curve import (
    compute_auc,
    plot_roc_curve,
    print_auc,
)
from src.models.amsran_gf import AMSRAN_GF
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = Path("models_saved") / "best_model.pt"

RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_FILE = RESULTS_DIR / "metrics.json"

REPORT_FILE = RESULTS_DIR / "classification_report.txt"

PREDICTIONS_FILE = RESULTS_DIR / "predictions.npy"

PROBABILITIES_FILE = RESULTS_DIR / "probabilities.npy"

# ==========================================================


def resolve_device(
    preference: str = "auto",
) -> torch.device:
    """
    Resolve a device preference for inference.

    Parameters
    ----------
    preference : str
        One of ``auto``, ``cpu`` or ``gpu``/``cuda``.

    Returns
    -------
    torch.device
        A valid PyTorch device. GPU requests gracefully fall back to CPU when
        CUDA is unavailable.
    """

    normalized = preference.strip().lower()

    if normalized in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning(
            "CUDA was requested but is unavailable. Falling back to CPU."
        )
        return torch.device("cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def extract_state_dict(
    checkpoint,
) -> dict:
    """
    Extract a model state dictionary from supported checkpoint formats.

    Supported formats are:
    - raw state_dict
    - dictionaries containing model_state_dict
    - dictionaries containing state_dict
    - dictionaries containing model
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a PyTorch state dict "
            "or a checkpoint dictionary."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "Checkpoint dictionary does not contain model weights. Expected one "
        "of: model_state_dict, state_dict, model, or a raw state_dict."
    )


def normalize_state_dict_keys(
    state_dict: dict,
) -> dict:
    """
    Normalize common checkpoint key prefixes.
    """

    if all(
        isinstance(key, str) and key.startswith("module.")
        for key in state_dict
    ):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    return state_dict


def load_trained_model(
    model_path: str | Path = MODEL_PATH,
    device: torch.device | None = None,
) -> AMSRAN_GF:
    """
    Load the trained AMSRAN-GF model for inference.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    if device is None:
        device = resolve_device()

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load checkpoint '{model_path}': {exc}"
        ) from exc

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    state_dict = normalize_state_dict_keys(
        extract_state_dict(
            checkpoint
        )
    )

    model = AMSRAN_GF().to(device)

    try:
        model.load_state_dict(
            state_dict
        )
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint weights are incompatible with AMSRAN-GF: {exc}"
        ) from exc

    model.eval()

    return model


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    signal: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run model inference for a tensor batch.

    The tensor must already follow the training data contract:
    ``(batch, channels, samples)``.
    """

    if signal.ndim != 3:
        raise ValueError(
            "Inference input must have shape (batch, channels, samples)."
        )

    outputs = model(
        signal.to(device)
    )

    probabilities = outputs["probabilities"]

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    return {
        **outputs,
        "predictions": predictions,
    }


@torch.no_grad()
def predict_single_beat(
    model: torch.nn.Module,
    beat: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run inference for one heartbeat segment.
    """

    if beat.ndim == 1:
        beat = beat.view(
            1,
            1,
            -1,
        )
    elif beat.ndim == 2:
        beat = beat.unsqueeze(0)

    return predict_batch(
        model=model,
        signal=beat,
        device=device,
    )


@torch.no_grad()
def run_inference():

    logger.info("=" * 70)
    logger.info("Running Inference")
    logger.info("=" * 70)

    device = resolve_device()

    logger.info(
        "Device : %s",
        device,
    )

    # ------------------------------------------------------

    model = load_trained_model(
        MODEL_PATH,
        device,
    )

    logger.info(
        "Loaded model from %s",
        MODEL_PATH,
    )

    # ------------------------------------------------------

    dataloaders = create_dataloaders()

    test_loader = dataloaders["test"]

    # ------------------------------------------------------

    predictions = []

    probabilities = []

    labels = []

    progress = tqdm(
        test_loader,
        desc="Inference",
    )

    for batch in progress:

        signal = batch["signal"].to(device)

        target = batch["label"].to(device)

        outputs = predict_batch(
            model=model,
            signal=signal,
            device=device,
        )

        probs = outputs["probabilities"]

        pred = outputs["predictions"]

        predictions.extend(
            pred.cpu().numpy()
        )

        probabilities.extend(
            probs.cpu().numpy()
        )

        labels.extend(
            target.cpu().numpy()
        )

    # ------------------------------------------------------

    predictions = np.asarray(predictions)

    probabilities = np.asarray(probabilities)

    labels = np.asarray(labels)

    # ------------------------------------------------------

    np.save(
        PREDICTIONS_FILE,
        predictions,
    )

    np.save(
        PROBABILITIES_FILE,
        probabilities,
    )

    logger.info(
        "Saved predictions."
    )

    # ------------------------------------------------------

    metrics = compute_metrics(
        labels,
        predictions,
    )

    print_metrics(
        metrics,
    )

    with open(
        METRICS_FILE,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    # ------------------------------------------------------

    report = get_classification_report(
        labels,
        predictions,
    )

    with open(
        REPORT_FILE,
        "w",
    ) as f:

        f.write(report)

    logger.info(
        "Saved classification report."
    )

    # ------------------------------------------------------

    plot_confusion_matrix(
        labels,
        predictions,
    )

    plot_confusion_matrix(
        labels,
        predictions,
        normalize=True,
        save_path=RESULTS_DIR /
        "confusion_matrix_normalized.png",
    )

    # ------------------------------------------------------

    roc_results = compute_auc(
        labels,
        probabilities,
    )

    print_auc(
        roc_results,
    )

    plot_roc_curve(
        roc_results,
    )

    logger.info("=" * 70)
    logger.info("Inference Complete")
    logger.info("=" * 70)


def main():

    run_inference()


if __name__ == "__main__":
    main()
