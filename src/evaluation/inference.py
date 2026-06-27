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


@torch.no_grad()
def run_inference():

    logger.info("=" * 70)
    logger.info("Running Inference")
    logger.info("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "Device : %s",
        device,
    )

    # ------------------------------------------------------

    model = AMSRAN_GF().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

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

        outputs = model(signal)

        logits = outputs["logits"]

        probs = torch.softmax(
            logits,
            dim=1,
        )

        pred = torch.argmax(
            probs,
            dim=1,
        )

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