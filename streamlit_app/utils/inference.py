"""Streamlit-safe wrappers around the existing AMSRAN-GF model and XAI APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from streamlit_app.config import CLASS_NAMES, MODEL_PATH
from streamlit_app.utils.loaders import validate_heartbeat_segment


@dataclass(frozen=True)
class PredictionResult:
    """Structured single-sample prediction result."""

    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: np.ndarray
    inference_time_ms: float
    attention_weights: np.ndarray | None
    gate_weights: list[np.ndarray]


def _import_torch_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import heavy ML modules lazily so non-ML pages remain responsive."""

    import torch

    from src.evaluation.inference import (
        load_trained_model,
        predict_single_beat,
        resolve_device,
    )
    from src.explainability.explainer import Explainer
    from src.explainability.report import ExplainabilityReport

    return (
        torch,
        load_trained_model,
        predict_single_beat,
        resolve_device,
        Explainer,
        ExplainabilityReport,
    )


def resolve_device(preference: str) -> Any:
    """Resolve a user device preference through the shared inference module."""

    _, _, _, shared_resolve_device, _, _ = _import_torch_modules()
    return shared_resolve_device(preference)


@st.cache_resource(show_spinner=False)
def load_model(device_preference: str) -> tuple[Any, Any]:
    """Load the existing AMSRAN-GF checkpoint through src.evaluation."""

    _, shared_load_model, _, _, _, _ = _import_torch_modules()
    device = resolve_device(device_preference)
    model = shared_load_model(MODEL_PATH, device)
    return model, device


def predict_signal(signal: np.ndarray, device_preference: str) -> PredictionResult:
    """Run single-window inference using the existing trained model."""

    torch, _, shared_predict_single_beat, _, _, _ = _import_torch_modules()
    model, device = load_model(device_preference)
    window = validate_heartbeat_segment(signal)
    tensor = torch.tensor(window, dtype=torch.float32, device=device).view(1, 1, -1)

    started = time.perf_counter()
    outputs = shared_predict_single_beat(model, tensor, device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    probabilities = outputs["probabilities"].detach().cpu().squeeze().numpy()
    predicted_index = int(np.argmax(probabilities))
    attention = outputs.get("attention_weights")
    if attention is not None:
        attention_array = attention.detach().cpu().squeeze().numpy()
    else:
        attention_array = None

    gate_arrays = [
        gate.detach().cpu().squeeze().numpy()
        for gate in outputs.get("gate_weights", [])
    ]

    return PredictionResult(
        predicted_class=list(CLASS_NAMES)[predicted_index],
        predicted_index=predicted_index,
        confidence=float(probabilities[predicted_index]),
        probabilities=probabilities,
        inference_time_ms=elapsed_ms,
        attention_weights=attention_array,
        gate_weights=gate_arrays,
    )


def generate_explanation(
    signal: np.ndarray,
    method: str,
    device_preference: str,
) -> dict[str, Any]:
    """Generate one explainability map through the existing Explainer API."""

    torch, _, _, _, Explainer, _ = _import_torch_modules()
    model, device = load_model(device_preference)
    window = validate_heartbeat_segment(signal)
    tensor = torch.tensor(window, dtype=torch.float32, device=device).view(1, 1, -1)
    explainer = Explainer(model=model, method=method, device=device)
    result = explainer.explain(ecg=tensor, show=False)
    return dict(result)


def generate_report(
    signal: np.ndarray,
    output_directory: Path,
    device_preference: str,
) -> dict[str, Any]:
    """Generate a full explainability report with the existing report generator."""

    torch, _, _, _, _, ExplainabilityReport = _import_torch_modules()
    model, device = load_model(device_preference)
    window = validate_heartbeat_segment(signal)
    tensor = torch.tensor(window, dtype=torch.float32, device=device).view(1, 1, -1)
    report = ExplainabilityReport(
        model=model,
        class_names=list(CLASS_NAMES),
        device=device,
    )
    return dict(report.generate(ecg=tensor, output_directory=str(output_directory)))
