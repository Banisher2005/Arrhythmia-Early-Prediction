"""Prediction summary component."""

from __future__ import annotations

import streamlit as st

from streamlit_app.utils.inference import PredictionResult


def render_prediction_card(result: PredictionResult, threshold: float) -> None:
    """Render the predicted class, confidence, and latency."""

    status = "Meets threshold" if result.confidence >= threshold else "Below threshold"
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", result.predicted_class)
    col_b.metric("Confidence", f"{result.confidence:.2%}", status)
    col_c.metric("Inference Time", f"{result.inference_time_ms:.1f} ms")

