"""Metric card components."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_metric_cards(metrics: dict[str, Any]) -> None:
    """Render headline model performance metrics."""

    values = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("weighted_precision", metrics.get("macro_precision"))),
        ("Recall", metrics.get("weighted_recall", metrics.get("macro_recall"))),
        ("F1 Score", metrics.get("weighted_f1", metrics.get("macro_f1"))),
    ]
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values):
        if isinstance(value, (int, float)):
            column.metric(label, f"{value:.2%}")
        else:
            column.metric(label, "Unavailable")

