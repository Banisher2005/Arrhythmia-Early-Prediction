"""Plotting helpers for ECG, metrics, and explainability views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


CLINICAL_BLUE = "#2563eb"
CLINICAL_GRAY = "#64748b"


def ecg_waveform_figure(signal: np.ndarray, sampling_rate: int) -> go.Figure:
    """Build an interactive ECG waveform figure."""

    signal = np.asarray(signal, dtype=float).reshape(-1)
    time_axis = np.arange(signal.size) / float(sampling_rate)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal,
            mode="lines",
            line={"color": CLINICAL_BLUE, "width": 1.8},
            name="ECG",
        )
    )
    figure.update_layout(
        height=360,
        margin={"l": 24, "r": 24, "t": 36, "b": 24},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
    )
    return figure


def probability_bar_figure(probabilities: np.ndarray, class_names: list[str]) -> go.Figure:
    """Build a class probability bar chart."""

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    figure = go.Figure(
        go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=CLINICAL_BLUE,
            text=[f"{value:.1%}" for value in probabilities],
            textposition="auto",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        xaxis_title="AAMI Class",
        yaxis_title="Probability",
        template="plotly_white",
    )
    return figure


def attribution_figure(signal: np.ndarray, attribution: np.ndarray, title: str) -> go.Figure:
    """Build an ECG plot colored by explanation intensity."""

    signal = np.asarray(signal, dtype=float).reshape(-1)
    attribution = np.asarray(attribution, dtype=float).reshape(-1)
    if attribution.size != signal.size:
        attribution = np.interp(
            np.linspace(0, attribution.size - 1, signal.size),
            np.arange(attribution.size),
            attribution,
        )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.arange(signal.size),
            y=signal,
            mode="lines",
            line={"color": "#111827", "width": 1.2},
            name="ECG",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=np.arange(signal.size),
            y=signal,
            mode="markers",
            marker={
                "size": 6,
                "color": attribution,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "Importance"},
            },
            name=title,
        )
    )
    figure.update_layout(
        title=title,
        height=360,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        template="plotly_white",
    )
    return figure


def attention_figure(signal: np.ndarray, attention: np.ndarray) -> go.Figure:
    """Build a dual-axis temporal attention plot."""

    signal = np.asarray(signal, dtype=float).reshape(-1)
    attention = np.asarray(attention, dtype=float).reshape(-1)
    if attention.size != signal.size:
        attention = np.interp(
            np.linspace(0, attention.size - 1, signal.size),
            np.arange(attention.size),
            attention,
        )
    x_axis = np.arange(signal.size)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=x_axis, y=signal, mode="lines", name="ECG", line={"color": "#111827"})
    )
    figure.add_trace(
        go.Scatter(
            x=x_axis,
            y=attention,
            mode="lines",
            name="Attention",
            yaxis="y2",
            line={"color": "#dc2626", "width": 2},
        )
    )
    figure.update_layout(
        height=340,
        margin={"l": 24, "r": 24, "t": 32, "b": 24},
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        yaxis2={"title": "Attention", "overlaying": "y", "side": "right"},
        template="plotly_white",
    )
    return figure


def gate_figure(gate_weights: list[np.ndarray] | np.ndarray) -> go.Figure:
    """Build an adaptive gate contribution chart."""

    gates = np.asarray(gate_weights, dtype=float)
    if gates.ndim >= 3:
        values = gates.mean(axis=tuple(range(1, gates.ndim)))
    elif gates.ndim == 2:
        values = gates.mean(axis=1)
    else:
        values = gates.reshape(-1)
    labels = [f"Gate {index}" for index in range(1, len(values) + 1)]
    figure = go.Figure(go.Bar(x=labels, y=values, marker_color=CLINICAL_BLUE))
    figure.update_layout(
        height=300,
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
        yaxis_title="Average Weight",
        template="plotly_white",
    )
    return figure


def training_curve_figure(history: dict[str, list[float]]) -> go.Figure:
    """Build training and validation metric curves."""

    figure = go.Figure()
    for key, values in history.items():
        figure.add_trace(go.Scatter(y=values, mode="lines+markers", name=key.replace("_", " ").title()))
    figure.update_layout(
        height=380,
        margin={"l": 24, "r": 24, "t": 32, "b": 24},
        xaxis_title="Epoch",
        yaxis_title="Value",
        template="plotly_white",
    )
    return figure


def benchmark_comparison_figure(
    rows: list[dict[str, object]],
    metric: str,
    title: str,
) -> go.Figure:
    """Build a method comparison chart from saved benchmark rows."""

    methods = [str(row.get("method", "Unknown")) for row in rows]
    values = [
        float(row.get(metric) or 0.0)
        for row in rows
    ]
    figure = go.Figure(
        go.Bar(
            x=methods,
            y=values,
            marker_color=CLINICAL_BLUE,
            text=[f"{value:.2f}" for value in values],
            textposition="auto",
        )
    )
    figure.update_layout(
        title=title,
        height=320,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        xaxis_title="Explainability Method",
        yaxis_title=title,
        template="plotly_white",
    )
    return figure


def image_exists(path: Path) -> bool:
    """Return whether an image artifact exists."""

    return path.exists() and path.is_file()
