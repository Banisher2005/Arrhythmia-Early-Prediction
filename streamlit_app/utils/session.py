"""Session-state helpers for the Streamlit application."""

from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.config import DEFAULT_SETTINGS


def initialize_session_state() -> None:
    """Create all app-level session keys with production defaults."""

    defaults: dict[str, Any] = {
        "device": DEFAULT_SETTINGS.device,
        "confidence_threshold": DEFAULT_SETTINGS.confidence_threshold,
        "batch_size": DEFAULT_SETTINGS.batch_size,
        "theme": DEFAULT_SETTINGS.theme,
        "ecg_signal": None,
        "ecg_filename": None,
        "prediction_result": None,
        "explainability_result": None,
        "last_report_dir": None,
        "generated_report": None,
        "explanation_cache": {},
        "batch_results": None,
        "batch_errors": [],
        "raw_signal": None,
        "segmented_beats": None,
        "segmented_labels": None,
        "segmented_encoded": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def update_setting(key: str, value: Any) -> None:
    """Persist a setting value in Streamlit session state."""

    st.session_state[key] = value


def clear_analysis_state() -> None:
    """Clear uploaded signal, inference, and explainability state."""

    for key in (
        "ecg_signal",
        "ecg_filename",
        "prediction_result",
        "explainability_result",
        "last_report_dir",
        "generated_report",
        "explanation_cache",
        "batch_results",
        "batch_errors",
        "raw_signal",
        "segmented_beats",
        "segmented_labels",
        "segmented_encoded",
    ):
        st.session_state[key] = None
