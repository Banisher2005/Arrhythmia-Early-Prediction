"""Streamlit entry point for the AMSRAN-GF ECG analysis application."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.footer import render_footer
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.ui import (
    inject_clinical_styles,
    render_hero,
    render_navigation_cards,
    status_badge,
)
from streamlit_app.config import (
    APP_SUBTITLE,
    APP_TITLE,
    CLASS_NAMES,
    MODEL_PATH,
    PROCESSED_DATASET_PATH,
    RESULTS_DIR,
    SAMPLING_RATE,
    SEGMENT_LENGTH,
    TRAINING_HISTORY_PATH,
)
from streamlit_app.utils.inference import resolve_device
from streamlit_app.utils.loaders import load_dataset_summary, read_json
from streamlit_app.utils.session import initialize_session_state


def configure_page() -> None:
    """Configure Streamlit page metadata."""

    st.set_page_config(
        page_title="AMSRAN-GF ECG Analysis",
        page_icon="ECG",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_global_styles() -> None:
    """Inject minimal clinical UI styling."""

    inject_clinical_styles()


def render_home() -> None:
    """Render the home page."""

    checkpoint_loaded = MODEL_PATH.exists()
    try:
        device_label = str(resolve_device(st.session_state.device)).upper()
    except Exception:
        device_label = "UNAVAILABLE"

    render_hero(
        "Clinical research dashboard",
        APP_TITLE,
        APP_SUBTITLE,
        [
            status_badge(f"Device: {device_label}", "ok"),
            status_badge(
                "Checkpoint loaded" if checkpoint_loaded else "Checkpoint missing",
                "ok" if checkpoint_loaded else "bad",
            ),
        ],
    )

    metrics = {}
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = read_json(metrics_path)
        except ValueError as exc:
            st.warning(str(exc))

    dataset_summary = load_dataset_summary(PROCESSED_DATASET_PATH)
    total_beats = dataset_summary.get("total_beats")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Accuracy", format_percent(metrics.get("accuracy")))
    metric_columns[1].metric("Weighted F1", format_percent(metrics.get("weighted_f1")))
    metric_columns[2].metric("Macro F1", format_percent(metrics.get("macro_f1")))
    metric_columns[3].metric("Classes", str(len(CLASS_NAMES)))
    metric_columns[4].metric(
        "ECG Beats",
        f"{total_beats:,}" if isinstance(total_beats, int) else "Unavailable",
    )
    metric_columns[5].metric("Checkpoint", "Ready" if checkpoint_loaded else "Missing")

    overview, dataset = st.columns([1.25, 1])
    with overview:
        st.subheader("AMSRAN-GF Architecture")
        st.write(
            "Multi-scale residual CNN feature extraction feeds adaptive gated "
            "fusion, BiLSTM sequence modeling, temporal attention, and a "
            "five-class AAMI classifier head."
        )
        st.info(
            "This app is a presentation layer over the existing research code. "
            "It does not retrain, alter checkpoints, or replace preprocessing."
        )
    with dataset:
        st.subheader("MIT-BIH Dataset Summary")
        st.write(f"Sampling rate: **{SAMPLING_RATE} Hz**")
        st.write(f"Beat window: **{SEGMENT_LENGTH} samples**")
        st.write(f"Classes: **{', '.join(CLASS_NAMES)}**")
        st.write("Evaluation protocol: **patient-independent DS1 / DS2**")

    st.subheader("Quick Navigation")
    render_navigation_cards(
        [
            ("ECG Analysis", "Segmented beats or annotated raw records.", "ECG Analysis"),
            ("Explainability", "Grad-CAM, Grad-CAM++, IG, SmoothGrad, Saliency.", "Explainability"),
            ("Performance", "Metrics, matrices, ROC, curves, benchmark.", "Model Performance"),
            ("Reports", "JSON, PNG, and PDF report downloads.", "Report Generator"),
            ("Batch Analysis", "Multiple ECG beat files with CSV and JSON export.", "Batch Analysis"),
            ("Settings", "Device, confidence threshold, batch size, theme.", "Settings"),
        ]
    )

    with st.expander("System Readiness", expanded=False):
        readiness = {
            "Trained model": MODEL_PATH.exists(),
            "Results directory": RESULTS_DIR.exists(),
            "Metrics file": metrics_path.exists(),
            "Training history": TRAINING_HISTORY_PATH.exists(),
            "Processed dataset": PROCESSED_DATASET_PATH.exists(),
        }
        for label, available in readiness.items():
            st.markdown(
                status_badge("Available" if available else "Missing", "ok" if available else "bad")
                + f" {label}",
                unsafe_allow_html=True,
            )


def format_percent(value: object) -> str:
    """Format a numeric value as a percentage."""

    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return "Unavailable"


def main() -> None:
    """Run the Streamlit app."""

    configure_page()
    initialize_session_state()
    inject_global_styles()
    render_sidebar()
    render_home()
    render_footer()


if __name__ == "__main__":
    main()
