"""Reusable sidebar component."""

from __future__ import annotations

import streamlit as st

from streamlit_app.config import CLASS_NAMES, MODEL_PATH, SAMPLING_RATE, SEGMENT_LENGTH
from streamlit_app.utils.inference import resolve_device
from streamlit_app.utils.session import initialize_session_state


def render_sidebar() -> None:
    """Render settings and project context in the sidebar."""

    initialize_session_state()
    st.sidebar.title("AMSRAN-GF")
    st.sidebar.caption("Clinical ECG AI demonstration")
    st.sidebar.divider()
    st.sidebar.selectbox(
        "Device",
        ("Auto", "CPU", "GPU"),
        key="device",
        help="GPU uses CUDA when available.",
    )
    st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="confidence_threshold",
    )
    st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=1024,
        step=1,
        key="batch_size",
    )
    st.sidebar.selectbox("Theme", ("System", "Light", "Dark"), key="theme")
    st.sidebar.divider()
    try:
        resolved_device = str(resolve_device(st.session_state.device)).upper()
    except Exception:
        resolved_device = "Unavailable"
    st.sidebar.markdown("**Runtime Status**")
    st.sidebar.write(f"Device: {resolved_device}")
    st.sidebar.write("Checkpoint: Ready" if MODEL_PATH.exists() else "Checkpoint: Missing")
    st.sidebar.divider()
    st.sidebar.markdown("**Signal Window**")
    st.sidebar.write(f"{SEGMENT_LENGTH} samples at {SAMPLING_RATE} Hz")
    st.sidebar.markdown("**Classes**")
    st.sidebar.write(", ".join(CLASS_NAMES))
