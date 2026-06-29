"""ECG upload component."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_app.config import SUPPORTED_UPLOAD_TYPES
from streamlit_app.utils.loaders import load_uploaded_ecg, validate_heartbeat_segment


def render_ecg_uploader() -> np.ndarray | None:
    """Render uploader and return parsed ECG samples."""

    uploaded_file = st.file_uploader(
        "Upload ECG signal",
        type=list(SUPPORTED_UPLOAD_TYPES),
        help="Supported formats: CSV, TXT, DAT, NPY.",
    )
    if uploaded_file is None:
        return st.session_state.get("ecg_signal")

    try:
        signal = load_uploaded_ecg(uploaded_file)
        validate_heartbeat_segment(signal)
    except Exception as exc:
        st.error(f"Could not read ECG file: {exc}")
        return None

    st.session_state.ecg_signal = signal
    st.session_state.ecg_filename = uploaded_file.name
    st.success(f"Loaded {uploaded_file.name} with {signal.size:,} samples.")
    return signal
