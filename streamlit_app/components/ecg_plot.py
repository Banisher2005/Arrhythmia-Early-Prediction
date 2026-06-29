"""ECG plotting component."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_app.config import SAMPLING_RATE
from streamlit_app.utils.plotting import ecg_waveform_figure


def render_ecg_plot(signal: np.ndarray, title: str = "ECG Waveform") -> None:
    """Render an interactive ECG waveform."""

    st.subheader(title)
    st.plotly_chart(ecg_waveform_figure(signal, SAMPLING_RATE), use_container_width=True)

