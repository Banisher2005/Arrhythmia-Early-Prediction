"""Attention visualization component."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_app.utils.plotting import attention_figure


def render_attention_plot(signal: np.ndarray, attention: np.ndarray | None) -> None:
    """Render temporal attention weights over ECG."""

    st.subheader("Temporal Attention")
    if attention is None:
        st.info("Attention weights are not available for the current prediction.")
        return
    st.plotly_chart(attention_figure(signal, attention), use_container_width=True)

