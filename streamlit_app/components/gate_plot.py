"""Adaptive gate visualization component."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_app.utils.plotting import gate_figure


def render_gate_plot(gate_weights: list[np.ndarray]) -> None:
    """Render adaptive gate weights."""

    st.subheader("Adaptive Gate Weights")
    if not gate_weights:
        st.info("Adaptive gate weights are not available for the current prediction.")
        return
    st.plotly_chart(gate_figure(gate_weights), use_container_width=True)

