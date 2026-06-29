"""Class probability chart component."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_app.config import CLASS_NAMES
from streamlit_app.utils.plotting import probability_bar_figure


def render_probability_chart(probabilities: np.ndarray) -> None:
    """Render class probabilities."""

    st.subheader("Class Probabilities")
    st.plotly_chart(
        probability_bar_figure(probabilities, list(CLASS_NAMES)),
        use_container_width=True,
    )

