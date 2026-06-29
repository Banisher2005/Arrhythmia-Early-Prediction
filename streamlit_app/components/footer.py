"""Reusable footer component."""

from __future__ import annotations

import streamlit as st


def render_footer() -> None:
    """Render a restrained clinical software footer."""

    st.divider()
    st.caption(
        "AMSRAN-GF ECG Analysis Workbench. Research demonstration only; "
        "not a medical device or substitute for clinician review."
    )

