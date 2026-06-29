"""Application settings page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.footer import render_footer
from streamlit_app.components.ui import inject_clinical_styles, status_badge
from streamlit_app.utils.inference import resolve_device
from streamlit_app.utils.session import clear_analysis_state, initialize_session_state


st.set_page_config(page_title="Settings", page_icon="CFG", layout="wide")
initialize_session_state()
inject_clinical_styles()

st.sidebar.title("AMSRAN-GF")
st.sidebar.caption("Settings")

st.title("Settings")
st.write("Configure runtime behavior for the Streamlit interface.")

st.selectbox("Device", ("Auto", "CPU", "GPU"), key="device")
st.slider("Confidence Threshold", 0.0, 1.0, step=0.01, key="confidence_threshold")
st.number_input("Batch Size", min_value=1, max_value=1024, step=1, key="batch_size")
st.selectbox("Theme", ("System", "Light", "Dark"), key="theme")

st.subheader("Current Configuration")
try:
    resolved_device = str(resolve_device(st.session_state.device)).upper()
except Exception:
    resolved_device = "Unavailable"

st.markdown(
    status_badge(f"Resolved Device: {resolved_device}", "ok"),
    unsafe_allow_html=True,
)
st.json(
    {
        "device": st.session_state.device,
        "confidence_threshold": st.session_state.confidence_threshold,
        "batch_size": st.session_state.batch_size,
        "theme": st.session_state.theme,
    }
)

if st.button("Clear Uploaded ECG and Results"):
    clear_analysis_state()
    st.success("Analysis state cleared.")

render_footer()
