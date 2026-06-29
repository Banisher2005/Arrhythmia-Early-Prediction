"""About page for the AMSRAN-GF Streamlit application."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.footer import render_footer
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.ui import inject_clinical_styles, render_hero, status_badge
from streamlit_app.config import CLASS_NAMES, SAMPLING_RATE, SEGMENT_LENGTH
from streamlit_app.utils.session import initialize_session_state


st.set_page_config(page_title="About", page_icon="INFO", layout="wide")
initialize_session_state()
inject_clinical_styles()
render_sidebar()

render_hero(
    "Project overview",
    "About AMSRAN-GF",
    "Adaptive Multi-Scale Residual Attention Network with Gated Fusion for ECG arrhythmia classification.",
    [
        status_badge("MIT-BIH", "ok"),
        status_badge("DS1 / DS2", "ok"),
        status_badge("Explainable AI", "ok"),
    ],
)

left, right = st.columns(2)
with left:
    st.subheader("Project Overview")
    st.write(
        "The repository implements a complete research pipeline: "
        "patient-independent DS1/DS2 splitting, preprocessing, beat "
        "segmentation, model inference, evaluation, and explainability."
    )

    st.subheader("Model Architecture")
    st.write(
        "The model combines multi-scale residual CNN feature extraction, "
        "adaptive gated fusion, BiLSTM sequence modeling, temporal attention, "
        "and a classifier head for five AAMI heartbeat classes."
    )

with right:
    st.subheader("Dataset")
    st.write(
        f"MIT-BIH Arrhythmia Database, sampled at {SAMPLING_RATE} Hz with "
        f"{SEGMENT_LENGTH}-sample heartbeat windows. Classes: {', '.join(CLASS_NAMES)}."
    )

    st.subheader("Explainability Methods")
    st.write(
        "The app reuses the existing Grad-CAM, Grad-CAM++, SmoothGrad, "
        "Integrated Gradients, saliency, temporal attention, and adaptive "
        "gate visualizations."
    )

st.subheader("Repository")
st.write("GitHub: https://github.com/Banisher2005/Arrhythmia-Early-Prediction")

render_footer()
