"""Explainability page for trained AMSRAN-GF predictions."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_app.components.attention_plot import render_attention_plot
from streamlit_app.components.ecg_plot import render_ecg_plot
from streamlit_app.components.footer import render_footer
from streamlit_app.components.gate_plot import render_gate_plot
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.ui import inject_clinical_styles, status_badge
from streamlit_app.components.uploader import render_ecg_uploader
from streamlit_app.config import MODEL_PATH
from streamlit_app.utils.inference import generate_explanation, predict_signal
from streamlit_app.utils.loaders import validate_heartbeat_segment
from streamlit_app.utils.plotting import attribution_figure
from streamlit_app.utils.session import initialize_session_state


METHOD_LABELS = {
    "gradcam++": "Grad-CAM++",
    "gradcam": "Grad-CAM",
    "integrated_gradients": "Integrated Gradients",
    "smoothgrad": "SmoothGrad",
    "saliency": "Saliency",
}


def extract_attribution(result: dict[str, object], method: str) -> np.ndarray | None:
    """Extract an attribution array from an existing explainer result."""

    preferred_keys = {
        "gradcam++": ("gradcam_pp", "gradcam++", "gradcam"),
        "gradcam": ("gradcam", "cam"),
        "smoothgrad": ("smoothgrad",),
        "integrated_gradients": ("integrated_gradients", "ig"),
        "saliency": ("saliency", "saliency_map"),
    }[method]
    for key in preferred_keys:
        value = result.get(key)
        if value is not None:
            return np.asarray(value, dtype=float).squeeze()
    for value in result.values():
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=float).squeeze()
    return None


st.set_page_config(page_title="Explainability", page_icon="XAI", layout="wide")
initialize_session_state()
inject_clinical_styles()
render_sidebar()

st.title("Explainability")
st.write(
    "Generate and compare explainability methods through the existing "
    "`src.explainability.Explainer` framework."
)

signal = render_ecg_uploader()
if signal is not None:
    render_ecg_plot(signal, "Uploaded ECG")

method = st.selectbox(
    "Explainability Method",
    tuple(METHOD_LABELS.keys()),
    format_func=METHOD_LABELS.get,
)

if signal is not None:
    try:
        window = validate_heartbeat_segment(signal)
    except ValueError as exc:
        st.error(str(exc))
        render_footer()
        st.stop()

    cache_key = f"{hash(window.tobytes())}:{st.session_state.device}"
    cache = st.session_state.setdefault("explanation_cache", {})
    method_cache = cache.setdefault(cache_key, {})

    if st.button("Generate Explanation", type="primary"):
        if not MODEL_PATH.exists():
            st.error(f"Model checkpoint not found: {MODEL_PATH}")
        else:
            try:
                with st.spinner(f"Running {METHOD_LABELS[method]}..."):
                    if "prediction" not in method_cache:
                        method_cache["prediction"] = predict_signal(
                            window,
                            st.session_state.device,
                        )
                    if method not in method_cache:
                        method_cache[method] = generate_explanation(
                            window,
                            method,
                            st.session_state.device,
                        )
                    st.session_state.explainability_result = {
                        "method": method,
                        "explanation": method_cache[method],
                        "prediction": method_cache["prediction"],
                    }
            except Exception as exc:
                st.error(f"Explainability failed: {exc}")

    cached_methods = [
        METHOD_LABELS[key]
        for key in METHOD_LABELS
        if key in method_cache
    ]
    if cached_methods:
        st.markdown(
            status_badge("Cached", "ok") + " " + ", ".join(cached_methods),
            unsafe_allow_html=True,
        )

stored = st.session_state.get("explainability_result")
if stored is not None and signal is not None:
    selected_method = str(stored["method"])
    attribution = extract_attribution(dict(stored["explanation"]), selected_method)
    window = validate_heartbeat_segment(signal)
    if attribution is not None:
        st.plotly_chart(
            attribution_figure(window, attribution, METHOD_LABELS[selected_method]),
            use_container_width=True,
        )
    else:
        st.info("The selected explainer completed, but no attribution array was returned.")

    prediction = stored["prediction"]
    render_attention_plot(window, prediction.attention_weights)
    render_gate_plot(prediction.gate_weights)

render_footer()
