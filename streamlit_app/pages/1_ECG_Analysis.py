"""ECG upload, visualization, and inference page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ecg_plot import render_ecg_plot
from streamlit_app.components.footer import render_footer
from streamlit_app.components.prediction_card import render_prediction_card
from streamlit_app.components.probability_chart import render_probability_chart
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.ui import inject_clinical_styles, status_badge
from streamlit_app.components.uploader import render_ecg_uploader
from streamlit_app.config import MODEL_PATH
from streamlit_app.utils.inference import predict_signal
from streamlit_app.utils.loaders import (
    load_uploaded_annotations,
    load_uploaded_ecg,
    segment_uploaded_record,
)
from streamlit_app.utils.session import clear_analysis_state, initialize_session_state


st.set_page_config(page_title="ECG Analysis", page_icon="ECG", layout="wide")
initialize_session_state()
inject_clinical_styles()
render_sidebar()

st.title("ECG Analysis")
st.write(
    "Run AMSRAN-GF inference on a segmented heartbeat or on beats extracted "
    "from an annotated raw ECG recording."
)

if not MODEL_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()

tab_segment, tab_raw = st.tabs(["Segmented Heartbeat", "Annotated Raw Recording"])

with tab_segment:
    st.markdown(
        status_badge("Workflow A", "ok") + " Upload one 180-sample segmented beat.",
        unsafe_allow_html=True,
    )
    signal = render_ecg_uploader()
    left, right = st.columns([1, 4])
    with left:
        run_inference = st.button(
            "Run Inference",
            type="primary",
            disabled=signal is None,
            key="segment_inference",
        )
    with right:
        if st.button("Clear Analysis", key="clear_segment"):
            clear_analysis_state()
            st.rerun()

    if signal is not None:
        st.session_state.ecg_signal = signal
        render_ecg_plot(signal, "Segmented Heartbeat")

    if run_inference and signal is not None:
        try:
            with st.spinner("Running AMSRAN-GF inference..."):
                st.session_state.prediction_result = predict_signal(
                    signal,
                    st.session_state.device,
                )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

with tab_raw:
    st.markdown(
        status_badge("Workflow B", "ok")
        + " Upload a raw ECG recording and MIT-BIH-style annotation text.",
        unsafe_allow_html=True,
    )
    raw_file = st.file_uploader(
        "Raw ECG recording",
        type=["csv", "txt", "dat", "npy"],
        key="raw_ecg_upload",
    )
    annotation_file = st.file_uploader(
        "Annotation file",
        type=["txt", "atr"],
        key="annotation_upload",
        help="Expected rows with Time Sample Symbol fields, matching the existing MIT-BIH parser.",
    )

    if st.button(
        "Preprocess and Segment",
        type="primary",
        disabled=raw_file is None or annotation_file is None,
    ):
        try:
            with st.spinner("Using existing preprocessing and segmentation modules..."):
                raw_signal = load_uploaded_ecg(raw_file)
                annotations = load_uploaded_annotations(annotation_file)
                beats, labels, encoded = segment_uploaded_record(raw_signal, annotations)
            st.session_state.raw_signal = raw_signal
            st.session_state.segmented_beats = beats
            st.session_state.segmented_labels = labels
            st.session_state.segmented_encoded = encoded
            st.success(f"Extracted {len(beats):,} valid heartbeat segments.")
        except Exception as exc:
            st.error(f"Raw ECG segmentation failed: {exc}")

    beats = st.session_state.get("segmented_beats")
    labels = st.session_state.get("segmented_labels")
    if beats is not None and labels is not None:
        beat_index = st.slider(
            "Beat index",
            min_value=0,
            max_value=max(len(beats) - 1, 0),
            value=0,
        )
        selected_beat = beats[beat_index]
        st.caption(f"Beat {beat_index} | Source label: {labels[beat_index]}")
        render_ecg_plot(selected_beat, "Selected Segmented Beat")
        if st.button("Run Inference on Selected Beat", type="primary"):
            try:
                with st.spinner("Running AMSRAN-GF inference..."):
                    st.session_state.ecg_signal = selected_beat
                    st.session_state.prediction_result = predict_signal(
                        selected_beat,
                        st.session_state.device,
                    )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")

result = st.session_state.get("prediction_result")
if result is not None:
    st.subheader("Inference Result")
    render_prediction_card(result, st.session_state.confidence_threshold)
    render_probability_chart(result.probabilities)
    if result.confidence < st.session_state.confidence_threshold:
        st.warning("Prediction confidence is below the configured review threshold.")

render_footer()
