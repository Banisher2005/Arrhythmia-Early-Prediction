"""Batch ECG heartbeat inference page."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_app.components.footer import render_footer
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.ui import inject_clinical_styles, status_badge
from streamlit_app.config import MODEL_PATH, SUPPORTED_UPLOAD_TYPES
from streamlit_app.utils.inference import predict_signal
from streamlit_app.utils.loaders import load_uploaded_ecg
from streamlit_app.utils.session import initialize_session_state


def result_to_row(filename: str, result: Any) -> dict[str, object]:
    """Convert a prediction result to a table row."""

    return {
        "file": filename,
        "prediction": result.predicted_class,
        "confidence": result.confidence,
        "inference_time_ms": result.inference_time_ms,
    }


st.set_page_config(page_title="Batch Analysis", page_icon="BATCH", layout="wide")
initialize_session_state()
inject_clinical_styles()
render_sidebar()

st.title("Batch Analysis")
st.write(
    "Upload multiple segmented heartbeat files and run cached AMSRAN-GF model "
    "inference for each file."
)

if not MODEL_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()

files = st.file_uploader(
    "Upload segmented heartbeat files",
    type=list(SUPPORTED_UPLOAD_TYPES),
    accept_multiple_files=True,
)

if st.button("Run Batch Inference", type="primary", disabled=not files):
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    progress = st.progress(0, text="Preparing batch inference...")
    for index, uploaded_file in enumerate(files):
        try:
            signal = load_uploaded_ecg(uploaded_file)
            result = predict_signal(signal, st.session_state.device)
            rows.append(result_to_row(uploaded_file.name, result))
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")
        progress.progress((index + 1) / len(files), text=f"Processed {index + 1}/{len(files)}")
    st.session_state.batch_results = rows
    st.session_state.batch_errors = errors
    progress.empty()

rows = st.session_state.get("batch_results")
errors = st.session_state.get("batch_errors", [])

if rows:
    st.markdown(status_badge("Complete", "ok") + f" {len(rows)} files analyzed.", unsafe_allow_html=True)
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame.style.format(
            {
                "confidence": "{:.2%}",
                "inference_time_ms": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps(rows, indent=2).encode("utf-8")
    col_a, col_b = st.columns(2)
    col_a.download_button("Download CSV", csv_bytes, "batch_predictions.csv", "text/csv")
    col_b.download_button(
        "Download JSON",
        json_bytes,
        "batch_predictions.json",
        "application/json",
    )

if errors:
    with st.expander("Files skipped due to validation errors", expanded=True):
        for error in errors:
            st.warning(error)

render_footer()
