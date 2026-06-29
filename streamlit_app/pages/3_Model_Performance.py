"""Model performance dashboard page."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from streamlit_app.components.footer import render_footer
from streamlit_app.components.metrics_card import render_metric_cards
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.ui import inject_clinical_styles, status_badge
from streamlit_app.config import BENCHMARK_PATHS, RESULTS_DIR, TRAINING_HISTORY_PATH
from streamlit_app.utils.loaders import (
    load_benchmark_results,
    load_training_history,
    read_json,
    read_text,
)
from streamlit_app.utils.plotting import benchmark_comparison_figure, training_curve_figure
from streamlit_app.utils.session import initialize_session_state


st.set_page_config(page_title="Model Performance", page_icon="MET", layout="wide")
initialize_session_state()
inject_clinical_styles()
render_sidebar()

st.title("Model Performance")
st.write("Review saved evaluation artifacts from the existing pipeline. No retraining is performed.")

metrics_path = RESULTS_DIR / "metrics.json"
if metrics_path.exists():
    try:
        metrics = read_json(metrics_path)
        render_metric_cards(metrics)

        per_class = metrics.get("per_class", {})
        if isinstance(per_class, dict) and per_class:
            st.subheader("Per-Class Metrics")
            st.dataframe(pd.DataFrame(per_class).T, use_container_width=True)
    except ValueError as exc:
        st.error(str(exc))
else:
    st.warning(f"Metrics file not found: {metrics_path}")

image_col_a, image_col_b = st.columns(2)
with image_col_a:
    st.subheader("Confusion Matrix")
    matrix_path = RESULTS_DIR / "confusion_matrix.png"
    if matrix_path.exists():
        st.image(str(matrix_path), use_container_width=True)
    else:
        st.info("Confusion matrix image is not available.")

with image_col_b:
    st.subheader("ROC Curve")
    roc_path = RESULTS_DIR / "roc_curve.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
    else:
        st.info("ROC curve image is not available.")

normalized_path = RESULTS_DIR / "confusion_matrix_normalized.png"
if normalized_path.exists():
    with st.expander("Normalized Confusion Matrix", expanded=False):
        st.image(str(normalized_path), use_container_width=True)

if TRAINING_HISTORY_PATH.exists():
    try:
        st.subheader("Training Curves")
        st.plotly_chart(
            training_curve_figure(load_training_history(TRAINING_HISTORY_PATH)),
            use_container_width=True,
        )
    except ValueError as exc:
        st.error(str(exc))

report_path = RESULTS_DIR / "classification_report.txt"
if report_path.exists():
    with st.expander("Classification Report", expanded=False):
        st.code(read_text(report_path), language="text")

st.subheader("Explainability Benchmark")
try:
    benchmark_rows = load_benchmark_results(BENCHMARK_PATHS)
except ValueError as exc:
    benchmark_rows = []
    st.error(str(exc))

if benchmark_rows:
    st.markdown(status_badge("Artifact loaded", "ok"), unsafe_allow_html=True)
    benchmark_frame = pd.DataFrame(benchmark_rows)
    st.dataframe(benchmark_frame, use_container_width=True)

    columns = st.columns(4)
    if "mean_runtime_ms" in benchmark_frame:
        columns[0].metric("Best Runtime", f"{benchmark_frame['mean_runtime_ms'].min():.2f} ms")
    if "fps" in benchmark_frame:
        columns[1].metric("Best FPS", f"{benchmark_frame['fps'].max():.2f}")
    if "gpu_memory_mb" in benchmark_frame:
        gpu_memory = benchmark_frame["gpu_memory_mb"].fillna(0)
        columns[2].metric("Peak GPU Memory", f"{gpu_memory.max():.2f} MB")
    columns[3].metric("Methods", str(len(benchmark_frame)))

    chart_a, chart_b = st.columns(2)
    with chart_a:
        if "mean_runtime_ms" in benchmark_frame:
            st.plotly_chart(
                benchmark_comparison_figure(
                    benchmark_rows,
                    "mean_runtime_ms",
                    "Runtime (ms)",
                ),
                use_container_width=True,
            )
    with chart_b:
        if "fps" in benchmark_frame:
            st.plotly_chart(
                benchmark_comparison_figure(
                    benchmark_rows,
                    "fps",
                    "Frames per Second",
                ),
                use_container_width=True,
            )
else:
    st.info("No explainability benchmark artifact was found in results/.")

render_footer()
