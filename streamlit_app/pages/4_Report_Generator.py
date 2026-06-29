"""Report generation page."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

from streamlit_app.components.footer import render_footer
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.ui import inject_clinical_styles, status_badge
from streamlit_app.components.uploader import render_ecg_uploader
from streamlit_app.config import EXPLAINABILITY_OUTPUT_DIR, MODEL_PATH
from streamlit_app.utils.inference import generate_report
from streamlit_app.utils.session import initialize_session_state


@st.cache_data(show_spinner=False)
def create_zip(directory_name: str) -> bytes:
    """Create a ZIP archive of generated report files."""

    directory = Path(directory_name)
    archive_path = directory.with_suffix(".zip")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(directory))
    return archive_path.read_bytes()


@st.cache_data(show_spinner=False)
def create_pdf_summary(directory_name: str, report: dict[str, object]) -> bytes | None:
    """Create a concise PDF summary when fpdf2 is installed."""

    try:
        from fpdf import FPDF
    except Exception:
        return None

    directory = Path(directory_name)
    pdf_path = directory / "summary.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AMSRAN-GF ECG Explainability Report", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.ln(4)
    for key in ("prediction", "confidence", "ground_truth"):
        pdf.cell(0, 8, f"{key.replace('_', ' ').title()}: {report.get(key)}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Generated Figures", ln=True)
    pdf.set_font("Helvetica", size=10)
    files = report.get("generated_files", {})
    if isinstance(files, dict):
        for label, filename in files.items():
            pdf.cell(0, 7, f"{label}: {filename}", ln=True)
    pdf.output(str(pdf_path))
    return pdf_path.read_bytes()


st.set_page_config(page_title="Report Generator", page_icon="PDF", layout="wide")
initialize_session_state()
inject_clinical_styles()
render_sidebar()

st.title("Report Generator")
st.write("Generate JSON, PNG figures, and a PDF summary using the existing explainability report generator.")

signal = render_ecg_uploader()
if st.button("Generate Full Report", type="primary", disabled=signal is None):
    if not MODEL_PATH.exists():
        st.error(f"Model checkpoint not found: {MODEL_PATH}")
    else:
        try:
            with st.spinner("Generating explainability report..."):
                report = generate_report(signal, EXPLAINABILITY_OUTPUT_DIR, st.session_state.device)
                st.session_state.last_report_dir = str(EXPLAINABILITY_OUTPUT_DIR)
                st.session_state.generated_report = report
            st.success(f"Report generated in {EXPLAINABILITY_OUTPUT_DIR}")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

report_dir = st.session_state.get("last_report_dir")
report = st.session_state.get("generated_report")
if report_dir and report:
    directory = Path(report_dir)
    st.subheader("Report Preview")
    st.markdown(status_badge("Generated", "ok") + f" {directory}", unsafe_allow_html=True)
    st.json(report)

    json_bytes = json.dumps(report, indent=2).encode("utf-8")
    st.download_button("Download JSON Report", json_bytes, "report.json", "application/json")

    zip_bytes = create_zip(str(directory))
    st.download_button("Download PNG Figures ZIP", zip_bytes, "explainability_figures.zip", "application/zip")

    pdf_bytes = create_pdf_summary(str(directory), report)
    if pdf_bytes is not None:
        st.download_button("Download PDF Report", pdf_bytes, "ecg_report.pdf", "application/pdf")
    else:
        st.info("Install fpdf2 from streamlit_app/requirements.txt to enable PDF export.")

render_footer()
