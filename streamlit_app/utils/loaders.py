"""File and artifact loading utilities for the Streamlit app."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.config import SEGMENT_LENGTH


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted JSON file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


@st.cache_data(show_spinner=False)
def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""

    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_training_history(path: Path) -> dict[str, list[float]]:
    """Load training history as numeric series."""

    history = read_json(path)
    return {
        key: [float(value) for value in values]
        for key, values in history.items()
        if isinstance(values, list)
    }


def _first_numeric_column(frame: pd.DataFrame) -> np.ndarray:
    """Return the first numeric dataframe column as a one-dimensional array."""

    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("The uploaded tabular file does not contain numeric ECG values.")
    return numeric.iloc[:, 0].to_numpy(dtype=np.float32)


def load_uploaded_ecg(uploaded_file: Any) -> np.ndarray:
    """Load ECG samples from CSV, TXT, DAT, or NPY Streamlit uploads."""

    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if suffix == ".npy":
        signal = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    elif suffix == ".csv":
        signal = _first_numeric_column(pd.read_csv(io.BytesIO(raw_bytes)))
    elif suffix in {".txt", ".dat"}:
        text = raw_bytes.decode("utf-8", errors="ignore")
        try:
            signal = np.loadtxt(io.StringIO(text), dtype=np.float32)
        except ValueError:
            frame = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            signal = _first_numeric_column(frame)
    else:
        raise ValueError(f"Unsupported ECG file type: {suffix}")

    signal = np.asarray(signal, dtype=np.float32).squeeze()
    if signal.ndim > 1:
        signal = signal.reshape(-1)
    if signal.size == 0:
        raise ValueError("The uploaded ECG file is empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("The uploaded ECG contains NaN or infinite values.")
    return signal


def load_uploaded_annotations(uploaded_file: Any):
    """
    Load MIT-BIH-style heartbeat annotations through the repository parser.
    """

    from src.preprocessing.load_annotations import parse_annotation_lines

    text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    annotations = parse_annotation_lines(text.splitlines())
    if annotations.positions.size == 0:
        raise ValueError("No valid heartbeat annotations were found.")
    return annotations


def segment_uploaded_record(
    signal: np.ndarray,
    annotations: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess and segment a raw recording using the existing pipeline.
    """

    from src.preprocessing.create_dataset import segment_record
    from src.preprocessing.filter_signal import preprocess_signal

    processed = preprocess_signal(np.asarray(signal, dtype=np.float32).reshape(-1))
    beats, labels, encoded, _positions, _skipped = segment_record(
        processed,
        annotations,
    )
    if beats.size == 0:
        raise ValueError(
            "No valid beat windows were produced from the supplied recording "
            "and annotations."
        )
    return beats, labels, encoded


def validate_heartbeat_segment(
    signal: np.ndarray,
    length: int = SEGMENT_LENGTH,
) -> np.ndarray:
    """
    Validate a heartbeat segment against the training input contract.

    The research pipeline segments beats in ``src.preprocessing.segment_beats``.
    Streamlit accepts already segmented beat windows and does not perform an
    alternate crop or extraction path.
    """

    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size != length:
        raise ValueError(
            f"Expected one segmented heartbeat with {length} samples, "
            f"but received {signal.size} samples. Use the existing "
            "preprocessing pipeline to create beat-centered segments."
        )
    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG segment contains NaN or infinite values.")
    return signal


@st.cache_data(show_spinner=False)
def load_dataset_summary(dataset_path: Path) -> dict[str, Any]:
    """Load processed MIT-BIH dataset metadata for dashboard display."""

    if not dataset_path.exists():
        return {
            "total_beats": None,
            "metadata": {},
        }

    dataset = np.load(dataset_path, allow_pickle=True)
    metadata: dict[str, Any] = {}
    if "metadata" in dataset:
        raw_metadata = dataset["metadata"]
        if raw_metadata.shape == ():
            metadata = dict(raw_metadata.item())

    total_beats = int(dataset["beats"].shape[0]) if "beats" in dataset else None
    return {
        "total_beats": total_beats,
        "metadata": metadata,
    }


@st.cache_data(show_spinner=False)
def load_benchmark_results(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    """Load the first available explainability benchmark artifact."""

    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted benchmark JSON file: {path}") from exc
        if isinstance(data, dict):
            return [data]
        if not isinstance(data, list):
            raise ValueError(f"Expected benchmark list or object in {path}.")
        return data
    return []
