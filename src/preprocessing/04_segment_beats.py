from pathlib import Path
import pandas as pd
import numpy as np


# Project Paths


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"


# Load ECG Signal


ecg = pd.read_csv(DATA_DIR / "100.csv")

ecg.columns = ecg.columns.str.replace("'", "").str.strip()

signal = ecg["MLII"].values


# Load Annotation File


annotations = pd.read_csv(
    DATA_DIR / "100annotations.txt",
    sep=r"\s+"
)


# Parameters


WINDOW_BEFORE = 90
WINDOW_AFTER = 90


# Segment Beats


beats = []
labels = []

for _, row in annotations.iterrows():

    center = int(row["Sample"])

    label = row["#"]

    start = center - WINDOW_BEFORE
    end = center + WINDOW_AFTER

    # Ignore beats near beginning or end
    if start < 0 or end >= len(signal):
        continue

    beat = signal[start:end]

    beats.append(beat)

    labels.append(label)


# Convert to NumPy


beats = np.array(beats)

labels = np.array(labels)


# Display Information


print("=" * 60)
print("HEARTBEAT SEGMENTATION COMPLETED")
print("=" * 60)

print(f"\nTotal Beats: {len(beats)}")

print(f"\nBeat Shape: {beats.shape}")

print(f"\nLabels Shape: {labels.shape}")

print("\nFirst Label:")

print(labels[0])

print("\nFirst Beat:")

print(beats[0])

print("\nLesson 4 Completed Successfully!")

# ==========================================================
# Save Processed Dataset
# ==========================================================

PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

np.save(PROCESSED_DIR / "beats.npy", beats)

np.save(PROCESSED_DIR / "labels.npy", labels)

print("\nProcessed dataset saved successfully!")

print(PROCESSED_DIR)