from pathlib import Path
import pandas as pd


# Project Paths


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


# Load Annotation File


annotations = pd.read_csv(
    DATA_DIR / "100annotations.txt",
    sep=r"\s+"
)

print("=" * 60)
print("ANNOTATION FILE LOADED")
print("=" * 60)


# Basic Information


print("\nColumns:")
print(annotations.columns)

print("\nFirst Five Rows:")
print(annotations.head())

print("\nDataset Shape:")
print(annotations.shape)


# Beat Symbols


print("\nUnique Beat Symbols:")
print(annotations["#"].unique())

print("\nBeat Counts:")
print(annotations["#"].value_counts())


# Beat Positions


beat_positions = annotations["Sample"]

print("\nFirst 10 Beat Positions:")
print(beat_positions.head(10))


# Convert Beat Positions to List


beat_positions = beat_positions.tolist()

print("\nPython Data Type:")
print(type(beat_positions))

print("\nFirst 10 Beat Positions (List):")
print(beat_positions[:10])

print("\nLesson 3 Completed Successfully!")