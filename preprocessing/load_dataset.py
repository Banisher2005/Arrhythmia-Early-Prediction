from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folder
DATA_DIR = BASE_DIR / "data" / "raw"

print("Project Root:", BASE_DIR)
print("Data Folder:", DATA_DIR)

# Load ECG
data = pd.read_csv(DATA_DIR / "100.csv")

# Remove unwanted quotes from column names
data.columns = data.columns.str.replace("'", "").str.strip()

print("Columns:")
print(data.columns)

print("\nFirst Five Rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

# Plot ECG Signal

plt.figure(figsize=(15,5))

plt.plot(
    data["MLII"][:3000],
    color="blue",
    linewidth=1
)

plt.title("ECG Signal - Record 100")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()


# Load Annotation File

annotations = pd.read_csv(
    DATA_DIR / "100annotations.txt",
    sep=r"\s+"
)

print("\nAnnotation Columns:")
print(annotations.columns)

print("\nFirst Five Annotation Rows:")
print(annotations.head())