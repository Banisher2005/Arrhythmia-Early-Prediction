# 🫀 AMSRAN-GF: Adaptive Multi-Scale Residual Attention Network with Gated Fusion for Early Arrhythmia Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![MIT-BIH](https://img.shields.io/badge/Dataset-MIT--BIH-success.svg)
![Status](https://img.shields.io/badge/Status-Research%20Project-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Deep Learning Framework for Automated ECG Arrhythmia Classification using the MIT-BIH Arrhythmia Database**

</div>

---

# 📖 Overview

AMSRAN-GF (Adaptive Multi-Scale Residual Attention Network with Gated Fusion) is an end-to-end deep learning framework for automated ECG heartbeat classification.

The project implements the complete research pipeline—from ECG preprocessing to model inference—using the **official patient-independent MIT-BIH inter-patient evaluation protocol (DS1/DS2)**.

Unlike conventional CNN-only approaches, AMSRAN-GF combines:

* Multi-Scale Residual CNN Feature Extraction
* Adaptive Gated Feature Fusion (AGF)
* Bidirectional LSTM
* Temporal Attention
* Weighted Focal Loss
* AdamW Optimization
* Cosine Annealing Learning Rate Scheduler
* Mixed Precision Training
* Early Stopping
* Complete Evaluation Pipeline

The goal is to provide a reproducible and extensible framework for ECG heartbeat classification while maintaining research-grade implementation quality.

---

# 🏗 Proposed Architecture

<p align="center">
<img src="docs/architecture.png" width="1000">
</p>

The proposed architecture consists of six major stages:

1. ECG Beat Input
2. Multi-Scale Residual CNN
3. Adaptive Gated Feature Fusion
4. Bidirectional LSTM
5. Temporal Attention
6. Fully Connected Classification Head

Output classes:

| Class | Description           |
| ----- | --------------------- |
| N     | Normal Beat           |
| S     | Supraventricular Beat |
| V     | Ventricular Beat      |
| F     | Fusion Beat           |
| Q     | Unknown Beat          |

---

# ✨ Features

* Complete ECG preprocessing pipeline
* Butterworth Band-Pass Filtering
* Heartbeat segmentation around R-peaks
* AAMI heartbeat mapping
* Official MIT-BIH DS1 / DS2 patient-independent split
* PyTorch Dataset & DataLoader pipeline
* Multi-Scale Residual CNN backbone
* Adaptive Gated Feature Fusion (AGF)
* Bidirectional LSTM
* Temporal Attention
* Weighted Focal Loss
* AdamW optimizer
* Cosine Annealing Scheduler
* Automatic Early Stopping
* Automatic checkpoint saving
* Confusion Matrix generation
* ROC Curve generation
* Classification Report generation
* Complete inference pipeline

---

# 📂 Dataset

Dataset:

**MIT-BIH Arrhythmia Database**

The project follows the official **de Chazal Inter-Patient Evaluation Protocol**.

Training:

DS1

Testing:

DS2

This prevents patient leakage between training and testing.

---

# 📁 Project Structure

```
Arrhythmia-Early-Prediction/
│
├── data/
│
├── docs/
│
├── models_saved/
│
├── results/
│
├── src/
│   ├── configs/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
│
├── requirements.txt
├── train.py
├── inference.py
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Banisher2005/Arrhythmia-Early-Prediction.git

cd Arrhythmia-Early-Prediction
```

Create environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Start training

```bash
python -m src.training.train
```

Training automatically includes

* Mixed Precision Training
* Early Stopping
* Automatic Checkpoint Saving
* Cosine Learning Rate Scheduling
* Validation after every epoch

The best model is automatically stored in

```
models_saved/best_model.pt
```

---

# 🔍 Inference

Run inference

```bash
python -m src.evaluation.inference
```

Automatically generates

* Predictions
* Classification Report
* Confusion Matrix
* Normalized Confusion Matrix
* ROC Curve
* Performance Metrics

---

# 📊 Evaluation

Implemented metrics include

* Accuracy
* Precision
* Recall
* F1 Score
* Macro F1
* Weighted F1
* ROC-AUC
* Confusion Matrix
* Classification Report

Generated outputs

```
results/

├── classification_report.txt
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── roc_curve.png
├── predictions.csv
```

---

# 🧠 Model Pipeline

```
Input ECG Beat
        │
        ▼
Band-Pass Filtering
        │
        ▼
Heartbeat Segmentation
        │
        ▼
Multi-Scale Residual CNN
        │
        ▼
Adaptive Gated Fusion
        │
        ▼
BiLSTM
        │
        ▼
Temporal Attention
        │
        ▼
Fully Connected Layer
        │
        ▼
Softmax
        │
        ▼
Predicted Class
```

---

# 📈 Current Baseline

| Metric     |               Value |
| ---------- | ------------------: |
| Dataset    |  MIT-BIH Arrhythmia |
| Framework  |             PyTorch |
| Classes    |                   5 |
| Optimizer  |               AdamW |
| Scheduler  |   CosineAnnealingLR |
| Loss       | Weighted Focal Loss |
| Evaluation |  Official DS1 / DS2 |

---

# 🔬 Future Work

* Early arrhythmia prediction before onset
* Explainable AI using Grad-CAM
* ECG transformer comparison
* Multi-lead ECG support
* Lightweight deployment model
* Real-time monitoring system
* Clinical validation
* Mobile deployment

---

# 📚 Citation

If you use this repository in your research, please cite:

```text
Abhinav Kumar.
AMSRAN-GF: Adaptive Multi-Scale Residual Attention Network with Gated Fusion
for Early Arrhythmia Prediction.
2026.
```

---

# 👨‍💻 Author

**Abhinav Kumar**

Computer Science Engineering Student

SRM Institute of Science and Technology

GitHub:

https://github.com/Banisher2005

---

# ⭐ Support

If you found this repository useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and supports future development.
