# Results

This directory stores all outputs generated during model training and evaluation.

## Generated Files

| File                              | Description                                   |
| --------------------------------- | --------------------------------------------- |
| `classification_report.txt`       | Precision, Recall and F1-score for each class |
| `confusion_matrix.png`            | Confusion Matrix                              |
| `confusion_matrix_normalized.png` | Normalized Confusion Matrix                   |
| `roc_curve.png`                   | Multi-class ROC Curve                         |
| `training_curves.png`             | Training & Validation Loss/Accuracy           |
| `predictions.csv`                 | Model predictions on the test dataset         |
| `metrics.json`                    | Evaluation metrics                            |

---

## Metrics

The project reports the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Macro F1
* Weighted F1
* ROC-AUC

---

## Evaluation Protocol

The model is evaluated using the official **MIT-BIH Inter-Patient DS1/DS2 Protocol**, ensuring that no patient appears in both training and testing sets.

---

## Output Figures

### Confusion Matrix

Shows the number of correct and incorrect predictions for each heartbeat class.

### Normalized Confusion Matrix

Shows per-class classification performance independent of class frequency.

### ROC Curve

Displays one-vs-rest ROC curves for all heartbeat classes together with micro and macro averaged ROC-AUC.

### Training Curves

Visualizes convergence of:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

---

## Heartbeat Classes

| Class | Description           |
| ----- | --------------------- |
| N     | Normal Beat           |
| S     | Supraventricular Beat |
| V     | Ventricular Beat      |
| F     | Fusion Beat           |
| Q     | Unknown Beat          |
