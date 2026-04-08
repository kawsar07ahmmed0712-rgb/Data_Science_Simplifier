from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    y_proba=None,
) -> dict[str, object]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }

    labels = sorted([str(label) for label in pd.Series(y_true).dropna().unique().tolist()])
    cm = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels)
    metrics["confusion_matrix"] = {
        "labels": labels,
        "matrix": cm.tolist(),
    }

    roc_auc = None
    try:
        unique_classes = pd.Series(y_true).dropna().unique()
        if y_proba is not None and len(unique_classes) == 2:
            if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                roc_auc = round(float(roc_auc_score(y_true, y_proba[:, 1])), 6)
    except Exception:
        roc_auc = None

    metrics["roc_auc"] = roc_auc
    return metrics