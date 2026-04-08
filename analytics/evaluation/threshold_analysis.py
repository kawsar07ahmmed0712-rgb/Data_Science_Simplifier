from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve


def analyze_binary_thresholds(
    y_true: pd.Series,
    y_proba,
) -> dict[str, object]:
    result = {
        "is_available": False,
        "default_threshold": 0.5,
        "best_f1_threshold": None,
        "best_f1_score": None,
    }

    if y_proba is None:
        return result

    try:
        classes = pd.Series(y_true).dropna().unique()
        if len(classes) != 2:
            return result

        if not isinstance(y_proba, np.ndarray) or y_proba.ndim != 2 or y_proba.shape[1] < 2:
            return result

        positive_probs = y_proba[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_true, positive_probs)

        if len(thresholds) == 0:
            return result

        f1_scores = []
        for threshold in thresholds:
            pred = (positive_probs >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, pred, zero_division=0))

        best_idx = int(np.argmax(f1_scores))
        result.update(
            {
                "is_available": True,
                "best_f1_threshold": round(float(thresholds[best_idx]), 6),
                "best_f1_score": round(float(f1_scores[best_idx]), 6),
            }
        )
        return result
    except Exception:
        return result