from __future__ import annotations

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def run_lof(
    X: pd.DataFrame,
    *,
    contamination: float = 0.02,
    n_neighbors: int = 20,
) -> dict[str, object]:
    effective_neighbors = min(max(5, n_neighbors), max(len(X) - 1, 5))

    model = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=effective_neighbors,
    )
    predictions = model.fit_predict(X)
    scores = model.negative_outlier_factor_

    mask = predictions == -1
    flagged_count = int(mask.sum())
    flagged_percentage = round((flagged_count / max(len(X), 1)) * 100.0, 4)

    return {
        "method": "lof",
        "mask": pd.Series(mask, index=X.index),
        "scores": pd.Series(scores, index=X.index),
        "flagged_count": flagged_count,
        "flagged_percentage": flagged_percentage,
    }