from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def run_isolation_forest(
    X: pd.DataFrame,
    *,
    contamination: float = 0.02,
    random_state: int = 42,
) -> dict[str, object]:
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    mask = predictions == -1
    flagged_count = int(mask.sum())
    flagged_percentage = round((flagged_count / max(len(X), 1)) * 100.0, 4)

    return {
        "method": "isolation_forest",
        "mask": pd.Series(mask, index=X.index),
        "scores": pd.Series(scores, index=X.index),
        "flagged_count": flagged_count,
        "flagged_percentage": flagged_percentage,
    }