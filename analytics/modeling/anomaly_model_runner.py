from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from core.exceptions import ModelingError


def run_anomaly_model(
    X: pd.DataFrame,
    *,
    contamination: float = 0.02,
    random_state: int = 42,
) -> tuple[object, dict[str, object], dict[str, object]]:
    if X is None or X.empty:
        raise ModelingError("Anomaly model requires a non-empty feature matrix.")

    try:
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=300,
        )
        labels = model.fit_predict(X)
        scores = model.decision_function(X)

        predictions = {
            "anomaly_label": pd.Series(labels, index=X.index, name="anomaly_label"),
            "anomaly_score": pd.Series(scores, index=X.index, name="anomaly_score"),
        }

        training_log = {
            "model_name": "IsolationForest",
            "model_family": "anomaly",
            "rows": int(X.shape[0]),
            "feature_count": int(X.shape[1]),
            "contamination": float(contamination),
        }

        return model, predictions, training_log
    except Exception as exc:
        raise ModelingError("Failed during anomaly model run.") from exc