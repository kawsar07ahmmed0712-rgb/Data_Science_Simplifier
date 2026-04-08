from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from core.exceptions import ModelingError


def build_classifier(model_name: str, params: dict[str, Any]):
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(**params)

    if model_name == "LogisticRegression":
        safe_params = {"max_iter": 1000, **params}
        return LogisticRegression(**safe_params)

    raise ModelingError(f"Unsupported classifier model: {model_name}")


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    model_spec: dict[str, Any],
) -> tuple[object, dict[str, object], dict[str, object]]:
    if X_train is None or X_test is None or y_train is None:
        raise ModelingError("Classifier training requires X_train, X_test, and y_train.")

    try:
        model = build_classifier(
            model_name=str(model_spec["model_name"]),
            params=dict(model_spec.get("params", {})),
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None

        predictions = {
            "y_pred": pd.Series(y_pred, index=X_test.index, name="prediction"),
            "y_proba": y_proba,
        }

        training_log = {
            "model_name": str(model_spec["model_name"]),
            "model_family": str(model_spec.get("model_family", "")),
            "train_rows": int(X_train.shape[0]),
            "train_feature_count": int(X_train.shape[1]),
            "test_rows": int(X_test.shape[0]),
            "classes": [str(label) for label in pd.Series(y_train).dropna().unique().tolist()],
        }

        return model, predictions, training_log
    except Exception as exc:
        raise ModelingError("Failed during classifier training.") from exc