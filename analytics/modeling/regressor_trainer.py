from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from core.exceptions import ModelingError


def build_regressor(model_name: str, params: dict[str, Any]):
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)

    if model_name == "LinearRegression":
        return LinearRegression(**params)

    raise ModelingError(f"Unsupported regressor model: {model_name}")


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    model_spec: dict[str, Any],
) -> tuple[object, dict[str, object], dict[str, object]]:
    if X_train is None or X_test is None or y_train is None:
        raise ModelingError("Regressor training requires X_train, X_test, and y_train.")

    try:
        model = build_regressor(
            model_name=str(model_spec["model_name"]),
            params=dict(model_spec.get("params", {})),
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        predictions = {
            "y_pred": pd.Series(y_pred, index=X_test.index, name="prediction"),
        }

        training_log = {
            "model_name": str(model_spec["model_name"]),
            "model_family": str(model_spec.get("model_family", "")),
            "train_rows": int(X_train.shape[0]),
            "train_feature_count": int(X_train.shape[1]),
            "test_rows": int(X_test.shape[0]),
        }

        return model, predictions, training_log
    except Exception as exc:
        raise ModelingError("Failed during regressor training.") from exc