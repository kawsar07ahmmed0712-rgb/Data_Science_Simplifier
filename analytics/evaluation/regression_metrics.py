from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = mse ** 0.5

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
    }