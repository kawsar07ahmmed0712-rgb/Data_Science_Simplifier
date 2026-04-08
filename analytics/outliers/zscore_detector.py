from __future__ import annotations

import pandas as pd


def detect_zscore_outliers(
    series: pd.Series,
    threshold: float = 3.0,
) -> dict[str, object]:
    clean = series.dropna()
    if clean.empty:
        return {
            "method": "zscore",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "mask": pd.Series(False, index=series.index),
        }

    mean_value = clean.mean()
    std_value = clean.std()

    if pd.isna(std_value) or std_value == 0:
        return {
            "method": "zscore",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "mask": pd.Series(False, index=series.index),
        }

    zscores = (series - mean_value) / std_value
    mask = zscores.abs() > threshold
    mask = mask.fillna(False)

    non_null_count = max(int(clean.shape[0]), 1)
    outlier_count = int(mask.sum())
    outlier_percentage = round((outlier_count / non_null_count) * 100.0, 4)

    lower_bound = mean_value - threshold * std_value
    upper_bound = mean_value + threshold * std_value

    return {
        "method": "zscore",
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage,
        "mask": mask,
    }