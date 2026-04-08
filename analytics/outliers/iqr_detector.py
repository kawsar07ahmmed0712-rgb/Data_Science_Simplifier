from __future__ import annotations

import pandas as pd


def detect_iqr_outliers(
    series: pd.Series,
    multiplier: float = 1.5,
) -> dict[str, object]:
    clean = series.dropna()
    if clean.empty:
        return {
            "method": "iqr",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "mask": pd.Series(False, index=series.index),
        }

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr):
        return {
            "method": "iqr",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "mask": pd.Series(False, index=series.index),
        }

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    mask = (series < lower_bound) | (series > upper_bound)
    mask = mask.fillna(False)

    non_null_count = max(int(clean.shape[0]), 1)
    outlier_count = int(mask.sum())
    outlier_percentage = round((outlier_count / non_null_count) * 100.0, 4)

    return {
        "method": "iqr",
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage,
        "mask": mask,
    }