from __future__ import annotations

import pandas as pd


def detect_mad_outliers(
    series: pd.Series,
    threshold: float = 3.5,
) -> dict[str, object]:
    clean = series.dropna()
    if clean.empty:
        return {
            "method": "mad",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "mask": pd.Series(False, index=series.index),
        }

    median_value = clean.median()
    abs_dev = (clean - median_value).abs()
    mad = abs_dev.median()

    if pd.isna(mad) or mad == 0:
        return {
            "method": "mad",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "mask": pd.Series(False, index=series.index),
        }

    modified_zscore = 0.6745 * (series - median_value).abs() / mad
    mask = modified_zscore > threshold
    mask = mask.fillna(False)

    non_null_count = max(int(clean.shape[0]), 1)
    outlier_count = int(mask.sum())
    outlier_percentage = round((outlier_count / non_null_count) * 100.0, 4)

    robust_std_approx = mad / 0.6745
    lower_bound = median_value - threshold * robust_std_approx
    upper_bound = median_value + threshold * robust_std_approx

    return {
        "method": "mad",
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage,
        "mask": mask,
    }