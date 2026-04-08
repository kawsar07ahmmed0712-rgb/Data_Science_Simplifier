from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analytics.outliers.iqr_detector import detect_iqr_outliers
from analytics.outliers.mad_detector import detect_mad_outliers
from analytics.outliers.zscore_detector import detect_zscore_outliers


def choose_outlier_method(
    series: pd.Series,
    *,
    skew_severity: str | None = None,
) -> str:
    clean = series.dropna()
    if clean.empty:
        return "iqr"

    unique_count = int(clean.nunique(dropna=True))
    if unique_count <= 5:
        return "iqr"

    if skew_severity == "high":
        return "mad"

    if skew_severity == "moderate":
        return "iqr"

    return "zscore"


def detect_outliers_for_column(
    series: pd.Series,
    *,
    skew_severity: str | None = None,
) -> dict[str, object]:
    if not is_numeric_dtype(series):
        return {
            "method": "unsupported",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "mask": pd.Series(False, index=series.index),
        }

    method = choose_outlier_method(series=series, skew_severity=skew_severity)

    if method == "mad":
        return detect_mad_outliers(series)
    if method == "zscore":
        return detect_zscore_outliers(series)
    return detect_iqr_outliers(series)