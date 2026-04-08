from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def analyze_skewness(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    moderate_threshold: float = 0.75,
    high_threshold: float = 1.5,
) -> dict[str, object]:
    columns = numeric_columns or [
        column for column in df.columns if is_numeric_dtype(df[column])
    ]

    by_column: dict[str, dict[str, object]] = {}

    for column in columns:
        series = df[column].dropna()
        if series.empty:
            by_column[column] = {
                "skew": None,
                "severity": "unknown",
            }
            continue

        skew_value = float(series.skew())

        if abs(skew_value) >= high_threshold:
            severity = "high"
        elif abs(skew_value) >= moderate_threshold:
            severity = "moderate"
        else:
            severity = "low"

        by_column[column] = {
            "skew": round(skew_value, 6),
            "severity": severity,
        }

    moderately_skewed = [
        column
        for column, info in by_column.items()
        if info["severity"] == "moderate"
    ]
    highly_skewed = [
        column
        for column, info in by_column.items()
        if info["severity"] == "high"
    ]

    return {
        "by_column": by_column,
        "moderately_skewed_columns": moderately_skewed,
        "highly_skewed_columns": highly_skewed,
    }
