from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def get_numeric_univariate_summary(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    columns = numeric_columns or [
        column for column in df.columns if is_numeric_dtype(df[column])
    ]

    result: dict[str, dict[str, float | int | None]] = {}

    for column in columns:
        series = df[column].dropna()
        if series.empty:
            result[column] = {
                "count": 0,
                "missing_count": int(df[column].isna().sum()),
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "q1": None,
                "q3": None,
                "iqr": None,
            }
            continue

        q1 = _safe_float(series.quantile(0.25))
        q3 = _safe_float(series.quantile(0.75))

        result[column] = {
            "count": int(series.shape[0]),
            "missing_count": int(df[column].isna().sum()),
            "mean": _safe_float(series.mean()),
            "median": _safe_float(series.median()),
            "std": _safe_float(series.std()),
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
            "q1": q1,
            "q3": q3,
            "iqr": _safe_float((q3 - q1) if q1 is not None and q3 is not None else None),
        }

    return result


def get_categorical_univariate_summary(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    max_top_values: int = 10,
) -> dict[str, dict[str, object]]:
    if categorical_columns is None:
        categorical_columns = [
            column for column in df.columns if not is_numeric_dtype(df[column])
        ]

    result: dict[str, dict[str, object]] = {}

    for column in categorical_columns:
        series = df[column]
        non_null = series.dropna()

        top_values = (
            non_null.astype(str).value_counts(dropna=False).head(max_top_values).to_dict()
            if not non_null.empty
            else {}
        )

        result[column] = {
            "count": int(non_null.shape[0]),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(non_null.nunique(dropna=True)),
            "top_values": {str(key): int(value) for key, value in top_values.items()},
        }

    return result


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return result
    except (TypeError, ValueError):
        return None