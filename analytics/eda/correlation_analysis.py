from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def compute_correlation_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    method: str = "pearson",
) -> dict[str, dict[str, float]]:
    columns = numeric_columns or [
        column for column in df.columns if is_numeric_dtype(df[column])
    ]

    if len(columns) < 2:
        return {}

    corr_df = df[columns].corr(method=method)

    result: dict[str, dict[str, float]] = {}
    for row in corr_df.index:
        result[str(row)] = {}
        for col in corr_df.columns:
            value = corr_df.loc[row, col]
            result[str(row)][str(col)] = _safe_float(value) or 0.0

    return result


def get_top_correlated_pairs(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    top_n: int = 20,
    min_abs_correlation: float = 0.20,
) -> list[dict[str, object]]:
    columns = numeric_columns or [
        column for column in df.columns if is_numeric_dtype(df[column])
    ]

    if len(columns) < 2:
        return []

    corr_df = df[columns].corr()
    pairs: list[dict[str, object]] = []

    for i, left_col in enumerate(columns):
        for right_col in columns[i + 1 :]:
            value = corr_df.loc[left_col, right_col]
            if pd.isna(value):
                continue

            abs_value = abs(float(value))
            if abs_value < min_abs_correlation:
                continue

            pairs.append(
                {
                    "left": left_col,
                    "right": right_col,
                    "correlation": round(float(value), 6),
                    "abs_correlation": round(abs_value, 6),
                }
            )

    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return pairs[:top_n]


def get_target_correlations(
    df: pd.DataFrame,
    target_column: str | None,
    numeric_columns: list[str] | None = None,
    top_n: int = 15,
) -> list[dict[str, object]]:
    if not target_column or target_column not in df.columns:
        return []

    if not is_numeric_dtype(df[target_column]):
        return []

    columns = numeric_columns or [
        column for column in df.columns if is_numeric_dtype(df[column])
    ]
    usable_columns = [column for column in columns if column != target_column]

    if not usable_columns:
        return []

    corr_series = df[usable_columns + [target_column]].corr()[target_column].drop(target_column)
    corr_series = corr_series.dropna()

    pairs = [
        {
            "feature": feature,
            "correlation_with_target": round(float(value), 6),
            "abs_correlation": round(abs(float(value)), 6),
        }
        for feature, value in corr_series.items()
    ]

    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return pairs[:top_n]


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return round(result, 6)
    except (TypeError, ValueError):
        return None