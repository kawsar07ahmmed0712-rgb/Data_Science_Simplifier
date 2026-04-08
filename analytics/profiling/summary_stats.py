from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def get_numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    numeric_columns = [column for column in df.columns if is_numeric_dtype(df[column])]
    if not numeric_columns:
        return {}

    described = df[numeric_columns].describe().transpose()
    result: dict[str, dict[str, float | int | None]] = {}

    for column in described.index:
        row = described.loc[column]
        result[column] = {
            "count": float(row.get("count", 0.0)),
            "mean": _safe_float(row.get("mean")),
            "std": _safe_float(row.get("std")),
            "min": _safe_float(row.get("min")),
            "25%": _safe_float(row.get("25%")),
            "50%": _safe_float(row.get("50%")),
            "75%": _safe_float(row.get("75%")),
            "max": _safe_float(row.get("max")),
            "skew": _safe_float(df[column].skew()),
        }

    return result


def get_categorical_summary(
    df: pd.DataFrame,
    max_top_categories: int = 10,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}

    for column in df.columns:
        if is_numeric_dtype(df[column]):
            continue

        series = df[column]
        non_null = series.dropna()
        value_counts = non_null.astype(str).value_counts().head(max_top_categories)

        mode_value = None
        if not non_null.empty:
            modes = non_null.mode(dropna=True)
            if not modes.empty:
                mode_value = str(modes.iloc[0])

        result[column] = {
            "count": int(non_null.shape[0]),
            "unique": int(non_null.nunique(dropna=True)),
            "mode": mode_value,
            "top_values": {str(index): int(value) for index, value in value_counts.items()},
        }

    return result


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
        if pd.isna(value_float):
            return None
        return value_float
    except (TypeError, ValueError):
        return None