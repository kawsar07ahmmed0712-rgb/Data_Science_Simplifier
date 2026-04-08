from __future__ import annotations

import re

import pandas as pd
from pandas.api.types import is_integer_dtype, is_object_dtype, is_string_dtype

_ID_NAME_PATTERN = re.compile(
    r"(^id$|_id$|^id_|identifier|uuid|guid|customer_id|order_id|transaction_id)",
    re.IGNORECASE,
)


def _non_null(series: pd.Series) -> pd.Series:
    return series.dropna()


def _is_monotonic_integer_identifier(series: pd.Series) -> bool:
    non_null = _non_null(series)
    if non_null.empty or not is_integer_dtype(non_null):
        return False
    return bool(non_null.is_monotonic_increasing and non_null.nunique() == len(non_null))


def is_id_like_column(series: pd.Series, column_name: str) -> bool:
    non_null = _non_null(series)
    if non_null.empty:
        return False

    unique_ratio = non_null.nunique() / max(len(non_null), 1)

    if _ID_NAME_PATTERN.search(column_name):
        return True

    if _is_monotonic_integer_identifier(non_null):
        return True

    if unique_ratio >= 0.98 and len(non_null) >= 20:
        if is_integer_dtype(non_null):
            return True
        if is_object_dtype(non_null) or is_string_dtype(non_null):
            avg_length = non_null.astype(str).str.len().mean()
            if avg_length >= 6:
                return True

    return False


def detect_id_like_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    detected: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue
        if is_id_like_column(df[column], column):
            detected.append(column)

    return detected