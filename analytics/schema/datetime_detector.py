from __future__ import annotations

import re

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

_DATETIME_NAME_PATTERN = re.compile(
    r"(date|time|timestamp|created|updated|modified|signup|joined|dob|birth)",
    re.IGNORECASE,
)


def _sample_non_null(series: pd.Series, max_sample: int = 200) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) <= max_sample:
        return non_null
    return non_null.sample(n=max_sample, random_state=42)


def is_datetime_like_column(series: pd.Series, column_name: str) -> bool:
    if is_datetime64_any_dtype(series):
        return True

    sample = _sample_non_null(series)
    if sample.empty:
        return False

    name_hint = bool(_DATETIME_NAME_PATTERN.search(column_name))
    parsed = pd.to_datetime(sample.astype(str), errors="coerce", infer_datetime_format=True)
    success_ratio = parsed.notna().mean()

    if name_hint and success_ratio >= 0.30:
        return True

    return success_ratio >= 0.75


def detect_datetime_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    detected: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue
        if is_datetime_like_column(df[column], column):
            detected.append(column)

    return detected