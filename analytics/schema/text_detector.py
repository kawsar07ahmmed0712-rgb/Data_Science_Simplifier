from __future__ import annotations

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


def _sample_non_null(series: pd.Series, max_sample: int = 500) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) <= max_sample:
        return non_null
    return non_null.sample(n=max_sample, random_state=42)


def is_text_like_column(series: pd.Series) -> bool:
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    sample = _sample_non_null(series)
    if sample.empty:
        return False

    sample_as_str = sample.astype(str).str.strip()
    avg_length = sample_as_str.str.len().mean()
    unique_ratio = sample_as_str.nunique() / max(len(sample_as_str), 1)

    return avg_length >= 25 and unique_ratio >= 0.30


def detect_text_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    detected: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue
        if is_text_like_column(df[column]):
            detected.append(column)

    return detected