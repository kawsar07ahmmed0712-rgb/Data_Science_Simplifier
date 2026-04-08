from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

DEFAULT_NULL_MARKERS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    "?",
    "-",
    "--",
}


def normalize_null_markers(
    df: pd.DataFrame,
    null_markers: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    markers = {marker.lower() for marker in (null_markers or DEFAULT_NULL_MARKERS)}
    normalized_df = df.copy()
    affected: dict[str, int] = {}

    for column in normalized_df.columns:
        series = normalized_df[column]
        if not (is_object_dtype(series) or is_string_dtype(series)):
            continue

        original_missing_count = int(series.isna().sum())

        normalized_df[column] = series.map(
            lambda value: np.nan
            if isinstance(value, str) and value.strip().lower() in markers
            else value
        )

        new_missing_count = int(normalized_df[column].isna().sum())
        delta = new_missing_count - original_missing_count
        if delta > 0:
            affected[column] = delta

    return normalized_df, affected