from __future__ import annotations

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


def trim_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned_df = df.copy()
    affected: dict[str, int] = {}

    for column in cleaned_df.columns:
        series = cleaned_df[column]
        if not (is_object_dtype(series) or is_string_dtype(series)):
            continue

        original = series.copy()

        cleaned_df[column] = series.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

        changed_count = int((original != cleaned_df[column]).fillna(False).sum())
        if changed_count > 0:
            affected[column] = changed_count

    return cleaned_df, affected