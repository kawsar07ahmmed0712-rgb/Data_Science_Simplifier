from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize_invalid_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned_df = df.copy()
    affected: dict[str, int] = {}

    for column in cleaned_df.columns:
        series = cleaned_df[column]

        if is_numeric_dtype(series):
            original_invalid_count = int(np.isinf(series.to_numpy(dtype=float, copy=True)).sum())
            cleaned_df[column] = series.replace([np.inf, -np.inf], np.nan)
            if original_invalid_count > 0:
                affected[column] = original_invalid_count
            continue

        original = series.copy()
        cleaned_df[column] = series.map(
            lambda value: np.nan
            if isinstance(value, str) and not value.strip()
            else value
        )

        changed_count = int(
            ((original.isna() != cleaned_df[column].isna()) | (original != cleaned_df[column]))
            .fillna(False)
            .sum()
        )
        if changed_count > 0:
            affected[column] = changed_count

    return cleaned_df, affected
