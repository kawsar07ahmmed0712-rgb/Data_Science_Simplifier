from __future__ import annotations

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from core.contracts import SchemaSummary


def fix_column_dtypes(
    df: pd.DataFrame,
    schema: SchemaSummary | None = None,
    *,
    target_column: str | None = None,
    numeric_threshold: float = 0.90,
    datetime_threshold: float = 0.75,
) -> tuple[pd.DataFrame, dict[str, str]]:
    fixed_df = df.copy()
    dtype_changes: dict[str, str] = {}

    datetime_columns = set(schema.datetime_columns if schema else [])
    text_columns = set(schema.text_columns if schema else [])
    categorical_columns = set(schema.categorical_columns if schema else [])
    id_like_columns = set(schema.id_like_columns if schema else [])

    for column in fixed_df.columns:
        if column == target_column:
            continue

        original_dtype = str(fixed_df[column].dtype)
        series = fixed_df[column]

        if column in datetime_columns:
            converted = pd.to_datetime(series, errors="coerce")
            success_ratio = converted.notna().mean()
            if success_ratio >= datetime_threshold:
                fixed_df[column] = converted
                dtype_changes[column] = str(fixed_df[column].dtype)
            continue

        if column in text_columns or column in categorical_columns or column in id_like_columns:
            continue

        if is_object_dtype(series) or is_string_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                continue

            numeric_candidate = pd.to_numeric(non_null, errors="coerce")
            numeric_success_ratio = numeric_candidate.notna().mean()

            if numeric_success_ratio >= numeric_threshold:
                fixed_df[column] = pd.to_numeric(series, errors="coerce")
                new_dtype = str(fixed_df[column].dtype)
                if new_dtype != original_dtype:
                    dtype_changes[column] = new_dtype

    return fixed_df, dtype_changes