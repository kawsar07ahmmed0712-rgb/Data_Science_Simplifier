from __future__ import annotations

import pandas as pd


def drop_constant_columns(
    df: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    excluded = exclude_columns or set()
    removable_columns: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue
        unique_count = int(df[column].nunique(dropna=True))
        if unique_count <= 1:
            removable_columns.append(column)

    if not removable_columns:
        return df.copy(), []

    cleaned_df = df.drop(columns=removable_columns)
    return cleaned_df, removable_columns