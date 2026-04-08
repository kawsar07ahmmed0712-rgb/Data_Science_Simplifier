from __future__ import annotations

import pandas as pd


def get_unique_counts(df: pd.DataFrame) -> dict[str, int]:
    return {column: int(df[column].nunique(dropna=True)) for column in df.columns}


def get_unique_ratios(df: pd.DataFrame) -> dict[str, float]:
    row_count = max(len(df), 1)
    return {
        column: round(unique_count / row_count, 6)
        for column, unique_count in get_unique_counts(df).items()
    }


def get_constant_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column, unique_count in get_unique_counts(df).items()
        if unique_count <= 1
    ]