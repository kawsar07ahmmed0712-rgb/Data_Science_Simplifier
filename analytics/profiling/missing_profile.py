from __future__ import annotations

import pandas as pd


def get_missing_counts(df: pd.DataFrame) -> dict[str, int]:
    return {column: int(value) for column, value in df.isna().sum().to_dict().items()}


def get_missing_percentages(df: pd.DataFrame) -> dict[str, float]:
    row_count = max(len(df), 1)
    return {
        column: round((count / row_count) * 100.0, 4)
        for column, count in get_missing_counts(df).items()
    }


def get_missing_columns(
    df: pd.DataFrame,
    min_missing_percentage: float = 0.0,
) -> list[str]:
    percentages = get_missing_percentages(df)
    return [
        column
        for column, percentage in percentages.items()
        if percentage > min_missing_percentage
    ]