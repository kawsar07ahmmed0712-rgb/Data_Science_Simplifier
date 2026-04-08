from __future__ import annotations

import pandas as pd


def get_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def get_duplicate_percentage(df: pd.DataFrame) -> float:
    row_count = max(len(df), 1)
    return round((get_duplicate_count(df) / row_count) * 100.0, 4)