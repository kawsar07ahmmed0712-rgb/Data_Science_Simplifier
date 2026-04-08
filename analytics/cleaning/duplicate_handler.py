from __future__ import annotations

import pandas as pd


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count == 0:
        return df.copy(), 0
    cleaned_df = df.drop_duplicates().reset_index(drop=True)
    return cleaned_df, duplicate_count