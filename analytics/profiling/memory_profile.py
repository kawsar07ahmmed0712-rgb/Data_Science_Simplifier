from __future__ import annotations

import pandas as pd


def get_memory_usage_bytes(df: pd.DataFrame, deep: bool = True) -> int:
    return int(df.memory_usage(deep=deep).sum())


def get_column_memory_usage_bytes(
    df: pd.DataFrame,
    deep: bool = True,
) -> dict[str, int]:
    usage = df.memory_usage(deep=deep)
    return {
        str(index): int(value)
        for index, value in usage.to_dict().items()
        if str(index) != "Index"
    }