from __future__ import annotations

import pandas as pd


def winsorize_series(
    series: pd.Series,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> pd.Series:
    clipped = series.copy()
    if lower_bound is not None:
        clipped = clipped.clip(lower=lower_bound)
    if upper_bound is not None:
        clipped = clipped.clip(upper=upper_bound)
    return clipped


def winsorize_dataframe(
    df: pd.DataFrame,
    bounds_by_column: dict[str, dict[str, float | None]],
) -> pd.DataFrame:
    result = df.copy()
    for column, bounds in bounds_by_column.items():
        if column not in result.columns:
            continue
        result[column] = winsorize_series(
            result[column],
            lower_bound=bounds.get("lower_bound"),
            upper_bound=bounds.get("upper_bound"),
        )
    return result