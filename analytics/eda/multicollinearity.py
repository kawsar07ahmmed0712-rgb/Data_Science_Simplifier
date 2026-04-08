from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def get_high_correlation_pairs(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    threshold: float = 0.85,
) -> list[dict[str, object]]:
    columns = numeric_columns or [
        column for column in df.columns if is_numeric_dtype(df[column])
    ]

    if len(columns) < 2:
        return []

    corr_df = df[columns].corr()
    results: list[dict[str, object]] = []

    for i, left_col in enumerate(columns):
        for right_col in columns[i + 1 :]:
            corr_value = corr_df.loc[left_col, right_col]
            if pd.isna(corr_value):
                continue

            abs_value = abs(float(corr_value))
            if abs_value >= threshold:
                results.append(
                    {
                        "left": left_col,
                        "right": right_col,
                        "correlation": round(float(corr_value), 6),
                        "abs_correlation": round(abs_value, 6),
                    }
                )

    results.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return results


def summarize_multicollinearity(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    threshold: float = 0.85,
) -> dict[str, object]:
    high_pairs = get_high_correlation_pairs(
        df=df,
        numeric_columns=numeric_columns,
        threshold=threshold,
    )

    involved_columns = sorted(
        {
            item["left"]
            for item in high_pairs
        }.union(
            {
                item["right"]
                for item in high_pairs
            }
        )
    )

    return {
        "threshold": threshold,
        "high_correlation_pairs": high_pairs,
        "involved_columns": involved_columns,
        "pair_count": len(high_pairs),
    }