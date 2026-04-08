from __future__ import annotations

import pandas as pd


def generate_segmentation_hints(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    max_unique_for_segmentation: int = 20,
    min_non_null_rows: int = 30,
) -> list[dict[str, object]]:
    if categorical_columns is None:
        categorical_columns = [
            column
            for column in df.columns
            if str(df[column].dtype) in {"object", "string", "category"}
        ]

    hints: list[dict[str, object]] = []

    for column in categorical_columns:
        non_null = df[column].dropna()
        if len(non_null) < min_non_null_rows:
            continue

        unique_count = int(non_null.nunique(dropna=True))
        if unique_count < 2 or unique_count > max_unique_for_segmentation:
            continue

        top_values = non_null.astype(str).value_counts().head(10).to_dict()
        hints.append(
            {
                "column": column,
                "unique_count": unique_count,
                "top_segments": {str(k): int(v) for k, v in top_values.items()},
                "reason": "good_candidate_for_groupwise_analysis",
            }
        )

    return hints