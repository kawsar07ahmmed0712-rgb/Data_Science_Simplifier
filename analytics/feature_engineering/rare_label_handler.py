from __future__ import annotations

import pandas as pd


def summarize_rare_labels(
    df: pd.DataFrame,
    categorical_columns: list[str],
    *,
    rare_threshold: float = 0.01,
    max_top_values: int = 10,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}

    for column in categorical_columns:
        if column not in df.columns:
            continue

        non_null = df[column].dropna().astype(str)
        if non_null.empty:
            summary[column] = {
                "unique_count": 0,
                "rare_labels": [],
                "rare_label_count": 0,
                "top_values": {},
            }
            continue

        value_counts = non_null.value_counts(normalize=True)
        rare_labels = value_counts[value_counts < rare_threshold].index.tolist()

        summary[column] = {
            "unique_count": int(non_null.nunique(dropna=True)),
            "rare_labels": [str(label) for label in rare_labels],
            "rare_label_count": int(len(rare_labels)),
            "top_values": {
                str(k): float(v)
                for k, v in value_counts.head(max_top_values).round(6).to_dict().items()
            },
        }

    return summary


def detect_high_cardinality_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
    *,
    unique_threshold: int = 50,
    unique_ratio_threshold: float = 0.50,
) -> list[str]:
    high_cardinality: list[str] = []

    for column in categorical_columns:
        if column not in df.columns:
            continue

        non_null = df[column].dropna()
        if non_null.empty:
            continue

        unique_count = int(non_null.nunique(dropna=True))
        unique_ratio = unique_count / max(len(non_null), 1)

        if unique_count >= unique_threshold or unique_ratio >= unique_ratio_threshold:
            high_cardinality.append(column)

    return high_cardinality