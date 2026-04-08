from __future__ import annotations

import pandas as pd

from core.contracts import SchemaSummary


def select_feature_groups(
    df: pd.DataFrame,
    *,
    schema: SchemaSummary | None = None,
    target_column: str | None = None,
    drop_columns: set[str] | None = None,
) -> dict[str, list[str]]:
    drops = set(drop_columns or set())
    if target_column:
        drops.add(target_column)

    numeric_columns = [
        column
        for column in (schema.numeric_columns if schema else [])
        if column in df.columns and column not in drops
    ]
    categorical_columns = [
        column
        for column in (schema.categorical_columns if schema else [])
        if column in df.columns and column not in drops
    ]
    datetime_columns = [
        column
        for column in (schema.datetime_columns if schema else [])
        if column in df.columns and column not in drops
    ]
    text_columns = [
        column
        for column in (schema.text_columns if schema else [])
        if column in df.columns and column not in drops
    ]
    id_like_columns = [
        column
        for column in (schema.id_like_columns if schema else [])
        if column in df.columns and column not in drops
    ]

    if schema is None:
        remaining_columns = [column for column in df.columns if column not in drops]
        numeric_columns = [
            column for column in remaining_columns if pd.api.types.is_numeric_dtype(df[column])
        ]
        categorical_columns = [
            column
            for column in remaining_columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]
        datetime_columns = []
        text_columns = []
        id_like_columns = []

    used_columns = set(numeric_columns + categorical_columns + datetime_columns + text_columns + id_like_columns)
    leftover_columns = [
        column
        for column in df.columns
        if column not in used_columns and column not in drops
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "text_columns": text_columns,
        "id_like_columns": id_like_columns,
        "leftover_columns": leftover_columns,
        "dropped_columns": sorted(drops),
    }