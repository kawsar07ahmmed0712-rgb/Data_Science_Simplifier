from __future__ import annotations

import pandas as pd

from analytics.schema.datetime_detector import detect_datetime_columns
from analytics.schema.dtype_inference import infer_dtype_label, is_numeric_like
from analytics.schema.id_detector import detect_id_like_columns
from analytics.schema.target_inference import infer_target_candidates
from analytics.schema.text_detector import detect_text_columns
from core.contracts import ColumnSchema, SchemaSummary
from core.enums import ColumnRole
from core.exceptions import SchemaDetectionError


def _detect_categorical_columns(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> list[str]:
    excluded = exclude or set()
    categorical_columns: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue

        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            categorical_columns.append(column)
            continue

        if is_numeric_like(series):
            continue

        nunique = non_null.nunique()
        unique_ratio = nunique / max(len(non_null), 1)

        if nunique <= 50 or unique_ratio <= 0.30:
            categorical_columns.append(column)

    return categorical_columns


def build_schema_summary(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> SchemaSummary:
    if df is None:
        raise SchemaDetectionError("Dataframe is None.")
    if df.empty and len(df.columns) == 0:
        raise SchemaDetectionError("Dataframe has no usable columns.")

    if target_column and target_column not in df.columns:
        raise SchemaDetectionError(
            f"Target column '{target_column}' does not exist in the dataframe."
        )

    target_candidates = infer_target_candidates(df, user_target=target_column)
    resolved_target = target_candidates[0].column if target_candidates else None

    exclude_for_roles = {resolved_target} if resolved_target else set()

    id_like_columns = detect_id_like_columns(df, exclude=exclude_for_roles)
    datetime_columns = detect_datetime_columns(df, exclude=exclude_for_roles)
    text_columns = detect_text_columns(
        df,
        exclude=exclude_for_roles.union(id_like_columns).union(datetime_columns),
    )

    categorical_columns = _detect_categorical_columns(
        df,
        exclude=exclude_for_roles.union(id_like_columns).union(datetime_columns).union(text_columns),
    )

    numeric_columns = [
        column
        for column in df.columns
        if column not in exclude_for_roles
        and column not in id_like_columns
        and column not in datetime_columns
        and column not in text_columns
        and column not in categorical_columns
        and is_numeric_like(df[column])
    ]

    column_schemas: list[ColumnSchema] = []

    for column in df.columns:
        series = df[column]
        notes: list[str] = []
        unique_count = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())

        if resolved_target and column == resolved_target:
            role = ColumnRole.TARGET
            notes.append("selected_as_target")
        elif unique_count <= 1:
            role = ColumnRole.CONSTANT
            notes.append("constant_or_single_value")
        elif column in id_like_columns:
            role = ColumnRole.ID
            notes.append("id_like_column")
        elif column in datetime_columns:
            role = ColumnRole.DATETIME
            notes.append("datetime_like_column")
        elif column in text_columns:
            role = ColumnRole.TEXT
            notes.append("free_text_like_column")
        elif column in categorical_columns:
            role = ColumnRole.CATEGORICAL
            notes.append("categorical_like_column")
        elif column in numeric_columns:
            role = ColumnRole.NUMERIC
            notes.append("numeric_like_column")
        else:
            role = ColumnRole.UNKNOWN
            notes.append("could_not_confidently_classify")

        column_schemas.append(
            ColumnSchema(
                name=column,
                inferred_dtype=infer_dtype_label(series),
                role=role,
                nullable=bool(series.isna().any()),
                unique_count=unique_count,
                missing_count=missing_count,
                notes=notes,
            )
        )

    return SchemaSummary(
        columns=column_schemas,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        text_columns=text_columns,
        id_like_columns=id_like_columns,
        target_candidates=[candidate.column for candidate in target_candidates],
    )