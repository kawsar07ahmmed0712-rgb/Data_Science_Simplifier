from __future__ import annotations

import pandas as pd

from analytics.cleaning.cleaning_log import append_log, append_warning, make_log_entry
from analytics.cleaning.constant_column_handler import drop_constant_columns
from analytics.cleaning.duplicate_handler import remove_duplicate_rows
from analytics.cleaning.dtype_fixer import fix_column_dtypes
from analytics.cleaning.invalid_value_handler import normalize_invalid_values
from analytics.cleaning.null_normalizer import normalize_null_markers
from analytics.cleaning.whitespace_cleaner import trim_whitespace
from core.contracts import CleaningSummary, SchemaSummary
from core.exceptions import CleaningError


def clean_dataframe(
    df: pd.DataFrame,
    schema: SchemaSummary | None = None,
    *,
    target_column: str | None = None,
    drop_constant_non_target_columns: bool = True,
) -> tuple[pd.DataFrame, CleaningSummary]:
    if df is None:
        raise CleaningError("Cannot clean a None dataframe.")

    summary = CleaningSummary()
    cleaned_df = df.copy()

    try:
        cleaned_df, whitespace_changes = trim_whitespace(cleaned_df)
        for column, affected_rows in whitespace_changes.items():
            append_log(
                summary,
                make_log_entry(
                    action="trim_whitespace",
                    column=column,
                    affected_rows=affected_rows,
                    details="leading_or_trailing_whitespace_trimmed",
                ),
            )

        cleaned_df, null_changes = normalize_null_markers(cleaned_df)
        for column, affected_rows in null_changes.items():
            append_log(
                summary,
                make_log_entry(
                    action="normalize_null_markers",
                    column=column,
                    affected_rows=affected_rows,
                    details="common_null_markers_replaced_with_nan",
                ),
            )

        cleaned_df, invalid_changes = normalize_invalid_values(cleaned_df)
        for column, affected_rows in invalid_changes.items():
            append_log(
                summary,
                make_log_entry(
                    action="normalize_invalid_values",
                    column=column,
                    affected_rows=affected_rows,
                    details="invalid_or_infinite_values_normalized",
                ),
            )

        cleaned_df, dtype_changes = fix_column_dtypes(
            cleaned_df,
            schema=schema,
            target_column=target_column,
        )
        summary.dtype_fixes.update(dtype_changes)
        for column, new_dtype in dtype_changes.items():
            append_log(
                summary,
                make_log_entry(
                    action="fix_dtype",
                    column=column,
                    details=f"converted_to_{new_dtype}",
                ),
            )

        cleaned_df, duplicates_removed = remove_duplicate_rows(cleaned_df)
        summary.duplicates_removed = duplicates_removed
        if duplicates_removed > 0:
            append_log(
                summary,
                make_log_entry(
                    action="remove_duplicates",
                    affected_rows=duplicates_removed,
                    details="exact_duplicate_rows_removed",
                ),
            )

        if drop_constant_non_target_columns:
            exclude_columns = {target_column} if target_column else set()
            cleaned_df, removed_columns = drop_constant_columns(
                cleaned_df,
                exclude_columns=exclude_columns,
            )
            summary.columns_removed.extend(removed_columns)

            for column in removed_columns:
                append_log(
                    summary,
                    make_log_entry(
                        action="drop_constant_column",
                        column=column,
                        details="column_had_zero_or_one_unique_non_null_value",
                    ),
                )

        if cleaned_df.empty:
            append_warning(summary, "cleaning_resulted_in_zero_rows")

        if len(cleaned_df.columns) == 0:
            append_warning(summary, "cleaning_resulted_in_zero_columns")

        return cleaned_df.reset_index(drop=True), summary

    except Exception as exc:
        raise CleaningError("Failed during dataframe cleaning pipeline.") from exc