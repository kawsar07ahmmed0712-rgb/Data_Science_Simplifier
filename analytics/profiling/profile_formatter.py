from __future__ import annotations

from core.contracts import ProfileSummary, SchemaSummary


def build_profile_warnings(
    *,
    row_count: int,
    column_count: int,
    duplicate_count: int,
    missing_percentages: dict[str, float],
    schema: SchemaSummary | None = None,
) -> list[str]:
    warnings: list[str] = []

    if row_count == 0:
        warnings.append("dataset_has_zero_rows")

    if column_count == 0:
        warnings.append("dataset_has_zero_columns")

    if duplicate_count > 0:
        warnings.append("dataset_contains_duplicate_rows")

    high_missing_columns = [
        column
        for column, percentage in missing_percentages.items()
        if percentage >= 40.0
    ]
    if high_missing_columns:
        warnings.append("high_missingness_detected")

    if schema:
        if schema.id_like_columns:
            warnings.append("id_like_columns_detected")
        if schema.datetime_columns:
            warnings.append("datetime_columns_detected")
        if schema.text_columns:
            warnings.append("free_text_columns_detected")

    return warnings


def build_profile_summary(
    *,
    row_count: int,
    column_count: int,
    duplicate_count: int,
    missing_counts: dict[str, int],
    missing_percentages: dict[str, float],
    unique_counts: dict[str, int],
    dtypes: dict[str, str],
    memory_usage_bytes: int,
    warnings: list[str],
) -> ProfileSummary:
    return ProfileSummary(
        row_count=row_count,
        column_count=column_count,
        duplicate_count=duplicate_count,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        unique_counts=unique_counts,
        dtypes=dtypes,
        memory_usage_bytes=memory_usage_bytes,
        warnings=warnings,
    )