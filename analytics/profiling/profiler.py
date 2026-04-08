from __future__ import annotations

import pandas as pd

from analytics.profiling.duplicates_profile import get_duplicate_count
from analytics.profiling.memory_profile import (
    get_column_memory_usage_bytes,
    get_memory_usage_bytes,
)
from analytics.profiling.missing_profile import (
    get_missing_counts,
    get_missing_percentages,
)
from analytics.profiling.profile_formatter import (
    build_profile_summary,
    build_profile_warnings,
)
from analytics.profiling.summary_stats import (
    get_categorical_summary,
    get_numeric_summary,
)
from analytics.profiling.uniqueness_profile import (
    get_constant_columns,
    get_unique_counts,
    get_unique_ratios,
)
from core.contracts import ProfileSummary, SchemaSummary
from core.exceptions import ProfilingError


def profile_dataframe(
    df: pd.DataFrame,
    schema: SchemaSummary | None = None,
) -> tuple[ProfileSummary, dict[str, object]]:
    if df is None:
        raise ProfilingError("Cannot profile a None dataframe.")

    try:
        row_count = int(df.shape[0])
        column_count = int(df.shape[1])
        duplicate_count = get_duplicate_count(df)
        missing_counts = get_missing_counts(df)
        missing_percentages = get_missing_percentages(df)
        unique_counts = get_unique_counts(df)
        memory_usage_bytes = get_memory_usage_bytes(df)
        dtypes = {column: str(dtype) for column, dtype in df.dtypes.astype(str).to_dict().items()}

        warnings = build_profile_warnings(
            row_count=row_count,
            column_count=column_count,
            duplicate_count=duplicate_count,
            missing_percentages=missing_percentages,
            schema=schema,
        )

        profile_summary = build_profile_summary(
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

        detailed_profile: dict[str, object] = {
            "shape": df.shape,
            "column_order": list(df.columns),
            "unique_ratios": get_unique_ratios(df),
            "constant_columns": get_constant_columns(df),
            "column_memory_usage_bytes": get_column_memory_usage_bytes(df),
            "numeric_summary": get_numeric_summary(df),
            "categorical_summary": get_categorical_summary(df),
        }

        return profile_summary, detailed_profile
    except Exception as exc:
        raise ProfilingError("Failed to profile dataframe.") from exc