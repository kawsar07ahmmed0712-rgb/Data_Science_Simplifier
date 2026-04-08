from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import bullet_lines, format_scalar, markdown_table


def build_data_quality_section(state: AnalysisState) -> str:
    profile = state.profile
    cleaning = state.cleaning
    profile_details = state.metadata.get("raw_profile_details", {})

    missing_rows = [
        [column, count, round(profile.missing_percentages.get(column, 0.0), 4)]
        for column, count in sorted(
            profile.missing_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if count > 0
    ][:12]

    uniqueness_rows = [
        [
            column,
            profile.unique_counts.get(column, 0),
            round(profile_details.get("unique_ratios", {}).get(column, 0.0), 4),
        ]
        for column in sorted(
            profile.unique_counts,
            key=lambda name: profile.unique_counts.get(name, 0),
            reverse=True,
        )[:12]
    ]

    lines = [
        "## Step 2 Missingness and Uniqueness Audit",
        f"Duplicate rows detected: {profile.duplicate_count}",
        f"Memory usage (bytes): {profile.memory_usage_bytes}",
        f"Cleaning warnings: {format_scalar(cleaning.warnings[:6])}",
        "",
        "### Missingness Summary",
        markdown_table(["Column", "Missing count", "Missing %"], missing_rows),
        "",
        "### Uniqueness Summary",
        markdown_table(["Column", "Unique count", "Unique ratio"], uniqueness_rows),
        "",
        "### Cleaning Actions",
        bullet_lines(
            [
                f"dtype_fixes: {format_scalar(cleaning.dtype_fixes)}",
                f"columns_removed: {format_scalar(cleaning.columns_removed)}",
                f"duplicates_removed: {cleaning.duplicates_removed}",
                f"applied_steps: {len(cleaning.steps_applied)}",
            ]
        ),
    ]

    return "\n".join(lines)
