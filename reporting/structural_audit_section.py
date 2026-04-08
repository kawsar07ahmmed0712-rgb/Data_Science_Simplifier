from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import bullet_lines, format_scalar, markdown_table, top_items


def build_structural_audit_section(state: AnalysisState) -> str:
    schema = state.schema
    profile_details = state.metadata.get("raw_profile_details", {})
    columns = top_items(schema.columns, limit=18)

    column_rows = [
        [
            column.name,
            column.role.value,
            column.inferred_dtype,
            "Yes" if column.nullable else "No",
            column.missing_count,
            column.unique_count,
            ", ".join(column.notes[:3]) or "N/A",
        ]
        for column in columns
    ]

    memory_usage = profile_details.get("column_memory_usage_bytes", {})
    memory_rows = [
        [column, bytes_used]
        for column, bytes_used in sorted(
            memory_usage.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]

    lines = [
        "## Step 1 Structural Audit",
        f"Raw shape: {format_scalar(state.raw_dataframe_shape)}",
        f"Cleaned shape: {format_scalar(state.cleaned_dataframe_shape)}",
        f"Target candidates: {format_scalar(schema.target_candidates[:5])}",
        f"Numeric columns: {len(schema.numeric_columns)}",
        f"Categorical columns: {len(schema.categorical_columns)}",
        f"Datetime columns: {len(schema.datetime_columns)}",
        f"Text columns: {len(schema.text_columns)}",
        f"ID-like columns: {len(schema.id_like_columns)}",
        "",
        "### Column Role and Datatype Audit",
        markdown_table(
            ["Column", "Role", "Inferred dtype", "Nullable", "Missing", "Unique", "Notes"],
            column_rows,
        ),
        "",
        "### Structural Flags",
        bullet_lines(
            [
                f"id_like_columns: {format_scalar(schema.id_like_columns[:8])}",
                f"datetime_columns: {format_scalar(schema.datetime_columns[:8])}",
                f"text_columns: {format_scalar(schema.text_columns[:8])}",
                f"constant_columns: {format_scalar(profile_details.get('constant_columns', [])[:8])}",
            ]
        ),
        "",
        "### Column Memory Footprint",
        markdown_table(["Column", "Memory bytes"], memory_rows),
    ]

    if len(schema.columns) > len(columns):
        lines.extend(
            [
                "",
                f"_Showing {len(columns)} of {len(schema.columns)} columns in the structural table._",
            ]
        )

    return "\n".join(lines)
