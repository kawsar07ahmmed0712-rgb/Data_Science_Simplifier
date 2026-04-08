from __future__ import annotations

from core.contracts import OutlierSummary


def summarize_outliers(summary: OutlierSummary) -> dict[str, object]:
    severe_columns = [
        item.column
        for item in summary.by_column
        if item.severity.value in {"high", "critical"}
    ]

    actionable_columns = [
        item.column
        for item in summary.by_column
        if item.action.value not in {"ignore", "report_only"}
    ]

    sorted_columns = sorted(
        summary.by_column,
        key=lambda item: item.outlier_percentage,
        reverse=True,
    )

    top_columns = [
        {
            "column": item.column,
            "method": item.method,
            "outlier_percentage": item.outlier_percentage,
            "severity": item.severity.value,
            "action": item.action.value,
        }
        for item in sorted_columns[:15]
    ]

    return {
        "total_flagged_rows": summary.total_flagged_rows,
        "column_count_with_outliers": sum(1 for item in summary.by_column if item.outlier_count > 0),
        "severe_columns": severe_columns,
        "actionable_columns": actionable_columns,
        "top_columns": top_columns,
        "warnings": summary.warnings,
    }