from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import bullet_lines, format_scalar, markdown_table


def build_outlier_section(state: AnalysisState) -> str:
    outlier_report = state.metadata.get("outlier_report", {})
    anomaly_summary = state.metadata.get("anomaly_summary", {})
    skewness = state.eda_results.get("skewness_analysis", {}) if state.eda_results else {}
    multicollinearity = state.eda_results.get("multicollinearity", {}) if state.eda_results else {}

    outlier_rows = [
        [
            item.get("column"),
            item.get("method"),
            item.get("outlier_percentage"),
            item.get("severity"),
            item.get("action"),
        ]
        for item in outlier_report.get("top_columns", [])[:12]
    ]

    anomaly_rows = [
        [item.get("method"), item.get("flagged_count"), item.get("flagged_percentage")]
        for item in anomaly_summary.get("methods_ranked", [])[:8]
    ]

    lines = [
        "## Step 4 Numerical Robustness",
        f"Total outlier-flagged rows: {outlier_report.get('total_flagged_rows', 0)}",
        f"Combined anomaly rows: {anomaly_summary.get('combined_flagged_rows', 0)}",
        "",
        "### Outlier Summary",
        markdown_table(
            ["Column", "Method", "Outlier %", "Severity", "Action"],
            outlier_rows,
        ),
        "",
        "### Anomaly Detector Summary",
        markdown_table(["Method", "Flagged rows", "Flagged %"], anomaly_rows),
        "",
        "### Shape and Collinearity Flags",
        bullet_lines(
            [
                f"highly_skewed_columns: {format_scalar(skewness.get('highly_skewed_columns', [])[:10])}",
                f"moderately_skewed_columns: {format_scalar(skewness.get('moderately_skewed_columns', [])[:10])}",
                f"multicollinearity_pairs: {multicollinearity.get('pair_count', 0)}",
                f"multicollinearity_columns: {format_scalar(multicollinearity.get('involved_columns', [])[:10])}",
            ]
        ),
    ]
    return "\n".join(lines)
