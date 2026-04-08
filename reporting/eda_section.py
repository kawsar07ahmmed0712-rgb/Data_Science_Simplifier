from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import bullet_lines, format_scalar, markdown_table


def _categorical_profile_rows(categorical_summary: dict[str, dict[str, object]]) -> list[list[object]]:
    rows: list[list[object]] = []
    for column, stats in list(categorical_summary.items())[:12]:
        rows.append(
            [
                column,
                stats.get("count"),
                stats.get("unique") or stats.get("unique_count"),
                stats.get("mode"),
                format_scalar(stats.get("top_values", {})),
            ]
        )
    return rows


def _numeric_profile_rows(numeric_summary: dict[str, dict[str, object]]) -> list[list[object]]:
    rows: list[list[object]] = []
    for column, stats in list(numeric_summary.items())[:12]:
        rows.append(
            [
                column,
                stats.get("mean"),
                stats.get("std"),
                stats.get("min"),
                stats.get("50%") or stats.get("median"),
                stats.get("max"),
                stats.get("skew"),
            ]
        )
    return rows


def build_eda_section(state: AnalysisState) -> str:
    eda = state.eda_results or {}
    profile_details = state.metadata.get("raw_profile_details", {})
    categorical_summary = profile_details.get("categorical_summary", {})
    numeric_summary = profile_details.get("numeric_summary", {})
    target_analysis = eda.get("target_analysis", {})

    categorical_signal_rows = [
        [item.get("feature"), format_scalar(item.get("row_percentages", {}))]
        for item in eda.get("bivariate_analysis", {}).get("categorical_vs_target", [])[:8]
    ]
    numeric_signal_rows = []
    for item in eda.get("bivariate_analysis", {}).get("numeric_vs_target", [])[:8]:
        if item.get("relationship_type") == "numeric_vs_numeric":
            numeric_signal_rows.append(
                [
                    item.get("feature"),
                    item.get("correlation_with_target"),
                    item.get("abs_correlation"),
                ]
            )
        else:
            numeric_signal_rows.append(
                [
                    item.get("feature"),
                    format_scalar(item.get("group_stats", [])),
                    "N/A",
                ]
            )

    lines = [
        "## Step 3 Categorical and Numerical Profile",
        f"Problem type hint: {eda.get('problem_type_hint')}",
        f"Target summary: {format_scalar(target_analysis.get('summary', {}))}",
        "",
        "### Categorical Profile",
        markdown_table(
            ["Column", "Count", "Unique", "Mode", "Top values"],
            _categorical_profile_rows(categorical_summary),
        ),
        "",
        "### Categorical Target Signal",
        markdown_table(["Feature", "Row-level target distribution"], categorical_signal_rows),
        "",
        "### Numerical Profile",
        markdown_table(
            ["Column", "Mean", "Std", "Min", "Median", "Max", "Skew"],
            _numeric_profile_rows(numeric_summary),
        ),
        "",
        "### Numerical Target Signal",
        markdown_table(["Feature", "Target relationship", "Abs correlation"], numeric_signal_rows),
        "",
        "### Correlation Highlights",
        bullet_lines(
            [
                f"top_correlated_pairs: {format_scalar(eda.get('correlation_analysis', {}).get('top_pairs', [])[:8])}",
                f"target_correlations: {format_scalar(eda.get('correlation_analysis', {}).get('target_correlations', [])[:8])}",
                f"segmentation_hints: {format_scalar(eda.get('segmentation_hints', [])[:8])}",
            ]
        ),
    ]
    return "\n".join(lines)
