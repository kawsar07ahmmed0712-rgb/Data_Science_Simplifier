from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import bullet_lines, format_scalar, markdown_table


def build_feature_engineering_section(state: AnalysisState) -> str:
    fe = state.feature_summary or {}
    input_groups = fe.get("input_feature_groups", {})
    policies = fe.get("transform_policies", {})

    group_rows = [
        ["numeric_columns", format_scalar(input_groups.get("numeric_columns", []))],
        ["categorical_columns", format_scalar(input_groups.get("categorical_columns", []))],
        ["datetime_columns", format_scalar(input_groups.get("datetime_columns", []))],
        ["text_columns", format_scalar(input_groups.get("text_columns", []))],
        ["dropped_columns", format_scalar(input_groups.get("dropped_columns", []))],
        ["leakage_risk_columns", format_scalar(input_groups.get("leakage_risk_columns", []))],
    ]

    policy_rows = [[key, value] for key, value in policies.items()]

    lines = [
        "## Step 5 Feature Engineering Snapshot",
        f"Generated feature count: {fe.get('generated_feature_count', 'N/A')}",
        f"Train transformed shape: {format_scalar(fe.get('train_transformed_shape'))}",
        f"Test transformed shape: {format_scalar(fe.get('test_transformed_shape'))}",
        "",
        "### Input Feature Groups",
        markdown_table(["Group", "Columns"], group_rows),
        "",
        "### Transform Policies",
        markdown_table(["Policy", "Value"], policy_rows),
        "",
        "### Feature Engineering Notes",
        bullet_lines(fe.get("notes", [])[:8]),
    ]
    return "\n".join(lines)
