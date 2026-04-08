from __future__ import annotations

import pandas as pd

from analytics.feature_engineering.feature_selector import select_feature_groups
from analytics.feature_engineering.leakage_guard import (
    detect_leakage_risk_columns,
    get_leakage_risk_columns,
)
from analytics.feature_engineering.rare_label_handler import (
    detect_high_cardinality_columns,
    summarize_rare_labels,
)
from core.contracts import SchemaSummary


def plan_feature_engineering(
    df: pd.DataFrame,
    *,
    schema: SchemaSummary | None = None,
    target_column: str | None = None,
    problem_type: str | None = None,
    outlier_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    leakage_risk_map = detect_leakage_risk_columns(
        df=df,
        target_column=target_column,
        schema=schema,
    )
    leakage_risk_columns = get_leakage_risk_columns(
        df=df,
        target_column=target_column,
        schema=schema,
    )

    drop_columns = set(leakage_risk_columns)
    feature_groups = select_feature_groups(
        df=df,
        schema=schema,
        target_column=target_column,
        drop_columns=drop_columns,
    )

    rare_label_summary = summarize_rare_labels(
        df=df,
        categorical_columns=feature_groups["categorical_columns"],
    )
    high_cardinality_columns = detect_high_cardinality_columns(
        df=df,
        categorical_columns=feature_groups["categorical_columns"],
    )

    actionable_outlier_columns = []
    if outlier_summary and isinstance(outlier_summary.get("top_columns"), list):
        actionable_outlier_columns = [
            item["column"]
            for item in outlier_summary["top_columns"]
            if item.get("action") not in {"ignore", "report_only"}
        ]

    scale_numeric = problem_type in {
        "classification",
        "regression",
        "clustering",
        "anomaly_detection",
    }

    notes: list[str] = []
    if feature_groups["text_columns"]:
        notes.append("text_columns_will_be_dropped_in_v1")
    if feature_groups["id_like_columns"]:
        notes.append("id_like_columns_excluded_from_modeling")
    if leakage_risk_columns:
        notes.append("potential_leakage_columns_excluded")
    if high_cardinality_columns:
        notes.append("high_cardinality_columns_present")
    if actionable_outlier_columns:
        notes.append("some_numeric_columns_may_need_outlier_treatment_later")

    return {
        "target_column": target_column,
        "problem_type": problem_type or "unknown",
        "numeric_columns": feature_groups["numeric_columns"],
        "categorical_columns": feature_groups["categorical_columns"],
        "datetime_columns": feature_groups["datetime_columns"],
        "text_columns": feature_groups["text_columns"],
        "id_like_columns": feature_groups["id_like_columns"],
        "leftover_columns": feature_groups["leftover_columns"],
        "dropped_columns": sorted(set(feature_groups["dropped_columns"]).union(feature_groups["id_like_columns"])),
        "leakage_risk_columns": leakage_risk_columns,
        "leakage_risk_map": leakage_risk_map,
        "rare_label_summary": rare_label_summary,
        "high_cardinality_columns": high_cardinality_columns,
        "actionable_outlier_columns": actionable_outlier_columns,
        "numeric_imputation_strategy": "median",
        "categorical_imputation_strategy": "most_frequent",
        "categorical_encoding": "onehot",
        "scale_numeric": scale_numeric,
        "text_strategy": "drop",
        "datetime_parts": ["year", "month", "day", "dayofweek"],
        "notes": notes,
    }