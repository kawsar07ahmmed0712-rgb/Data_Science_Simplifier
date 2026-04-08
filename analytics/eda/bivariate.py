from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analytics.eda.target_analysis import infer_target_problem_type


def analyze_numeric_vs_target(
    df: pd.DataFrame,
    target_column: str | None,
    numeric_columns: list[str] | None = None,
    max_features: int = 20,
) -> list[dict[str, object]]:
    if not target_column or target_column not in df.columns:
        return []

    problem_type = infer_target_problem_type(df, target_column)
    candidate_columns = numeric_columns or [
        column for column in df.columns if is_numeric_dtype(df[column])
    ]
    feature_columns = [column for column in candidate_columns if column != target_column]

    results: list[dict[str, object]] = []

    if problem_type == "regression" and is_numeric_dtype(df[target_column]):
        corr_df = df[feature_columns + [target_column]].corr()[target_column].drop(target_column)
        corr_df = corr_df.dropna()

        for feature, corr_value in corr_df.items():
            results.append(
                {
                    "feature": feature,
                    "relationship_type": "numeric_vs_numeric",
                    "correlation_with_target": round(float(corr_value), 6),
                    "abs_correlation": round(abs(float(corr_value)), 6),
                }
            )

        results.sort(key=lambda item: item["abs_correlation"], reverse=True)
        return results[:max_features]

    target_series = df[target_column].astype(str)
    for feature in feature_columns:
        temp_df = pd.DataFrame(
            {
                "feature": df[feature],
                "target": target_series,
            }
        ).dropna()

        if temp_df.empty:
            continue

        grouped = temp_df.groupby("target")["feature"].agg(["mean", "median", "count"]).reset_index()
        grouped_records = []
        for _, row in grouped.iterrows():
            grouped_records.append(
                {
                    "target_class": str(row["target"]),
                    "mean": _safe_float(row["mean"]),
                    "median": _safe_float(row["median"]),
                    "count": int(row["count"]),
                }
            )

        results.append(
            {
                "feature": feature,
                "relationship_type": "numeric_vs_class_target",
                "group_stats": grouped_records,
            }
        )

    return results[:max_features]


def analyze_categorical_vs_target(
    df: pd.DataFrame,
    target_column: str | None,
    categorical_columns: list[str] | None = None,
    max_features: int = 20,
    max_categories_per_feature: int = 15,
) -> list[dict[str, object]]:
    if not target_column or target_column not in df.columns:
        return []

    candidate_columns = categorical_columns or [
        column for column in df.columns if not is_numeric_dtype(df[column])
    ]
    feature_columns = [column for column in candidate_columns if column != target_column]

    results: list[dict[str, object]] = []

    for feature in feature_columns[:max_features]:
        temp_df = pd.DataFrame(
            {
                "feature": df[feature].astype(str),
                "target": df[target_column].astype(str),
            }
        ).dropna()

        if temp_df.empty:
            continue

        crosstab = pd.crosstab(temp_df["feature"], temp_df["target"])
        crosstab = crosstab.head(max_categories_per_feature)

        counts = {
            str(index): {str(col): int(value) for col, value in row.items()}
            for index, row in crosstab.to_dict(orient="index").items()
        }

        row_pct = pd.crosstab(temp_df["feature"], temp_df["target"], normalize="index").mul(100).round(4)
        percentages = {
            str(index): {str(col): float(value) for col, value in row.items()}
            for index, row in row_pct.to_dict(orient="index").items()
        }

        results.append(
            {
                "feature": feature,
                "relationship_type": "categorical_vs_target",
                "counts": counts,
                "row_percentages": percentages,
            }
        )

    return results


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return round(result, 6)
    except (TypeError, ValueError):
        return None