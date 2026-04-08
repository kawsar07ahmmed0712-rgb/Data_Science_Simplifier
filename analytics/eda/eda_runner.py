from __future__ import annotations

import pandas as pd

from analytics.eda.bivariate import (
    analyze_categorical_vs_target,
    analyze_numeric_vs_target,
)
from analytics.eda.correlation_analysis import (
    compute_correlation_matrix,
    get_target_correlations,
    get_top_correlated_pairs,
)
from analytics.eda.multicollinearity import summarize_multicollinearity
from analytics.eda.segmentation_hints import generate_segmentation_hints
from analytics.eda.skewness_analysis import analyze_skewness
from analytics.eda.target_analysis import analyze_target, infer_target_problem_type
from analytics.eda.univariate import (
    get_categorical_univariate_summary,
    get_numeric_univariate_summary,
)
from core.contracts import SchemaSummary
from core.exceptions import EDAError


def run_eda(
    df: pd.DataFrame,
    schema: SchemaSummary | None = None,
    target_column: str | None = None,
) -> dict[str, object]:
    if df is None:
        raise EDAError("Cannot run EDA on a None dataframe.")

    try:
        numeric_columns = (
            schema.numeric_columns
            if schema and schema.numeric_columns
            else [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
        )
        categorical_columns = (
            schema.categorical_columns
            if schema and schema.categorical_columns
            else [
                column
                for column in df.columns
                if not pd.api.types.is_numeric_dtype(df[column])
            ]
        )

        target_info = analyze_target(df, target_column=target_column)

        resolved_problem_type = "unspecified"
        if target_column and target_column in df.columns:
            resolved_problem_type = infer_target_problem_type(df, target_column)

        numeric_univariate = get_numeric_univariate_summary(
            df=df,
            numeric_columns=numeric_columns,
        )
        categorical_univariate = get_categorical_univariate_summary(
            df=df,
            categorical_columns=categorical_columns,
        )
        correlation_matrix = compute_correlation_matrix(
            df=df,
            numeric_columns=numeric_columns,
        )
        top_correlated_pairs = get_top_correlated_pairs(
            df=df,
            numeric_columns=numeric_columns,
        )
        target_correlations = get_target_correlations(
            df=df,
            target_column=target_column,
            numeric_columns=numeric_columns,
        )
        skewness = analyze_skewness(
            df=df,
            numeric_columns=numeric_columns,
        )
        multicollinearity = summarize_multicollinearity(
            df=df,
            numeric_columns=numeric_columns,
        )
        numeric_vs_target = analyze_numeric_vs_target(
            df=df,
            target_column=target_column,
            numeric_columns=numeric_columns,
        )
        categorical_vs_target = analyze_categorical_vs_target(
            df=df,
            target_column=target_column,
            categorical_columns=categorical_columns,
        )
        segmentation_hints = generate_segmentation_hints(
            df=df,
            categorical_columns=categorical_columns,
        )

        return {
            "problem_type_hint": resolved_problem_type,
            "target_analysis": target_info,
            "univariate": {
                "numeric": numeric_univariate,
                "categorical": categorical_univariate,
            },
            "correlation_analysis": {
                "matrix": correlation_matrix,
                "top_pairs": top_correlated_pairs,
                "target_correlations": target_correlations,
            },
            "skewness_analysis": skewness,
            "multicollinearity": multicollinearity,
            "bivariate_analysis": {
                "numeric_vs_target": numeric_vs_target,
                "categorical_vs_target": categorical_vs_target,
            },
            "segmentation_hints": segmentation_hints,
        }
    except Exception as exc:
        raise EDAError("Failed during EDA pipeline execution.") from exc