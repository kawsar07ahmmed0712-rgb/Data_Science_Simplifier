from __future__ import annotations

import pandas as pd

from analytics.outliers.column_outlier_detector import detect_outliers_for_column
from analytics.outliers.outlier_registry import (
    build_outlier_column_summary,
    build_outlier_summary,
)
from analytics.outliers.outlier_report import summarize_outliers
from analytics.outliers.outlier_scorer import score_outlier_severity
from analytics.outliers.treatment_policy import recommend_outlier_action
from core.contracts import OutlierSummary, SchemaSummary
from core.enums import SeverityLevel
from core.exceptions import OutlierDetectionError


def run_outlier_detection(
    df: pd.DataFrame,
    *,
    schema: SchemaSummary | None = None,
    skewness_analysis: dict[str, object] | None = None,
    exclude_columns: set[str] | None = None,
) -> tuple[OutlierSummary, dict[str, object]]:
    if df is None:
        raise OutlierDetectionError("Cannot run outlier detection on a None dataframe.")

    excluded = exclude_columns or set()
    numeric_columns = (
        schema.numeric_columns
        if schema and schema.numeric_columns
        else [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    )
    numeric_columns = [column for column in numeric_columns if column not in excluded]

    combined_mask = pd.Series(False, index=df.index)
    column_summaries = []
    warnings: list[str] = []

    try:
        skew_by_column = {}
        if skewness_analysis and isinstance(skewness_analysis.get("by_column"), dict):
            skew_by_column = skewness_analysis["by_column"]

        for column in numeric_columns:
            series = df[column]
            clean = series.dropna()
            if clean.empty:
                continue

            skew_info = skew_by_column.get(column, {})
            skew_severity = skew_info.get("severity")

            detection = detect_outliers_for_column(
                series=series,
                skew_severity=skew_severity,
            )

            severity = score_outlier_severity(
                outlier_percentage=float(detection["outlier_percentage"]),
            )
            action = recommend_outlier_action(
                severity=severity,
                skew_severity=skew_severity,
                unique_count=int(clean.nunique(dropna=True)),
            )

            notes = []
            if skew_severity:
                notes.append(f"skew_severity={skew_severity}")
            if detection["outlier_count"] == 0:
                notes.append("no_outliers_detected")

            column_summary = build_outlier_column_summary(
                column=column,
                method=str(detection["method"]),
                lower_bound=_safe_float(detection.get("lower_bound")),
                upper_bound=_safe_float(detection.get("upper_bound")),
                outlier_count=int(detection["outlier_count"]),
                outlier_percentage=float(detection["outlier_percentage"]),
                severity=severity,
                action=action,
                notes=notes,
            )
            column_summaries.append(column_summary)

            mask = detection.get("mask")
            if mask is not None:
                combined_mask = combined_mask | mask.fillna(False)

        if not column_summaries:
            warnings.append("no_numeric_columns_for_outlier_detection")

        outlier_summary = build_outlier_summary(
            column_summaries=column_summaries,
            combined_mask=combined_mask,
            warnings=warnings,
        )
        outlier_report = summarize_outliers(outlier_summary)

        critical_columns = [
            item.column
            for item in outlier_summary.by_column
            if item.severity == SeverityLevel.CRITICAL
        ]
        outlier_report["critical_columns"] = critical_columns

        return outlier_summary, outlier_report
    except Exception as exc:
        raise OutlierDetectionError("Failed during outlier detection pipeline.") from exc


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
        if pd.isna(value_float):
            return None
        return value_float
    except (TypeError, ValueError):
        return None