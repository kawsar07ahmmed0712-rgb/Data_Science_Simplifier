from __future__ import annotations

import pandas as pd

from core.contracts import OutlierColumnSummary, OutlierSummary
from core.enums import OutlierAction, SeverityLevel


def build_outlier_column_summary(
    *,
    column: str,
    method: str,
    lower_bound: float | None,
    upper_bound: float | None,
    outlier_count: int,
    outlier_percentage: float,
    severity: SeverityLevel,
    action: OutlierAction,
    notes: list[str] | None = None,
) -> OutlierColumnSummary:
    return OutlierColumnSummary(
        column=column,
        method=method,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        outlier_count=outlier_count,
        outlier_percentage=outlier_percentage,
        severity=severity,
        action=action,
        notes=notes or [],
    )


def build_outlier_summary(
    column_summaries: list[OutlierColumnSummary],
    combined_mask: pd.Series | None = None,
    warnings: list[str] | None = None,
) -> OutlierSummary:
    total_flagged_rows = int(combined_mask.sum()) if combined_mask is not None else 0
    return OutlierSummary(
        by_column=column_summaries,
        total_flagged_rows=total_flagged_rows,
        warnings=warnings or [],
    )