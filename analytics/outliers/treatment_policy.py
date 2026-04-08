from __future__ import annotations

from core.enums import OutlierAction, SeverityLevel


def recommend_outlier_action(
    *,
    severity: SeverityLevel,
    skew_severity: str | None = None,
    unique_count: int | None = None,
) -> OutlierAction:
    if severity == SeverityLevel.INFO:
        return OutlierAction.IGNORE

    if unique_count is not None and unique_count <= 5:
        return OutlierAction.REPORT_ONLY

    if skew_severity == "high":
        if severity in {SeverityLevel.HIGH, SeverityLevel.CRITICAL}:
            return OutlierAction.WINSORIZE
        return OutlierAction.FLAG

    if severity in {SeverityLevel.HIGH, SeverityLevel.CRITICAL}:
        return OutlierAction.CAP

    if severity == SeverityLevel.MEDIUM:
        return OutlierAction.FLAG

    return OutlierAction.REPORT_ONLY