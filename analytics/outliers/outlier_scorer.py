from __future__ import annotations

from core.enums import SeverityLevel


def score_outlier_severity(
    outlier_percentage: float,
) -> SeverityLevel:
    if outlier_percentage >= 15.0:
        return SeverityLevel.CRITICAL
    if outlier_percentage >= 8.0:
        return SeverityLevel.HIGH
    if outlier_percentage >= 3.0:
        return SeverityLevel.MEDIUM
    if outlier_percentage > 0.0:
        return SeverityLevel.LOW
    return SeverityLevel.INFO