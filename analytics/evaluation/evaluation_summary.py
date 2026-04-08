from __future__ import annotations


def build_evaluation_summary(
    *,
    task_type: str,
    metrics: dict[str, object],
    threshold_analysis: dict[str, object] | None = None,
    error_analysis: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "task_type": task_type,
        "metrics": metrics,
        "threshold_analysis": threshold_analysis or {},
        "error_analysis": error_analysis or {},
    }