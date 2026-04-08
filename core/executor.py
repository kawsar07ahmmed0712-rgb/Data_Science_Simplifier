from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from core.contracts import ModelMetric, ModelResult, WorkflowEvent
from core.enums import ProblemType, RunStage
from core.state import AnalysisState

ProgressCallback = Callable[[WorkflowEvent, AnalysisState], None]


def model_to_dict(obj: Any) -> Any:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {key: model_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [model_to_dict(item) for item in obj]
    return obj


def to_problem_type(value: str | ProblemType | None) -> ProblemType:
    if isinstance(value, ProblemType):
        return value
    if not value:
        return ProblemType.UNKNOWN
    normalized = str(value).strip().lower()
    mapping = {
        "classification": ProblemType.CLASSIFICATION,
        "regression": ProblemType.REGRESSION,
        "clustering": ProblemType.CLUSTERING,
        "anomaly_detection": ProblemType.ANOMALY_DETECTION,
        "anomaly": ProblemType.ANOMALY_DETECTION,
        "time_series": ProblemType.TIME_SERIES,
        "unsupervised": ProblemType.UNSUPERVISED,
    }
    return mapping.get(normalized, ProblemType.UNKNOWN)


def update_state_model_result(
    state: AnalysisState,
    *,
    evaluation: dict[str, Any],
    model_record: dict[str, Any],
    problem_type: ProblemType,
) -> None:
    metrics_payload = evaluation.get("metrics", {})
    metrics: list[ModelMetric] = []

    for key, value in metrics_payload.items():
        if isinstance(value, (dict, list)) or value is None:
            continue
        if isinstance(value, bool):
            continue
        try:
            metrics.append(
                ModelMetric(
                    name=str(key),
                    value=float(value),
                    higher_is_better=key not in {"mse", "rmse", "mae", "davies_bouldin_score"},
                )
            )
        except Exception:
            continue

    state.model_result = ModelResult(
        model_name=str(model_record.get("model_name", "")),
        model_family=str(model_record.get("model_family", "")),
        problem_type=problem_type,
        metrics=metrics,
        feature_importance=dict(model_record.get("feature_importance", {})),
        notes=[],
    )
    state.touch()


def _notify_progress(
    state: AnalysisState,
    stage: RunStage,
    progress_callback: ProgressCallback | None,
) -> None:
    if progress_callback is None:
        return

    event = state.get_workflow_event(stage)
    if event is not None:
        progress_callback(event, state)


def push_stage(
    state: AnalysisState,
    stage: RunStage,
    *,
    progress_callback: ProgressCallback | None = None,
    summary: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    state.start_workflow_stage(stage, summary=summary, details=details)
    _notify_progress(state, stage, progress_callback)


def complete_stage(
    state: AnalysisState,
    stage: RunStage,
    *,
    progress_callback: ProgressCallback | None = None,
    summary: str | None = None,
    details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> None:
    state.complete_workflow_stage(
        stage,
        summary=summary,
        details=details,
        warnings=warnings,
    )
    _notify_progress(state, stage, progress_callback)


def skip_stage(
    state: AnalysisState,
    stage: RunStage,
    *,
    progress_callback: ProgressCallback | None = None,
    summary: str,
    details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> None:
    state.skip_workflow_stage(
        stage,
        summary=summary,
        details=details,
        warnings=warnings,
    )
    _notify_progress(state, stage, progress_callback)


def fail_stage(
    state: AnalysisState,
    stage: RunStage,
    *,
    progress_callback: ProgressCallback | None = None,
    summary: str,
    details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> None:
    state.fail_workflow_stage(
        stage,
        summary=summary,
        details=details,
        warnings=warnings,
    )
    _notify_progress(state, stage, progress_callback)
