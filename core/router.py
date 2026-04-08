from __future__ import annotations

from core.state import AnalysisState
from core.enums import RunStage, ProblemType


def should_run_modeling(state: AnalysisState) -> bool:
    if state.plan.target_column and state.problem_type in {
        ProblemType.CLASSIFICATION,
        ProblemType.REGRESSION,
        ProblemType.TIME_SERIES,
    }:
        return True
    return False


def should_run_outliers(state: AnalysisState) -> bool:
    return bool(state.flags.get("should_run_outlier_detection", True))


def should_run_explainability(state: AnalysisState) -> bool:
    return bool(state.model_result.model_name)


def get_next_terminal_stage(state: AnalysisState) -> RunStage:
    if state.flags.get("run_failed"):
        return RunStage.FAILED
    return RunStage.COMPLETED