from __future__ import annotations

from typing import Any

from core.executor import model_to_dict
from core.state import AnalysisState


def build_run_manifest(
    *,
    state: AnalysisState,
    saved_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "current_stage": state.current_stage.value,
        "problem_type": state.problem_type.value,
        "file_meta": model_to_dict(state.file_meta),
        "profile": model_to_dict(state.profile),
        "plan": model_to_dict(state.plan),
        "cleaning": model_to_dict(state.cleaning),
        "split": model_to_dict(state.split),
        "model_result": model_to_dict(state.model_result),
        "issues": [model_to_dict(item) for item in state.issues],
        "artifacts": saved_paths,
    }