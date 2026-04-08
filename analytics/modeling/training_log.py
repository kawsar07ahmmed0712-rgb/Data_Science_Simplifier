from __future__ import annotations


def build_training_summary(
    *,
    task_type: str,
    model_spec: dict[str, object],
    training_log: dict[str, object],
) -> dict[str, object]:
    return {
        "task_type": task_type,
        "model_name": model_spec.get("model_name", ""),
        "model_family": model_spec.get("model_family", ""),
        "model_params": model_spec.get("params", {}),
        "alternatives": model_spec.get("alternatives", []),
        "training_log": training_log,
    }