from __future__ import annotations

import pandas as pd


def extract_feature_importance(
    model,
    *,
    feature_names: list[str] | None = None,
    top_n: int = 25,
) -> dict[str, float]:
    if feature_names is None:
        feature_names = []

    importance_map: dict[str, float] = {}

    if hasattr(model, "feature_importances_"):
        importances = list(model.feature_importances_)
        names = feature_names or [f"feature_{idx}" for idx in range(len(importances))]
        importance_map = {
            str(name): round(float(value), 6)
            for name, value in zip(names, importances)
        }
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if hasattr(coef, "ndim") and coef.ndim > 1:
            coef = coef[0]
        importances = [abs(float(value)) for value in coef]
        names = feature_names or [f"feature_{idx}" for idx in range(len(importances))]
        importance_map = {
            str(name): round(float(value), 6)
            for name, value in zip(names, importances)
        }

    sorted_items = sorted(
        importance_map.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    return {name: score for name, score in sorted_items}


def build_model_record(
    *,
    model_spec: dict[str, object],
    feature_names: list[str] | None = None,
    model=None,
) -> dict[str, object]:
    return {
        "model_name": model_spec.get("model_name", ""),
        "model_family": model_spec.get("model_family", ""),
        "params": model_spec.get("params", {}),
        "feature_importance": extract_feature_importance(
            model,
            feature_names=feature_names,
        ) if model is not None else {},
    }