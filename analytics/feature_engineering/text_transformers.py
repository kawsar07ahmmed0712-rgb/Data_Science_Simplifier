from __future__ import annotations


def get_text_feature_strategy() -> dict[str, str]:
    return {
        "strategy": "drop_for_v1",
        "reason": "free_text_features_are_not_modeled_in_v1_pipeline",
    }