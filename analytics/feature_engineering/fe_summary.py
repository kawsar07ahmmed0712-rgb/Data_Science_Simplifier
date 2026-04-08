from __future__ import annotations


def build_feature_engineering_summary(
    *,
    fe_plan: dict[str, object],
    feature_names: list[str],
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
) -> dict[str, object]:
    return {
        "input_feature_groups": {
            "numeric_columns": fe_plan.get("numeric_columns", []),
            "categorical_columns": fe_plan.get("categorical_columns", []),
            "datetime_columns": fe_plan.get("datetime_columns", []),
            "text_columns": fe_plan.get("text_columns", []),
            "id_like_columns": fe_plan.get("id_like_columns", []),
            "dropped_columns": fe_plan.get("dropped_columns", []),
            "leakage_risk_columns": fe_plan.get("leakage_risk_columns", []),
        },
        "transform_policies": {
            "scale_numeric": fe_plan.get("scale_numeric", False),
            "categorical_encoding": fe_plan.get("categorical_encoding", "onehot"),
            "text_strategy": fe_plan.get("text_strategy", "drop"),
            "datetime_parts": fe_plan.get("datetime_parts", []),
        },
        "rare_label_summary": fe_plan.get("rare_label_summary", {}),
        "high_cardinality_columns": fe_plan.get("high_cardinality_columns", []),
        "generated_feature_count": len(feature_names),
        "generated_feature_names": feature_names,
        "train_transformed_shape": train_shape,
        "test_transformed_shape": test_shape,
        "notes": fe_plan.get("notes", []),
    }