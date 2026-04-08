from __future__ import annotations

import pandas as pd

from analytics.modeling.anomaly_model_runner import run_anomaly_model
from analytics.modeling.classifier_trainer import train_classifier
from analytics.modeling.clustering_runner import run_clustering
from analytics.modeling.model_registry import build_model_record
from analytics.modeling.model_selector import select_baseline_model_spec
from analytics.modeling.regressor_trainer import train_regressor
from analytics.modeling.task_detector import detect_task_type
from analytics.modeling.training_log import build_training_summary
from core.enums import ProblemType
from core.exceptions import ModelingError


def run_baseline_model(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    problem_type_hint: str | None = None,
    random_state: int = 42,
) -> tuple[object, dict[str, object], dict[str, object], dict[str, object]]:
    task_type = detect_task_type(
        problem_type_hint=problem_type_hint,
        y_train=y_train,
    )
    model_spec = select_baseline_model_spec(
        task_type=task_type,
        random_state=random_state,
    )

    if task_type == ProblemType.CLASSIFICATION:
        if X_test is None or y_train is None or y_test is None:
            raise ModelingError("Classification baseline requires train/test features and targets.")

        model, predictions, training_log = train_classifier(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            model_spec=model_spec,
        )

    elif task_type == ProblemType.REGRESSION:
        if X_test is None or y_train is None or y_test is None:
            raise ModelingError("Regression baseline requires train/test features and targets.")

        model, predictions, training_log = train_regressor(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            model_spec=model_spec,
        )

    elif task_type == ProblemType.CLUSTERING:
        model, predictions, training_log = run_clustering(
            X=X_train,
            n_clusters=int(model_spec["params"].get("n_clusters", 3)),
            random_state=random_state,
        )

    elif task_type == ProblemType.ANOMALY_DETECTION:
        model, predictions, training_log = run_anomaly_model(
            X=X_train,
            contamination=float(model_spec["params"].get("contamination", 0.02)),
            random_state=random_state,
        )

    else:
        raise ModelingError(f"Unsupported or unresolved task type: {task_type.value}")

    model_record = build_model_record(
        model_spec=model_spec,
        feature_names=list(X_train.columns),
        model=model,
    )
    training_summary = build_training_summary(
        task_type=task_type.value,
        model_spec=model_spec,
        training_log=training_log,
    )

    return model, predictions, model_record, training_summary