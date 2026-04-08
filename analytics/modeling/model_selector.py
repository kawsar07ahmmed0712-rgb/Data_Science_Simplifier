from __future__ import annotations

from core.enums import ModelFamily, ProblemType


def select_baseline_model_spec(
    *,
    task_type: ProblemType,
    random_state: int = 42,
) -> dict[str, object]:
    if task_type == ProblemType.CLASSIFICATION:
        return {
            "task_type": task_type.value,
            "model_name": "RandomForestClassifier",
            "model_family": ModelFamily.ENSEMBLE.value,
            "params": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": random_state,
                "n_jobs": -1,
                "class_weight": "balanced",
            },
            "alternatives": ["LogisticRegression"],
        }

    if task_type == ProblemType.REGRESSION:
        return {
            "task_type": task_type.value,
            "model_name": "RandomForestRegressor",
            "model_family": ModelFamily.ENSEMBLE.value,
            "params": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": random_state,
                "n_jobs": -1,
            },
            "alternatives": ["LinearRegression"],
        }

    if task_type == ProblemType.CLUSTERING:
        return {
            "task_type": task_type.value,
            "model_name": "KMeans",
            "model_family": ModelFamily.DISTANCE.value,
            "params": {
                "n_clusters": 3,
                "random_state": random_state,
                "n_init": 10,
            },
            "alternatives": [],
        }

    if task_type == ProblemType.ANOMALY_DETECTION:
        return {
            "task_type": task_type.value,
            "model_name": "IsolationForest",
            "model_family": ModelFamily.ANOMALY.value,
            "params": {
                "n_estimators": 300,
                "contamination": 0.02,
                "random_state": random_state,
            },
            "alternatives": [],
        }

    return {
        "task_type": ProblemType.UNKNOWN.value,
        "model_name": "",
        "model_family": ModelFamily.BASELINE.value,
        "params": {},
        "alternatives": [],
    }