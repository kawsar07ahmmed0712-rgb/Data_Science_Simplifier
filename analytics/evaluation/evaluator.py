from __future__ import annotations

import pandas as pd

from analytics.evaluation.classification_metrics import compute_classification_metrics
from analytics.evaluation.clustering_metrics import compute_clustering_metrics
from analytics.evaluation.error_analysis import (
    analyze_classification_errors,
    analyze_regression_errors,
)
from analytics.evaluation.evaluation_summary import build_evaluation_summary
from analytics.evaluation.regression_metrics import compute_regression_metrics
from analytics.evaluation.threshold_analysis import analyze_binary_thresholds
from core.enums import ProblemType
from core.exceptions import EvaluationError


def evaluate_predictions(
    *,
    task_type: str | ProblemType,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    predictions: dict[str, object] | None = None,
) -> dict[str, object]:
    if predictions is None:
        raise EvaluationError("Predictions are required for evaluation.")

    normalized_task_type = str(task_type.value if isinstance(task_type, ProblemType) else task_type)

    try:
        if normalized_task_type == ProblemType.CLASSIFICATION.value:
            if y_test is None:
                raise EvaluationError("Classification evaluation requires y_test.")

            y_pred = predictions["y_pred"]
            y_proba = predictions.get("y_proba")

            metrics = compute_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
            )
            threshold_info = analyze_binary_thresholds(
                y_true=y_test,
                y_proba=y_proba,
            )
            errors = analyze_classification_errors(
                y_true=y_test,
                y_pred=y_pred,
            )
            return build_evaluation_summary(
                task_type=normalized_task_type,
                metrics=metrics,
                threshold_analysis=threshold_info,
                error_analysis=errors,
            )

        if normalized_task_type == ProblemType.REGRESSION.value:
            if y_test is None:
                raise EvaluationError("Regression evaluation requires y_test.")

            y_pred = predictions["y_pred"]
            metrics = compute_regression_metrics(
                y_true=y_test,
                y_pred=y_pred,
            )
            errors = analyze_regression_errors(
                y_true=y_test,
                y_pred=y_pred,
            )
            return build_evaluation_summary(
                task_type=normalized_task_type,
                metrics=metrics,
                error_analysis=errors,
            )

        if normalized_task_type == ProblemType.CLUSTERING.value:
            if X_test is None:
                raise EvaluationError("Clustering evaluation requires X_test.")
            cluster_labels = predictions["cluster_labels"]
            metrics = compute_clustering_metrics(
                X=X_test,
                cluster_labels=cluster_labels,
            )
            return build_evaluation_summary(
                task_type=normalized_task_type,
                metrics=metrics,
            )

        if normalized_task_type == ProblemType.ANOMALY_DETECTION.value:
            labels = predictions.get("anomaly_label")
            scores = predictions.get("anomaly_score")
            if labels is None or scores is None:
                raise EvaluationError("Anomaly evaluation requires anomaly labels and scores.")

            anomaly_count = int((labels == -1).sum())
            anomaly_rate = round(anomaly_count / max(len(labels), 1), 6)

            return build_evaluation_summary(
                task_type=normalized_task_type,
                metrics={
                    "anomaly_count": anomaly_count,
                    "anomaly_rate": anomaly_rate,
                    "score_min": round(float(scores.min()), 6),
                    "score_max": round(float(scores.max()), 6),
                },
            )

        raise EvaluationError(f"Unsupported task type for evaluation: {normalized_task_type}")
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError("Failed during evaluation pipeline.") from exc