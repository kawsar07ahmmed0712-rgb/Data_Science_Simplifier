from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from core.enums import ProblemType


def detect_task_type(
    *,
    problem_type_hint: str | None = None,
    y_train: pd.Series | None = None,
) -> ProblemType:
    if problem_type_hint:
        normalized = str(problem_type_hint).strip().lower()
        mapping = {
            "classification": ProblemType.CLASSIFICATION,
            "regression": ProblemType.REGRESSION,
            "clustering": ProblemType.CLUSTERING,
            "anomaly_detection": ProblemType.ANOMALY_DETECTION,
            "anomaly": ProblemType.ANOMALY_DETECTION,
            "time_series": ProblemType.TIME_SERIES,
            "unsupervised": ProblemType.UNSUPERVISED,
        }
        if normalized in mapping:
            return mapping[normalized]

    if y_train is None:
        return ProblemType.UNSUPERVISED

    clean = y_train.dropna()
    if clean.empty:
        return ProblemType.UNKNOWN

    if is_bool_dtype(clean):
        return ProblemType.CLASSIFICATION

    unique_count = int(clean.nunique(dropna=True))
    unique_ratio = unique_count / max(len(clean), 1)

    if is_numeric_dtype(clean):
        if unique_count <= 10 and unique_ratio <= 0.20:
            return ProblemType.CLASSIFICATION
        return ProblemType.REGRESSION

    return ProblemType.CLASSIFICATION