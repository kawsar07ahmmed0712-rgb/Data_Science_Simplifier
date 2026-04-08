from __future__ import annotations

import pandas as pd

from core.exceptions import OutlierDetectionError


def run_ecod(
    X: pd.DataFrame,
    *,
    contamination: float = 0.02,
) -> dict[str, object]:
    try:
        from pyod.models.ecod import ECOD
    except Exception as exc:
        raise OutlierDetectionError(
            "pyod is required for ECOD anomaly detection but is not available."
        ) from exc

    model = ECOD(contamination=contamination)
    model.fit(X)

    predictions = model.labels_
    scores = model.decision_scores_

    mask = predictions == 1
    flagged_count = int(mask.sum())
    flagged_percentage = round((flagged_count / max(len(X), 1)) * 100.0, 4)

    return {
        "method": "ecod",
        "mask": pd.Series(mask, index=X.index),
        "scores": pd.Series(scores, index=X.index),
        "flagged_count": flagged_count,
        "flagged_percentage": flagged_percentage,
    }