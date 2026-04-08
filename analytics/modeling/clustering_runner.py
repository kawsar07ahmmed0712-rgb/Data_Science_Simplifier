from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from core.exceptions import ModelingError


def run_clustering(
    X: pd.DataFrame,
    *,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[object, dict[str, object], dict[str, object]]:
    if X is None or X.empty:
        raise ModelingError("Clustering requires a non-empty feature matrix.")

    try:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        labels = model.fit_predict(X)

        predictions = {
            "cluster_labels": pd.Series(labels, index=X.index, name="cluster"),
        }
        training_log = {
            "model_name": "KMeans",
            "model_family": "distance",
            "rows": int(X.shape[0]),
            "feature_count": int(X.shape[1]),
            "n_clusters": int(n_clusters),
        }

        return model, predictions, training_log
    except Exception as exc:
        raise ModelingError("Failed during clustering run.") from exc