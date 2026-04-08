from __future__ import annotations

import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


def compute_clustering_metrics(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict[str, float | None]:
    unique_clusters = pd.Series(cluster_labels).nunique(dropna=True)

    if unique_clusters < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "cluster_count": int(unique_clusters),
        }

    return {
        "silhouette_score": round(float(silhouette_score(X, cluster_labels)), 6),
        "davies_bouldin_score": round(float(davies_bouldin_score(X, cluster_labels)), 6),
        "cluster_count": int(unique_clusters),
    }