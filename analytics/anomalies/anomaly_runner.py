from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from analytics.anomalies.anomaly_registry import build_anomaly_registry
from analytics.anomalies.anomaly_summary import summarize_anomaly_results
from analytics.anomalies.ecod_detector import run_ecod
from analytics.anomalies.isolation_forest_detector import run_isolation_forest
from analytics.anomalies.lof_detector import run_lof
from core.contracts import SchemaSummary
from core.exceptions import OutlierDetectionError


def _prepare_numeric_matrix(
    df: pd.DataFrame,
    *,
    schema: SchemaSummary | None = None,
    exclude_columns: set[str] | None = None,
) -> pd.DataFrame:
    excluded = exclude_columns or set()
    numeric_columns = (
        schema.numeric_columns
        if schema and schema.numeric_columns
        else [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    )
    numeric_columns = [column for column in numeric_columns if column not in excluded]

    if not numeric_columns:
        return pd.DataFrame(index=df.index)

    X = df[numeric_columns].copy()
    X = X.fillna(X.median(numeric_only=True))

    if X.isna().all(axis=None):
        return pd.DataFrame(index=df.index)

    scaler = StandardScaler()
    transformed = scaler.fit_transform(X)
    return pd.DataFrame(transformed, columns=numeric_columns, index=df.index)


def run_anomaly_detection(
    df: pd.DataFrame,
    *,
    schema: SchemaSummary | None = None,
    exclude_columns: set[str] | None = None,
    contamination: float = 0.02,
    use_ecod: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    if df is None:
        raise OutlierDetectionError("Cannot run anomaly detection on a None dataframe.")

    try:
        X = _prepare_numeric_matrix(
            df=df,
            schema=schema,
            exclude_columns=exclude_columns,
        )

        if X.empty or X.shape[1] == 0 or X.shape[0] < 20:
            registry = {
                "methods": [],
                "per_method": {},
                "combined_flagged_rows": 0,
                "combined_flagged_percentage": 0.0,
                "warnings": ["insufficient_numeric_data_for_anomaly_detection"],
            }
            return registry, summarize_anomaly_results(registry)

        results = [
            run_isolation_forest(X, contamination=contamination),
            run_lof(X, contamination=contamination),
        ]

        if use_ecod:
            try:
                results.append(run_ecod(X, contamination=contamination))
            except Exception:
                pass

        combined_mask = pd.Series(False, index=X.index)
        for item in results:
            combined_mask = combined_mask | item["mask"].fillna(False)

        registry = build_anomaly_registry(
            results=results,
            combined_mask=combined_mask,
        )
        summary = summarize_anomaly_results(registry)
        summary["used_methods"] = registry.get("methods", [])
        return registry, summary
    except Exception as exc:
        raise OutlierDetectionError("Failed during anomaly detection pipeline.") from exc