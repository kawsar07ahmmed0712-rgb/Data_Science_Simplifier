from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def infer_target_problem_type(df: pd.DataFrame, target_column: str) -> str:
    if target_column not in df.columns:
        return "unknown"

    series = df[target_column].dropna()
    if series.empty:
        return "unknown"

    if is_bool_dtype(series):
        return "classification"

    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / max(len(series), 1)

    if is_numeric_dtype(series):
        if unique_count <= 10 and unique_ratio <= 0.20:
            return "classification"
        return "regression"

    if unique_count <= 30:
        return "classification"

    return "classification"


def analyze_target(
    df: pd.DataFrame,
    target_column: str | None,
) -> dict[str, object]:
    if not target_column:
        return {
            "has_target": False,
            "target_column": None,
            "problem_type_hint": "unspecified",
            "summary": {},
        }

    if target_column not in df.columns:
        return {
            "has_target": False,
            "target_column": target_column,
            "problem_type_hint": "missing_target_column",
            "summary": {},
        }

    series = df[target_column]
    non_null = series.dropna()

    if non_null.empty:
        return {
            "has_target": True,
            "target_column": target_column,
            "problem_type_hint": "unknown",
            "summary": {
                "non_null_count": 0,
                "missing_count": int(series.isna().sum()),
            },
        }

    problem_type_hint = infer_target_problem_type(df, target_column)
    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = round(unique_count / max(len(non_null), 1), 6)

    summary: dict[str, object] = {
        "non_null_count": int(non_null.shape[0]),
        "missing_count": int(series.isna().sum()),
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
    }

    if problem_type_hint == "classification":
        distribution = non_null.astype(str).value_counts(dropna=False).to_dict()
        distribution_pct = (
            non_null.astype(str).value_counts(normalize=True, dropna=False).mul(100).round(4).to_dict()
        )

        imbalance_ratio = None
        if distribution:
            counts = list(distribution.values())
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = round(max_count / max(min_count, 1), 4)

        summary.update(
            {
                "class_distribution": {str(k): int(v) for k, v in distribution.items()},
                "class_distribution_pct": {str(k): float(v) for k, v in distribution_pct.items()},
                "imbalance_ratio": imbalance_ratio,
            }
        )
    elif problem_type_hint == "regression":
        summary.update(
            {
                "mean": _safe_float(non_null.mean()),
                "median": _safe_float(non_null.median()),
                "std": _safe_float(non_null.std()),
                "min": _safe_float(non_null.min()),
                "max": _safe_float(non_null.max()),
            }
        )

    return {
        "has_target": True,
        "target_column": target_column,
        "problem_type_hint": problem_type_hint,
        "summary": summary,
    }


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return result
    except (TypeError, ValueError):
        return None