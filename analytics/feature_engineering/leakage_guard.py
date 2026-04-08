from __future__ import annotations

import re

import pandas as pd

from core.contracts import SchemaSummary

_LEAKAGE_NAME_PATTERNS = [
    r"pred",
    r"prediction",
    r"target",
    r"label",
    r"outcome",
    r"result",
    r"status",
    r"approved",
    r"rejected",
    r"decision",
]

_EXACT_TARGET_SUFFIX_PATTERNS = [
    r"_encoded$",
    r"_label$",
    r"_target$",
    r"_pred$",
]


def detect_leakage_risk_columns(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    schema: SchemaSummary | None = None,
) -> dict[str, list[str]]:
    risk_reasons: dict[str, list[str]] = {}

    if not target_column or target_column not in df.columns:
        return risk_reasons

    target_lower = target_column.lower()

    for column in df.columns:
        if column == target_column:
            continue

        reasons: list[str] = []
        column_lower = column.lower()

        if target_lower in column_lower:
            reasons.append("column_name_contains_target_name")

        for pattern in _LEAKAGE_NAME_PATTERNS:
            if re.search(pattern, column_lower):
                reasons.append(f"name_matches_pattern:{pattern}")

        if any(re.search(pattern, column_lower) for pattern in _EXACT_TARGET_SUFFIX_PATTERNS):
            reasons.append("derived_target_like_name")

        try:
            comparable = pd.DataFrame(
                {
                    "feature": df[column],
                    "target": df[target_column],
                }
            ).dropna()

            if not comparable.empty:
                identical_ratio = (comparable["feature"].astype(str) == comparable["target"].astype(str)).mean()
                if identical_ratio >= 0.98:
                    reasons.append("feature_almost_identical_to_target")
        except Exception:
            pass

        if schema and column in schema.id_like_columns:
            if target_lower in column_lower:
                reasons.append("id_like_column_related_to_target")

        if reasons:
            risk_reasons[column] = sorted(set(reasons))

    return risk_reasons


def get_leakage_risk_columns(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    schema: SchemaSummary | None = None,
) -> list[str]:
    risk_map = detect_leakage_risk_columns(
        df=df,
        target_column=target_column,
        schema=schema,
    )
    return sorted(risk_map.keys())