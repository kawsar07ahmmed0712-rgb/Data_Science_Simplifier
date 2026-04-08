from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from analytics.schema.id_detector import detect_id_like_columns


@dataclass(frozen=True, slots=True)
class TargetCandidate:
    column: str
    score: float
    reasons: list[str]


def _classification_like_score(series: pd.Series) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    non_null = series.dropna()
    if non_null.empty:
        return score, reasons

    nunique = non_null.nunique()
    unique_ratio = nunique / max(len(non_null), 1)

    if is_bool_dtype(non_null):
        score += 5.0
        reasons.append("boolean_target_like")

    if nunique <= 10:
        score += 3.0
        reasons.append("low_cardinality")

    if 10 < nunique <= 30:
        score += 1.5
        reasons.append("moderate_cardinality")

    if unique_ratio < 0.20:
        score += 1.5
        reasons.append("low_unique_ratio")

    return score, reasons


def _regression_like_score(series: pd.Series) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    non_null = series.dropna()
    if non_null.empty or not is_numeric_dtype(non_null):
        return score, reasons

    nunique = non_null.nunique()
    unique_ratio = nunique / max(len(non_null), 1)

    if nunique >= 20:
        score += 2.0
        reasons.append("many_unique_values")

    if unique_ratio >= 0.10:
        score += 2.0
        reasons.append("continuous_like")

    return score, reasons


def infer_target_candidates(
    df: pd.DataFrame,
    user_target: str | None = None,
) -> list[TargetCandidate]:
    if user_target:
        if user_target not in df.columns:
            raise ValueError(f"Provided target '{user_target}' is not in dataframe columns.")
        return [
            TargetCandidate(
                column=user_target,
                score=999.0,
                reasons=["user_selected_target"],
            )
        ]

    id_like_columns = set(detect_id_like_columns(df))
    candidates: list[TargetCandidate] = []

    for column in df.columns:
        if column in id_like_columns:
            continue

        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        nunique = non_null.nunique()
        if nunique <= 1:
            continue

        classification_score, classification_reasons = _classification_like_score(series)
        regression_score, regression_reasons = _regression_like_score(series)

        score = max(classification_score, regression_score)
        reasons = classification_reasons if classification_score >= regression_score else regression_reasons

        if score > 0:
            candidates.append(
                TargetCandidate(
                    column=column,
                    score=score,
                    reasons=reasons,
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:5]