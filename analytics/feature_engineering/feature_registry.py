from __future__ import annotations

import pandas as pd


def get_transformed_feature_names(preprocessor, fallback_feature_count: int) -> list[str]:
    try:
        names = list(preprocessor.get_feature_names_out())
        return [_clean_feature_name(name) for name in names]
    except Exception:
        return [f"feature_{idx}" for idx in range(fallback_feature_count)]


def to_feature_dataframe(
    transformed,
    *,
    feature_names: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    df = pd.DataFrame(transformed, columns=feature_names, index=index)
    return df


def _clean_feature_name(name: str) -> str:
    cleaned = name.replace("num__", "")
    cleaned = cleaned.replace("cat__", "")
    cleaned = cleaned.replace("dt__", "")
    return cleaned