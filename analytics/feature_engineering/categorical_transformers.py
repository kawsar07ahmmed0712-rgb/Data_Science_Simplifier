from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_categorical_pipeline(
    *,
    imputation_strategy: str = "most_frequent",
    encoding: str = "onehot",
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=imputation_strategy)),
    ]

    if encoding == "onehot":
        steps.append(("encoder", _build_one_hot_encoder()))
    else:
        raise ValueError(f"Unsupported categorical encoding strategy: {encoding}")

    return Pipeline(steps=steps)