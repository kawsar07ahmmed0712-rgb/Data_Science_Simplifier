from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_numeric_pipeline(
    *,
    imputation_strategy: str = "median",
    scale: bool = True,
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=imputation_strategy)),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)