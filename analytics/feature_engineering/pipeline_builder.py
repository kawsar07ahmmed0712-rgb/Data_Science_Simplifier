from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer

from analytics.feature_engineering.categorical_transformers import build_categorical_pipeline
from analytics.feature_engineering.datetime_transformers import build_datetime_pipeline
from analytics.feature_engineering.feature_registry import (
    get_transformed_feature_names,
    to_feature_dataframe,
)
from analytics.feature_engineering.fe_summary import build_feature_engineering_summary
from analytics.feature_engineering.numeric_transformers import build_numeric_pipeline
from core.exceptions import FeatureEngineeringError


def build_feature_preprocessor(
    fe_plan: dict[str, Any],
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []

    numeric_columns = fe_plan.get("numeric_columns", [])
    categorical_columns = fe_plan.get("categorical_columns", [])
    datetime_columns = fe_plan.get("datetime_columns", [])

    if numeric_columns:
        transformers.append(
            (
                "num",
                build_numeric_pipeline(
                    imputation_strategy=str(fe_plan.get("numeric_imputation_strategy", "median")),
                    scale=bool(fe_plan.get("scale_numeric", True)),
                ),
                list(numeric_columns),
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "cat",
                build_categorical_pipeline(
                    imputation_strategy=str(fe_plan.get("categorical_imputation_strategy", "most_frequent")),
                    encoding=str(fe_plan.get("categorical_encoding", "onehot")),
                ),
                list(categorical_columns),
            )
        )

    if datetime_columns:
        transformers.append(
            (
                "dt",
                build_datetime_pipeline(
                    parts=tuple(fe_plan.get("datetime_parts", ["year", "month", "day", "dayofweek"])),
                ),
                list(datetime_columns),
            )
        )

    if not transformers:
        raise FeatureEngineeringError("No usable feature groups available for preprocessing.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def run_feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    fe_plan: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], ColumnTransformer]:
    if train_df is None or test_df is None:
        raise FeatureEngineeringError("Train/test dataframes are required.")

    try:
        preprocessor = build_feature_preprocessor(fe_plan)
        transformed_train = preprocessor.fit_transform(train_df)
        transformed_test = preprocessor.transform(test_df)

        feature_names = get_transformed_feature_names(
            preprocessor=preprocessor,
            fallback_feature_count=transformed_train.shape[1],
        )

        X_train = to_feature_dataframe(
            transformed_train,
            feature_names=feature_names,
            index=train_df.index,
        )
        X_test = to_feature_dataframe(
            transformed_test,
            feature_names=feature_names,
            index=test_df.index,
        )

        summary = build_feature_engineering_summary(
            fe_plan=fe_plan,
            feature_names=feature_names,
            train_shape=X_train.shape,
            test_shape=X_test.shape,
        )

        return X_train, X_test, summary, preprocessor
    except Exception as exc:
        raise FeatureEngineeringError("Failed during feature engineering pipeline.") from exc