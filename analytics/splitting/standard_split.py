from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from analytics.splitting.split_summary import build_split_summary
from core.enums import SplitStrategy
from core.exceptions import SplitError


def run_standard_split(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None, object]:
    if df is None or df.empty:
        raise SplitError("Cannot perform standard split on empty dataframe.")

    try:
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
            summary = build_split_summary(
                strategy=SplitStrategy.STANDARD,
                target_column=target_column,
                train_rows=len(X_train),
                test_rows=len(X_test),
                stratified=False,
                notes=["standard_random_split"],
            )
            return X_train, X_test, y_train, y_test, summary

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )
        summary = build_split_summary(
            strategy=SplitStrategy.STANDARD,
            target_column=None,
            train_rows=len(train_df),
            test_rows=len(test_df),
            stratified=False,
            notes=["standard_random_split_without_target"],
        )
        return train_df, test_df, None, None, summary
    except Exception as exc:
        raise SplitError("Failed during standard split.") from exc