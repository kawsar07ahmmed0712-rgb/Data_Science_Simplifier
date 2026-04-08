from __future__ import annotations

import pandas as pd

from analytics.splitting.split_summary import build_split_summary
from core.enums import SplitStrategy
from core.exceptions import SplitError


def run_time_split(
    df: pd.DataFrame,
    *,
    datetime_column: str,
    target_column: str | None = None,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None, object]:
    if df is None or df.empty:
        raise SplitError("Cannot perform time split on empty dataframe.")

    if datetime_column not in df.columns:
        raise SplitError(f"Datetime column '{datetime_column}' not found for time split.")

    temp_df = df.copy()
    temp_df[datetime_column] = pd.to_datetime(temp_df[datetime_column], errors="coerce")
    temp_df = temp_df.sort_values(by=datetime_column).reset_index(drop=True)

    split_index = int(len(temp_df) * (1 - test_size))
    split_index = max(1, min(split_index, len(temp_df) - 1))

    train_df = temp_df.iloc[:split_index].copy()
    test_df = temp_df.iloc[split_index:].copy()

    if target_column and target_column in temp_df.columns:
        X_train = train_df.drop(columns=[target_column])
        X_test = test_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        y_test = test_df[target_column]
        summary = build_split_summary(
            strategy=SplitStrategy.TIME_BASED,
            target_column=target_column,
            train_rows=len(X_train),
            test_rows=len(X_test),
            stratified=False,
            notes=[f"time_based_split_using:{datetime_column}"],
        )
        return X_train, X_test, y_train, y_test, summary

    summary = build_split_summary(
        strategy=SplitStrategy.TIME_BASED,
        target_column=None,
        train_rows=len(train_df),
        test_rows=len(test_df),
        stratified=False,
        notes=[f"time_based_split_using:{datetime_column}", "no_target_provided"],
    )
    return train_df, test_df, None, None, summary