from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from analytics.splitting.split_summary import build_split_summary
from core.enums import SplitStrategy
from core.exceptions import SplitError


def run_stratified_split(
    df: pd.DataFrame,
    *,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    if df is None or df.empty:
        raise SplitError("Cannot perform stratified split on empty dataframe.")

    if target_column not in df.columns:
        raise SplitError(f"Target column '{target_column}' not found for stratified split.")

    y = df[target_column]
    class_counts = y.value_counts(dropna=False)

    if (class_counts < 2).any():
        raise SplitError(
            "Stratified split requires at least 2 samples in every target class."
        )

    try:
        X = df.drop(columns=[target_column])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        summary = build_split_summary(
            strategy=SplitStrategy.STRATIFIED,
            target_column=target_column,
            train_rows=len(X_train),
            test_rows=len(X_test),
            stratified=True,
            notes=["stratified_split_by_target"],
        )
        return X_train, X_test, y_train, y_test, summary
    except Exception as exc:
        raise SplitError("Failed during stratified split.") from exc