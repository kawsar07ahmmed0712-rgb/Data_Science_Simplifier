from __future__ import annotations

import pandas as pd

from analytics.splitting.standard_split import run_standard_split
from analytics.splitting.stratified_split import run_stratified_split
from analytics.splitting.time_split import run_time_split
from core.contracts import SchemaSummary
from core.enums import ProblemType, SplitStrategy
from core.exceptions import SplitError


def choose_split_strategy(
    *,
    problem_type: str | ProblemType | None = None,
    target_column: str | None = None,
    schema: SchemaSummary | None = None,
    preferred_strategy: str | None = None,
) -> SplitStrategy:
    if preferred_strategy:
        mapping = {
            "standard": SplitStrategy.STANDARD,
            "stratified": SplitStrategy.STRATIFIED,
            "time_based": SplitStrategy.TIME_BASED,
            "time": SplitStrategy.TIME_BASED,
        }
        if preferred_strategy in mapping:
            return mapping[preferred_strategy]

    if schema and schema.datetime_columns and target_column:
        return SplitStrategy.TIME_BASED

    if target_column and str(problem_type) in {
        ProblemType.CLASSIFICATION.value,
        str(ProblemType.CLASSIFICATION),
        "classification",
    }:
        return SplitStrategy.STRATIFIED

    return SplitStrategy.STANDARD


def split_dataset(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    problem_type: str | ProblemType | None = None,
    schema: SchemaSummary | None = None,
    preferred_strategy: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None, object]:
    if df is None or df.empty:
        raise SplitError("Cannot split an empty dataframe.")

    strategy = choose_split_strategy(
        problem_type=problem_type,
        target_column=target_column,
        schema=schema,
        preferred_strategy=preferred_strategy,
    )

    if strategy == SplitStrategy.TIME_BASED:
        datetime_columns = schema.datetime_columns if schema else []
        if not datetime_columns:
            return run_standard_split(
                df=df,
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
            )
        return run_time_split(
            df=df,
            datetime_column=datetime_columns[0],
            target_column=target_column,
            test_size=test_size,
        )

    if strategy == SplitStrategy.STRATIFIED:
        try:
            return run_stratified_split(
                df=df,
                target_column=str(target_column),
                test_size=test_size,
                random_state=random_state,
            )
        except SplitError:
            return run_standard_split(
                df=df,
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
            )

    return run_standard_split(
        df=df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )