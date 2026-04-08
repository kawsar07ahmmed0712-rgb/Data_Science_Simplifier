from __future__ import annotations

import pandas as pd

from core.contracts import AnalysisPlan, SchemaSummary
from core.enums import ProblemType
from core.exceptions import ValidationError


def validate_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValidationError("Dataframe is None.")
    if df.empty and len(df.columns) == 0:
        raise ValidationError("Dataframe has no usable rows or columns.")


def validate_target_column(
    df: pd.DataFrame,
    target_column: str | None,
) -> None:
    if target_column is None:
        return
    if target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' not found in dataframe.")


def validate_plan_against_schema(
    plan: AnalysisPlan,
    schema: SchemaSummary,
) -> None:
    if plan.target_column and plan.target_column not in [col.name for col in schema.columns]:
        raise ValidationError("Planner target column is missing from schema.")

    if (
        plan.problem_type in {ProblemType.CLASSIFICATION, ProblemType.REGRESSION, ProblemType.TIME_SERIES}
        and not plan.target_column
    ):
        raise ValidationError("Supervised plan requires a target column.")