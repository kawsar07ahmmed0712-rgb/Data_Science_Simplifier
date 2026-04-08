from __future__ import annotations

from core.contracts import SplitSummary
from core.enums import SplitStrategy


def build_split_summary(
    *,
    strategy: SplitStrategy,
    target_column: str | None,
    train_rows: int,
    test_rows: int,
    validation_rows: int = 0,
    stratified: bool = False,
    notes: list[str] | None = None,
) -> SplitSummary:
    return SplitSummary(
        strategy=strategy,
        target_column=target_column,
        train_rows=train_rows,
        test_rows=test_rows,
        validation_rows=validation_rows,
        stratified=stratified,
        notes=notes or [],
    )