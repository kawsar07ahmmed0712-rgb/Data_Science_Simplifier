from __future__ import annotations

from core.contracts import CleaningLogEntry, CleaningSummary


def make_log_entry(
    action: str,
    *,
    column: str | None = None,
    affected_rows: int | None = None,
    details: str = "",
) -> CleaningLogEntry:
    return CleaningLogEntry(
        action=action,
        column=column,
        affected_rows=affected_rows,
        details=details,
    )


def append_log(summary: CleaningSummary, entry: CleaningLogEntry) -> None:
    summary.steps_applied.append(entry)


def append_warning(summary: CleaningSummary, warning: str) -> None:
    summary.warnings.append(warning)