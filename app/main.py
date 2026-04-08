from __future__ import annotations

from pathlib import Path

from app.startup import ensure_project_directories
from core.pipeline import run_pipeline_from_source
from core.state import AnalysisState


def run_file_pipeline(
    source: str | Path,
    *,
    target_column: str | None = None,
) -> AnalysisState:
    ensure_project_directories()
    return run_pipeline_from_source(source, target_column=target_column)