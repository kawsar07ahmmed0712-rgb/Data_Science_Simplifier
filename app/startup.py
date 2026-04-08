from __future__ import annotations

from pathlib import Path

from config import get_paths


def ensure_project_directories() -> None:
    paths = get_paths()
    required_dirs = [
        paths.outputs_dir,
        paths.reports_dir,
        paths.charts_dir,
        paths.datasets_dir,
        paths.models_dir,
        paths.pipelines_dir,
        paths.metadata_dir,
        paths.runs_dir,
    ]
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)