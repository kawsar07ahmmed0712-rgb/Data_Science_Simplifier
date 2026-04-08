from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunArtifactPaths:
    run_dir: Path
    reports_dir: Path
    charts_dir: Path
    datasets_dir: Path
    models_dir: Path
    pipelines_dir: Path
    metadata_dir: Path