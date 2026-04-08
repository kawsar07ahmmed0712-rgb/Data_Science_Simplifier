from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path
    config_dir: Path
    app_dir: Path
    ui_dir: Path
    core_dir: Path
    agents_dir: Path
    integrations_dir: Path
    analytics_dir: Path
    reporting_dir: Path
    artifacts_dir: Path
    utils_dir: Path
    prompts_dir: Path
    docs_dir: Path
    tests_dir: Path
    sample_data_dir: Path
    outputs_dir: Path
    reports_dir: Path
    charts_dir: Path
    datasets_dir: Path
    models_dir: Path
    pipelines_dir: Path
    metadata_dir: Path
    runs_dir: Path


def get_paths(root: Path | None = None) -> ProjectPaths:
    project_root = (root or Path(__file__).resolve().parents[1]).resolve()

    return ProjectPaths(
        root=project_root,
        config_dir=project_root / "config",
        app_dir=project_root / "app",
        ui_dir=project_root / "ui",
        core_dir=project_root / "core",
        agents_dir=project_root / "agents",
        integrations_dir=project_root / "integrations",
        analytics_dir=project_root / "analytics",
        reporting_dir=project_root / "reporting",
        artifacts_dir=project_root / "artifacts",
        utils_dir=project_root / "utils",
        prompts_dir=project_root / "prompts",
        docs_dir=project_root / "docs",
        tests_dir=project_root / "tests",
        sample_data_dir=project_root / "sample_data",
        outputs_dir=project_root / "outputs",
        reports_dir=project_root / "outputs" / "reports",
        charts_dir=project_root / "outputs" / "charts",
        datasets_dir=project_root / "outputs" / "datasets",
        models_dir=project_root / "outputs" / "models",
        pipelines_dir=project_root / "outputs" / "pipelines",
        metadata_dir=project_root / "outputs" / "metadata",
        runs_dir=project_root / "outputs" / "runs",
    )