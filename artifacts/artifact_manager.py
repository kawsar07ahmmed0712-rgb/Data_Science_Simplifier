from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from artifacts.chart_exporter import save_basic_charts
from artifacts.dataset_exporter import save_dataset
from artifacts.metadata_exporter import save_json_metadata
from artifacts.model_exporter import save_model_object
from artifacts.path_registry import RunArtifactPaths
from artifacts.pipeline_exporter import save_preprocessor
from artifacts.run_manifest import build_run_manifest
from config import get_paths
from core.executor import model_to_dict
from core.state import AnalysisState


def _prepare_run_dirs(run_id: str) -> RunArtifactPaths:
    paths = get_paths()
    run_dir = Path(paths.runs_dir) / run_id
    reports_dir = run_dir / "reports"
    charts_dir = run_dir / "charts"
    datasets_dir = run_dir / "datasets"
    models_dir = run_dir / "models"
    pipelines_dir = run_dir / "pipelines"
    metadata_dir = run_dir / "metadata"

    for directory in [run_dir, reports_dir, charts_dir, datasets_dir, models_dir, pipelines_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return RunArtifactPaths(
        run_dir=run_dir,
        reports_dir=reports_dir,
        charts_dir=charts_dir,
        datasets_dir=datasets_dir,
        models_dir=models_dir,
        pipelines_dir=pipelines_dir,
        metadata_dir=metadata_dir,
    )


def persist_pipeline_outputs(
    *,
    state: AnalysisState,
    cleaned_df: pd.DataFrame,
    X_train: pd.DataFrame | None,
    X_test: pd.DataFrame | None,
    y_train: pd.Series | None,
    y_test: pd.Series | None,
    preprocessor: Any,
    report_paths: dict[str, str],
) -> dict[str, str]:
    run_paths = _prepare_run_dirs(state.run_id)
    saved: dict[str, str] = {}

    cleaned_path = save_dataset(cleaned_df, run_paths.datasets_dir / "cleaned_data.csv")
    saved["cleaned_dataset"] = str(cleaned_path)

    if X_train is not None:
        train_path = save_dataset(X_train, run_paths.datasets_dir / "X_train.csv")
        saved["x_train_dataset"] = str(train_path)
    if X_test is not None:
        test_path = save_dataset(X_test, run_paths.datasets_dir / "X_test.csv")
        saved["x_test_dataset"] = str(test_path)
    if y_train is not None:
        y_train_path = save_dataset(y_train.to_frame(name="target"), run_paths.datasets_dir / "y_train.csv")
        saved["y_train_dataset"] = str(y_train_path)
    if y_test is not None:
        y_test_path = save_dataset(y_test.to_frame(name="target"), run_paths.datasets_dir / "y_test.csv")
        saved["y_test_dataset"] = str(y_test_path)

    for key, path in report_paths.items():
        saved[key] = str(path)

    model_obj = state.metadata.get("model_object")
    if model_obj is not None:
        model_path = save_model_object(model_obj, run_paths.models_dir / "model.joblib")
        saved["model_artifact"] = str(model_path)

    if preprocessor is not None:
        pipe_path = save_preprocessor(preprocessor, run_paths.pipelines_dir / "preprocessor.joblib")
        saved["preprocessor_artifact"] = str(pipe_path)

    chart_paths = save_basic_charts(
        state=state,
        cleaned_df=cleaned_df,
        charts_dir=run_paths.charts_dir,
    )
    saved.update({f"chart_{k}": str(v) for k, v in chart_paths.items()})

    manifest = build_run_manifest(state=state, saved_paths=saved)
    manifest_path = save_json_metadata(manifest, run_paths.metadata_dir / "run_manifest.json")
    saved["run_manifest"] = str(manifest_path)

    state.metadata["run_dir"] = str(run_paths.run_dir)
    return saved