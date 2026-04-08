from __future__ import annotations

from pathlib import Path

from config.paths import get_paths
from core.enums import RunStage, WorkflowStatus
from core.pipeline import run_pipeline_from_source


def test_run_minimal_csv_completes_with_fallback_agents(monkeypatch, tmp_path) -> None:
    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("ollama_unavailable")

    test_paths = get_paths(root=tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    sample_path = project_root / "sample_data" / "tiny_classification.csv"

    monkeypatch.setattr("agents.base_agent.generate_text", _raise_runtime_error)
    monkeypatch.setattr("reporting.report_exporter.get_paths", lambda: test_paths)
    monkeypatch.setattr("artifacts.artifact_manager.get_paths", lambda: test_paths)

    state = run_pipeline_from_source(sample_path, target_column="target")

    assert state.flags["run_failed"] is False
    assert state.current_stage == RunStage.COMPLETED
    assert state.metadata["report_paths"]["markdown_report"]
    assert Path(state.metadata["report_paths"]["markdown_report"]).exists()
    assert Path(state.metadata["report_paths"]["html_report"]).exists()
    assert any(event.status == WorkflowStatus.COMPLETED for event in state.workflow_events)
