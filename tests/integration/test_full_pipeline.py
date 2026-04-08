from __future__ import annotations

import pandas as pd

from core.enums import RunStage, WorkflowStatus
from core.exceptions import SchemaDetectionError
from core.pipeline import run_pipeline_from_dataframe


def test_pipeline_marks_failed_workflow_event(monkeypatch) -> None:
    def _explode_schema(*_args, **_kwargs):
        raise SchemaDetectionError("schema_audit_failed")

    monkeypatch.setattr("core.pipeline.build_schema_summary", _explode_schema)

    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    state = run_pipeline_from_dataframe(df=df, target_column="target")
    schema_event = state.get_workflow_event(RunStage.SCHEMA)

    assert state.flags["run_failed"] is True
    assert schema_event is not None
    assert schema_event.status == WorkflowStatus.FAILED
    assert state.issues
    assert state.issues[0].message == "schema_audit_failed"
