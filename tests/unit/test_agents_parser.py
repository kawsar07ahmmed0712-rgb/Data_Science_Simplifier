from __future__ import annotations

from agents.planner_agent import run_planner_agent
from core.contracts import ColumnSchema, ProfileSummary, SchemaSummary
from core.enums import ColumnRole, ProblemType, RunStage


def test_planner_fallback_generates_canonical_plan(monkeypatch) -> None:
    def _raise_runtime_error(self, *, user_prompt: str) -> str:
        raise RuntimeError("ollama_unavailable")

    monkeypatch.setattr("agents.planner_agent.BaseAgent.run", _raise_runtime_error)

    schema = SchemaSummary(
        columns=[
            ColumnSchema(
                name="target",
                inferred_dtype="int64",
                role=ColumnRole.TARGET,
                nullable=False,
                unique_count=2,
                missing_count=0,
                notes=["selected_as_target"],
            ),
            ColumnSchema(
                name="income",
                inferred_dtype="float64",
                role=ColumnRole.NUMERIC,
                nullable=True,
                unique_count=48,
                missing_count=1,
                notes=["numeric_like_column"],
            ),
            ColumnSchema(
                name="segment",
                inferred_dtype="object",
                role=ColumnRole.CATEGORICAL,
                nullable=True,
                unique_count=4,
                missing_count=0,
                notes=["categorical_like_column"],
            ),
        ],
        numeric_columns=["income"],
        categorical_columns=["segment"],
        target_candidates=["target"],
    )
    profile = ProfileSummary(
        row_count=50,
        column_count=3,
        duplicate_count=0,
        missing_counts={"income": 1},
        missing_percentages={"income": 2.0},
        unique_counts={"target": 2, "income": 48, "segment": 4},
        dtypes={"target": "int64", "income": "float64", "segment": "object"},
        memory_usage_bytes=4096,
        warnings=["minor_missingness_detected"],
    )

    plan, message = run_planner_agent(
        schema=schema,
        profile=profile,
        target_column="target",
    )

    assert plan.problem_type == ProblemType.CLASSIFICATION
    assert plan.target_column == "target"
    assert plan.steps
    assert plan.steps[0].stage == RunStage.INGESTION
    assert [step.stage for step in plan.steps][-1] == RunStage.ARTIFACTS
    assert any("planner_agent_fallback_due_to" in warning for warning in message.warnings)
