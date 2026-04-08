from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

from core.contracts import (
    AgentMessage,
    AnalysisPlan,
    ArtifactRecord,
    CleaningSummary,
    FileMeta,
    ModelResult,
    OutlierSummary,
    ProfileSummary,
    RunIssue,
    SchemaSummary,
    SplitSummary,
    WorkflowEvent,
)
from core.enums import ProblemType, RunStage, WorkflowStatus
from core.workflow import get_workflow_stage_spec, get_workflow_stage_specs


@dataclass(slots=True)
class AnalysisState:
    run_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    current_stage: RunStage = RunStage.INITIALIZED
    problem_type: ProblemType = ProblemType.UNKNOWN

    file_meta: FileMeta = field(default_factory=FileMeta)
    schema: SchemaSummary = field(default_factory=SchemaSummary)
    profile: ProfileSummary = field(default_factory=ProfileSummary)
    plan: AnalysisPlan = field(default_factory=AnalysisPlan)
    cleaning: CleaningSummary = field(default_factory=CleaningSummary)
    outliers: OutlierSummary = field(default_factory=OutlierSummary)
    split: SplitSummary = field(default_factory=SplitSummary)
    model_result: ModelResult = field(default_factory=ModelResult)

    eda_results: dict[str, Any] = field(default_factory=dict)
    feature_summary: dict[str, Any] = field(default_factory=dict)
    explainability_summary: dict[str, Any] = field(default_factory=dict)

    agent_messages: list[AgentMessage] = field(default_factory=list)
    workflow_events: list[WorkflowEvent] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    issues: list[RunIssue] = field(default_factory=list)

    raw_dataframe_shape: tuple[int, int] | None = None
    cleaned_dataframe_shape: tuple[int, int] | None = None
    transformed_train_shape: tuple[int, int] | None = None
    transformed_test_shape: tuple[int, int] | None = None

    insight_text: str = ""
    critique_text: str = ""
    recommendation_text: str = ""

    flags: dict[str, bool] = field(
        default_factory=lambda: {
            "has_target": False,
            "needs_replan": False,
            "can_train_model": False,
            "should_run_outlier_detection": True,
            "should_run_explainability": False,
            "run_failed": False,
        }
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ensure_workflow_events()

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def set_stage(self, stage: RunStage) -> None:
        self.current_stage = stage
        self.touch()

    def ensure_workflow_events(self) -> None:
        if self.workflow_events:
            return

        self.workflow_events = [
            WorkflowEvent(
                stage=spec.stage,
                title=spec.title,
                status=WorkflowStatus.PENDING,
                summary=spec.description,
                details={"description": spec.description, "order": spec.order},
            )
            for spec in get_workflow_stage_specs()
        ]

    def get_workflow_event(self, stage: RunStage) -> WorkflowEvent | None:
        self.ensure_workflow_events()
        for event in self.workflow_events:
            if event.stage == stage:
                return event
        return None

    def _merge_event_details(
        self,
        current_details: dict[str, Any],
        new_details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not new_details:
            return current_details

        merged = dict(current_details)
        merged.update(new_details)
        return merged

    def start_workflow_stage(
        self,
        stage: RunStage,
        *,
        summary: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        self.ensure_workflow_events()
        now = datetime.now(UTC)

        for event in self.workflow_events:
            if event.stage != stage and event.status == WorkflowStatus.RUNNING:
                event.status = WorkflowStatus.COMPLETED
                event.ended_at = now

        event = self.get_workflow_event(stage)
        if event is None:
            spec = get_workflow_stage_spec(stage)
            event = WorkflowEvent(
                stage=stage,
                title=spec.title if spec else stage.value.replace("_", " ").title(),
                status=WorkflowStatus.PENDING,
                summary=spec.description if spec else "",
                details=details or {},
            )
            self.workflow_events.append(event)

        event.status = WorkflowStatus.RUNNING
        if event.started_at is None:
            event.started_at = now
        event.ended_at = None
        if summary:
            event.summary = summary
        event.details = self._merge_event_details(event.details, details)

        self.current_stage = stage
        self.touch()
        return event

    def complete_workflow_stage(
        self,
        stage: RunStage,
        *,
        summary: str | None = None,
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> WorkflowEvent | None:
        event = self.get_workflow_event(stage)
        if event is None:
            return None

        now = datetime.now(UTC)
        if event.started_at is None:
            event.started_at = now
        event.ended_at = now
        event.status = WorkflowStatus.COMPLETED
        if summary:
            event.summary = summary
        if warnings is not None:
            event.warnings = warnings
        event.details = self._merge_event_details(event.details, details)
        self.touch()
        return event

    def skip_workflow_stage(
        self,
        stage: RunStage,
        *,
        summary: str,
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> WorkflowEvent | None:
        event = self.get_workflow_event(stage)
        if event is None:
            return None

        now = datetime.now(UTC)
        event.status = WorkflowStatus.SKIPPED
        if event.started_at is None:
            event.started_at = now
        event.ended_at = now
        event.summary = summary
        if warnings is not None:
            event.warnings = warnings
        event.details = self._merge_event_details(event.details, details)
        self.touch()
        return event

    def fail_workflow_stage(
        self,
        stage: RunStage,
        *,
        summary: str,
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> WorkflowEvent | None:
        event = self.get_workflow_event(stage)
        if event is None:
            return None

        now = datetime.now(UTC)
        if event.started_at is None:
            event.started_at = now
        event.ended_at = now
        event.status = WorkflowStatus.FAILED
        event.summary = summary
        if warnings is not None:
            event.warnings = warnings
        event.details = self._merge_event_details(event.details, details)
        self.touch()
        return event

    def add_agent_message(self, message: AgentMessage) -> None:
        self.agent_messages.append(message)
        self.touch()

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        self.artifacts.append(artifact)
        self.touch()

    def add_issue(self, issue: RunIssue) -> None:
        self.issues.append(issue)
        self.touch()

    def mark_failed(self, message: str | None = None) -> None:
        self.flags["run_failed"] = True
        if self.current_stage not in {RunStage.INITIALIZED, RunStage.COMPLETED, RunStage.FAILED}:
            event = self.get_workflow_event(self.current_stage)
            if event is None or event.status != WorkflowStatus.FAILED:
                self.fail_workflow_stage(
                    self.current_stage,
                    summary=message or f"{self.current_stage.value} failed.",
                )
        self.current_stage = RunStage.FAILED
        self.touch()

    def mark_completed(self, summary: str | None = None) -> None:
        if self.current_stage not in {RunStage.INITIALIZED, RunStage.COMPLETED, RunStage.FAILED}:
            event = self.get_workflow_event(self.current_stage)
            if event is None or event.status != WorkflowStatus.COMPLETED:
                self.complete_workflow_stage(
                    self.current_stage,
                    summary=summary or f"{self.current_stage.value} completed successfully.",
                )
        self.current_stage = RunStage.COMPLETED
        self.touch()

    def update_problem_type(self, problem_type: ProblemType) -> None:
        self.problem_type = problem_type
        self.touch()
