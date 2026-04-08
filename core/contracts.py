from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.enums import (
    AgentName,
    ArtifactType,
    ColumnRole,
    OutlierAction,
    ProblemType,
    RunStage,
    SeverityLevel,
    SplitStrategy,
    WorkflowStatus,
)


class FileMeta(BaseModel):
    file_name: str = ""
    file_path: str = ""
    file_size_bytes: int = 0
    delimiter: str = ","
    encoding: str = "utf-8"
    row_count_estimate: int | None = None
    column_count_estimate: int | None = None


class ColumnSchema(BaseModel):
    name: str
    inferred_dtype: str
    role: ColumnRole = ColumnRole.UNKNOWN
    nullable: bool = True
    unique_count: int | None = None
    missing_count: int | None = None
    notes: list[str] = Field(default_factory=list)


class SchemaSummary(BaseModel):
    columns: list[ColumnSchema] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    id_like_columns: list[str] = Field(default_factory=list)
    target_candidates: list[str] = Field(default_factory=list)


class ProfileSummary(BaseModel):
    row_count: int = 0
    column_count: int = 0
    duplicate_count: int = 0
    missing_counts: dict[str, int] = Field(default_factory=dict)
    missing_percentages: dict[str, float] = Field(default_factory=dict)
    unique_counts: dict[str, int] = Field(default_factory=dict)
    dtypes: dict[str, str] = Field(default_factory=dict)
    memory_usage_bytes: int = 0
    warnings: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    step_id: str
    title: str
    stage: RunStage
    description: str
    required: bool = True
    order: int = 0


class AnalysisPlan(BaseModel):
    problem_type: ProblemType = ProblemType.UNKNOWN
    split_strategy: SplitStrategy = SplitStrategy.NONE
    target_column: str | None = None
    chart_suggestions: list[str] = Field(default_factory=list)
    feature_engineering_actions: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    planner_notes: str = ""


class CleaningLogEntry(BaseModel):
    action: str
    column: str | None = None
    affected_rows: int | None = None
    details: str = ""


class CleaningSummary(BaseModel):
    steps_applied: list[CleaningLogEntry] = Field(default_factory=list)
    columns_removed: list[str] = Field(default_factory=list)
    duplicates_removed: int = 0
    dtype_fixes: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class OutlierColumnSummary(BaseModel):
    column: str
    method: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    severity: SeverityLevel = SeverityLevel.INFO
    action: OutlierAction = OutlierAction.REPORT_ONLY
    notes: list[str] = Field(default_factory=list)


class OutlierSummary(BaseModel):
    by_column: list[OutlierColumnSummary] = Field(default_factory=list)
    total_flagged_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


class SplitSummary(BaseModel):
    strategy: SplitStrategy = SplitStrategy.NONE
    target_column: str | None = None
    train_rows: int = 0
    test_rows: int = 0
    validation_rows: int = 0
    stratified: bool = False
    notes: list[str] = Field(default_factory=list)


class ModelMetric(BaseModel):
    name: str
    value: float
    higher_is_better: bool = True


class ModelResult(BaseModel):
    model_name: str = ""
    model_family: str = ""
    problem_type: ProblemType = ProblemType.UNKNOWN
    metrics: list[ModelMetric] = Field(default_factory=list)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AgentMessage(BaseModel):
    agent: AgentName
    content: str
    structured_output: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class WorkflowEvent(BaseModel):
    stage: RunStage
    title: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime | None = None
    ended_at: datetime | None = None
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(BaseModel):
    artifact_type: ArtifactType
    name: str
    path: str
    description: str = ""


class RunIssue(BaseModel):
    stage: RunStage
    severity: SeverityLevel
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
