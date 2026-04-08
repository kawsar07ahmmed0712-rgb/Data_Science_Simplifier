from __future__ import annotations

from dataclasses import dataclass

from core.contracts import PlanStep
from core.enums import ProblemType, RunStage


@dataclass(frozen=True, slots=True)
class WorkflowStageSpec:
    stage: RunStage
    title: str
    description: str
    order: int


WORKFLOW_STAGE_SPECS: tuple[WorkflowStageSpec, ...] = (
    WorkflowStageSpec(
        stage=RunStage.INGESTION,
        title="Ingestion",
        description="Load the uploaded file and detect delimiter, encoding, and shape metadata.",
        order=10,
    ),
    WorkflowStageSpec(
        stage=RunStage.SCHEMA,
        title="Structural Audit",
        description="Infer column roles, candidate targets, datatype families, and structural risks.",
        order=20,
    ),
    WorkflowStageSpec(
        stage=RunStage.PROFILING,
        title="Profiling",
        description="Measure missingness, uniqueness, duplicates, memory usage, and high-level data quality.",
        order=30,
    ),
    WorkflowStageSpec(
        stage=RunStage.PLANNING,
        title="Planning",
        description="Choose the problem type, target strategy, workflow path, and feature engineering focus.",
        order=40,
    ),
    WorkflowStageSpec(
        stage=RunStage.CLEANING,
        title="Cleaning",
        description="Apply cleaning rules such as duplicate removal, dtype fixes, and invalid-value normalization.",
        order=50,
    ),
    WorkflowStageSpec(
        stage=RunStage.EDA,
        title="Exploratory Analysis",
        description="Build univariate, bivariate, correlation, skewness, and target-aware summaries.",
        order=60,
    ),
    WorkflowStageSpec(
        stage=RunStage.OUTLIERS,
        title="Outlier Audit",
        description="Assess numeric outliers and recommend treatment policy by column.",
        order=70,
    ),
    WorkflowStageSpec(
        stage=RunStage.ANOMALIES,
        title="Anomaly Scan",
        description="Run anomaly detectors over the numeric feature space and summarize flagged records.",
        order=80,
    ),
    WorkflowStageSpec(
        stage=RunStage.SPLITTING,
        title="Data Splitting",
        description="Split the dataset into train and test partitions when supervised modeling is possible.",
        order=90,
    ),
    WorkflowStageSpec(
        stage=RunStage.FEATURE_ENGINEERING,
        title="Feature Engineering",
        description="Build the feature plan and prepare transformations for modeling.",
        order=100,
    ),
    WorkflowStageSpec(
        stage=RunStage.MODELING,
        title="Baseline Modeling",
        description="Train a baseline model and generate predictions for the resolved task type.",
        order=110,
    ),
    WorkflowStageSpec(
        stage=RunStage.EVALUATION,
        title="Evaluation",
        description="Compute task-appropriate metrics and summarize baseline performance.",
        order=120,
    ),
    WorkflowStageSpec(
        stage=RunStage.EXPLAINABILITY,
        title="Explainability",
        description="Explain baseline model behavior and the most influential features.",
        order=130,
    ),
    WorkflowStageSpec(
        stage=RunStage.INSIGHTS,
        title="Insights",
        description="Generate an executive narrative of the most important findings.",
        order=140,
    ),
    WorkflowStageSpec(
        stage=RunStage.CRITIQUE,
        title="Critique",
        description="Review the workflow for weaknesses, modeling risks, and improvement opportunities.",
        order=150,
    ),
    WorkflowStageSpec(
        stage=RunStage.REPORTING,
        title="Reporting",
        description="Assemble the master report with structural-audit, modeling, and narrative sections.",
        order=160,
    ),
    WorkflowStageSpec(
        stage=RunStage.ARTIFACTS,
        title="Artifacts",
        description="Persist reports, datasets, charts, metadata, and pipeline artifacts for download.",
        order=170,
    ),
)


_WORKFLOW_SPEC_BY_STAGE = {spec.stage: spec for spec in WORKFLOW_STAGE_SPECS}


def get_workflow_stage_specs() -> tuple[WorkflowStageSpec, ...]:
    return WORKFLOW_STAGE_SPECS


def get_workflow_stage_spec(stage: RunStage) -> WorkflowStageSpec | None:
    return _WORKFLOW_SPEC_BY_STAGE.get(stage)


def _is_supervised(problem_type: ProblemType) -> bool:
    return problem_type in {
        ProblemType.CLASSIFICATION,
        ProblemType.REGRESSION,
        ProblemType.TIME_SERIES,
    }


def build_canonical_plan_steps(
    *,
    problem_type: ProblemType,
    target_column: str | None,
) -> list[PlanStep]:
    supervised = bool(target_column) and _is_supervised(problem_type)
    explainable = supervised

    steps: list[PlanStep] = []
    for spec in WORKFLOW_STAGE_SPECS:
        required = True
        if spec.stage in {RunStage.SPLITTING, RunStage.MODELING, RunStage.EVALUATION}:
            required = supervised
        elif spec.stage == RunStage.EXPLAINABILITY:
            required = explainable

        steps.append(
            PlanStep(
                step_id=spec.stage.value,
                title=spec.title,
                stage=spec.stage,
                description=spec.description,
                required=required,
                order=spec.order,
            )
        )

    return steps
