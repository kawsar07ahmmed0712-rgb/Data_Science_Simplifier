from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agents.critic_agent import run_critic_agent
from agents.explanation_agent import run_explanation_agent
from agents.insight_agent import run_insight_agent
from agents.planner_agent import run_planner_agent
from agents.recommendation_agent import run_recommendation_agent
from analytics.anomalies.anomaly_runner import run_anomaly_detection
from analytics.cleaning.cleaner import clean_dataframe
from analytics.eda.eda_runner import run_eda
from analytics.evaluation.evaluator import evaluate_predictions
from analytics.feature_engineering.fe_planner import plan_feature_engineering
from analytics.feature_engineering.pipeline_builder import run_feature_engineering
from analytics.ingestion.loader import load_csv
from analytics.modeling.baseline_trainer import run_baseline_model
from analytics.outliers.outlier_router import run_outlier_detection
from analytics.profiling.profiler import profile_dataframe
from analytics.schema.schema_detector import build_schema_summary
from analytics.splitting.split_router import split_dataset
from artifacts.artifact_manager import persist_pipeline_outputs
from config import get_settings
from core.contracts import ArtifactRecord, RunIssue
from core.enums import ArtifactType, RunStage, SeverityLevel
from core.exceptions import AgenticCSVError, ValidationError
from core.executor import (
    ProgressCallback,
    complete_stage,
    fail_stage,
    model_to_dict,
    push_stage,
    skip_stage,
    to_problem_type,
    update_state_model_result,
)
from core.router import should_run_explainability, should_run_modeling, should_run_outliers
from core.run_context import RunContext
from core.state import AnalysisState
from core.validation_manager import (
    validate_dataframe,
    validate_plan_against_schema,
    validate_target_column,
)
from reporting.report_exporter import export_reports


def _resolve_source_name(source: str | Path | Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    return str(getattr(source, "name", "uploaded_source.csv"))


def _safe_detail(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value
    return str(value)


def _append_run_issue(
    state: AnalysisState,
    *,
    stage: RunStage,
    severity: SeverityLevel,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    state.add_issue(
        RunIssue(
            stage=stage,
            severity=severity,
            message=message,
            details=details or {},
        )
    )


def run_pipeline_from_source(
    source: str | Path | Any,
    *,
    target_column: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisState:
    source_name = _resolve_source_name(source)
    context = RunContext(source_name=source_name, user_target_column=target_column)
    state = AnalysisState(run_id=context.run_id)

    push_stage(
        state,
        RunStage.INGESTION,
        progress_callback=progress_callback,
        summary="Loading the source file and detecting metadata.",
        details={"source_name": source_name},
    )

    try:
        df, file_meta = load_csv(source)
        state.file_meta = file_meta
        complete_stage(
            state,
            RunStage.INGESTION,
            progress_callback=progress_callback,
            summary=(
                f"Loaded {file_meta.file_name or source_name} with {df.shape[0]} rows and {df.shape[1]} columns."
            ),
            details={
                "file_name": file_meta.file_name,
                "delimiter": file_meta.delimiter,
                "encoding": file_meta.encoding,
                "shape": list(df.shape),
            },
        )
    except (AgenticCSVError, ValidationError) as exc:
        fail_stage(
            state,
            RunStage.INGESTION,
            progress_callback=progress_callback,
            summary=f"Failed to load source file: {exc}",
        )
        state.mark_failed(str(exc))
        _append_run_issue(
            state,
            stage=RunStage.INGESTION,
            severity=SeverityLevel.CRITICAL,
            message=str(exc),
        )
        return state
    except Exception as exc:
        fail_stage(
            state,
            RunStage.INGESTION,
            progress_callback=progress_callback,
            summary=f"Unhandled ingestion failure: {exc}",
        )
        state.mark_failed(str(exc))
        _append_run_issue(
            state,
            stage=RunStage.INGESTION,
            severity=SeverityLevel.CRITICAL,
            message=f"Unhandled pipeline failure: {exc}",
        )
        return state

    return run_pipeline_from_dataframe(
        df=df,
        state=state,
        target_column=target_column,
        progress_callback=progress_callback,
    )


def run_pipeline_from_dataframe(
    *,
    df: pd.DataFrame,
    state: AnalysisState | None = None,
    target_column: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisState:
    settings = get_settings()
    state = state or AnalysisState(run_id=RunContext().run_id)

    try:
        validate_dataframe(df)
        state.raw_dataframe_shape = df.shape

        push_stage(
            state,
            RunStage.SCHEMA,
            progress_callback=progress_callback,
            summary="Profiling structural roles, datatypes, and candidate targets.",
        )
        raw_schema = build_schema_summary(df, target_column=target_column)
        validate_target_column(df, target_column)
        complete_stage(
            state,
            RunStage.SCHEMA,
            progress_callback=progress_callback,
            summary=(
                f"Detected {len(raw_schema.columns)} columns: {len(raw_schema.numeric_columns)} numeric, "
                f"{len(raw_schema.categorical_columns)} categorical, {len(raw_schema.datetime_columns)} datetime."
            ),
            details={
                "target_candidates": raw_schema.target_candidates,
                "id_like_columns": raw_schema.id_like_columns,
                "text_columns": raw_schema.text_columns,
            },
        )

        push_stage(
            state,
            RunStage.PROFILING,
            progress_callback=progress_callback,
            summary="Computing missingness, uniqueness, duplicates, and memory profile.",
        )
        raw_profile, raw_profile_details = profile_dataframe(df, schema=raw_schema)
        state.profile = raw_profile
        state.schema = raw_schema
        state.metadata["raw_profile_details"] = raw_profile_details
        complete_stage(
            state,
            RunStage.PROFILING,
            progress_callback=progress_callback,
            summary=(
                f"Profiled {raw_profile.row_count} rows and {raw_profile.column_count} columns with "
                f"{raw_profile.duplicate_count} duplicate rows detected."
            ),
            details={
                "warning_count": len(raw_profile.warnings),
                "constant_columns": raw_profile_details.get("constant_columns", []),
            },
        )

        resolved_target = target_column or (raw_schema.target_candidates[0] if raw_schema.target_candidates else None)

        push_stage(
            state,
            RunStage.PLANNING,
            progress_callback=progress_callback,
            summary="Choosing the analysis path, target strategy, and baseline workflow.",
        )
        plan, planner_msg = run_planner_agent(
            schema=raw_schema,
            profile=raw_profile,
            target_column=resolved_target,
        )
        state.plan = plan
        state.add_agent_message(planner_msg)
        validate_plan_against_schema(plan, raw_schema)
        state.update_problem_type(plan.problem_type)
        state.flags["has_target"] = bool(plan.target_column)
        complete_stage(
            state,
            RunStage.PLANNING,
            progress_callback=progress_callback,
            summary=(
                f"Resolved problem type '{plan.problem_type.value}'"
                + (f" with target '{plan.target_column}'." if plan.target_column else " without a supervised target.")
            ),
            details={
                "split_strategy": plan.split_strategy.value,
                "risk_flags": plan.risk_flags,
                "planner_warnings": planner_msg.warnings,
            },
            warnings=planner_msg.warnings,
        )

        push_stage(
            state,
            RunStage.CLEANING,
            progress_callback=progress_callback,
            summary="Applying baseline cleaning operations to the dataset.",
        )
        cleaned_df, cleaning_summary = clean_dataframe(
            df,
            schema=raw_schema,
            target_column=plan.target_column,
        )
        state.cleaning = cleaning_summary
        state.cleaned_dataframe_shape = cleaned_df.shape

        cleaned_schema = build_schema_summary(cleaned_df, target_column=plan.target_column)
        state.schema = cleaned_schema
        complete_stage(
            state,
            RunStage.CLEANING,
            progress_callback=progress_callback,
            summary=(
                f"Cleaned dataset shape is {cleaned_df.shape[0]} rows by {cleaned_df.shape[1]} columns. "
                f"Removed {cleaning_summary.duplicates_removed} duplicates."
            ),
            details={
                "columns_removed": cleaning_summary.columns_removed,
                "dtype_fixes": cleaning_summary.dtype_fixes,
                "warning_count": len(cleaning_summary.warnings),
            },
            warnings=cleaning_summary.warnings,
        )

        push_stage(
            state,
            RunStage.EDA,
            progress_callback=progress_callback,
            summary="Building univariate, bivariate, correlation, and target-aware summaries.",
        )
        eda_results = run_eda(
            cleaned_df,
            schema=cleaned_schema,
            target_column=plan.target_column,
        )
        state.eda_results = eda_results
        state.problem_type = to_problem_type(eda_results.get("problem_type_hint"))
        complete_stage(
            state,
            RunStage.EDA,
            progress_callback=progress_callback,
            summary=(
                f"EDA completed with problem type hint '{eda_results.get('problem_type_hint')}' and "
                f"{len(eda_results.get('segmentation_hints', []))} segmentation hints."
            ),
            details={
                "top_correlated_pairs": eda_results.get("correlation_analysis", {}).get("top_pairs", [])[:5],
                "highly_skewed_columns": eda_results.get("skewness_analysis", {}).get("highly_skewed_columns", [])[:10],
            },
        )

        if should_run_outliers(state):
            push_stage(
                state,
                RunStage.OUTLIERS,
                progress_callback=progress_callback,
                summary="Scanning numeric features for outliers and treatment recommendations.",
            )
            outlier_summary, outlier_report = run_outlier_detection(
                cleaned_df,
                schema=cleaned_schema,
                skewness_analysis=eda_results.get("skewness_analysis"),
                exclude_columns={plan.target_column} if plan.target_column else set(),
            )
            state.outliers = outlier_summary
            state.metadata["outlier_report"] = outlier_report
            complete_stage(
                state,
                RunStage.OUTLIERS,
                progress_callback=progress_callback,
                summary=(
                    f"Flagged {outlier_report.get('total_flagged_rows', 0)} rows across "
                    f"{outlier_report.get('column_count_with_outliers', 0)} columns."
                ),
                details={
                    "critical_columns": outlier_report.get("critical_columns", []),
                    "top_columns": outlier_report.get("top_columns", [])[:5],
                },
                warnings=outlier_report.get("warnings", []),
            )
        else:
            outlier_report = {}
            skip_stage(
                state,
                RunStage.OUTLIERS,
                progress_callback=progress_callback,
                summary="Outlier detection was disabled by runtime flags.",
            )

        push_stage(
            state,
            RunStage.ANOMALIES,
            progress_callback=progress_callback,
            summary="Running anomaly detectors on the numeric feature space.",
        )
        anomaly_registry, anomaly_summary = run_anomaly_detection(
            cleaned_df,
            schema=cleaned_schema,
            exclude_columns={plan.target_column} if plan.target_column else set(),
        )
        state.metadata["anomaly_registry"] = anomaly_registry
        state.metadata["anomaly_summary"] = anomaly_summary
        complete_stage(
            state,
            RunStage.ANOMALIES,
            progress_callback=progress_callback,
            summary=(
                f"Anomaly scan flagged {anomaly_summary.get('combined_flagged_rows', 0)} rows using "
                f"{len(anomaly_summary.get('used_methods', []))} detector(s)."
            ),
            details={
                "used_methods": anomaly_summary.get("used_methods", []),
            },
            warnings=anomaly_summary.get("warnings", []),
        )

        evaluation: dict[str, Any] = {}
        model_record: dict[str, Any] = {}
        training_summary: dict[str, Any] = {}
        preprocessor = None
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        fe_plan: dict[str, Any] = {}

        if should_run_modeling(state):
            push_stage(
                state,
                RunStage.SPLITTING,
                progress_callback=progress_callback,
                summary="Splitting the cleaned dataset into train and test partitions.",
            )
            X_train_raw, X_test_raw, y_train, y_test, split_summary = split_dataset(
                cleaned_df,
                target_column=plan.target_column,
                problem_type=eda_results.get("problem_type_hint"),
                schema=cleaned_schema,
                test_size=settings.default_test_size,
                random_state=settings.random_seed,
            )
            state.split = split_summary
            complete_stage(
                state,
                RunStage.SPLITTING,
                progress_callback=progress_callback,
                summary=(
                    f"Created a {split_summary.strategy.value} split with "
                    f"{split_summary.train_rows} train rows and {split_summary.test_rows} test rows."
                ),
                details={"notes": split_summary.notes},
                warnings=split_summary.notes,
            )

            push_stage(
                state,
                RunStage.FEATURE_ENGINEERING,
                progress_callback=progress_callback,
                summary="Planning and transforming features for baseline modeling.",
            )
            fe_plan = plan_feature_engineering(
                cleaned_df,
                schema=cleaned_schema,
                target_column=plan.target_column,
                problem_type=eda_results.get("problem_type_hint"),
                outlier_summary=outlier_report,
            )
            state.metadata["feature_plan"] = fe_plan

            X_train, X_test, fe_summary, preprocessor = run_feature_engineering(
                X_train_raw,
                X_test_raw,
                fe_plan=fe_plan,
            )
            state.feature_summary = fe_summary
            state.transformed_train_shape = X_train.shape
            state.transformed_test_shape = X_test.shape
            complete_stage(
                state,
                RunStage.FEATURE_ENGINEERING,
                progress_callback=progress_callback,
                summary=(
                    f"Prepared transformed features with train shape {X_train.shape[0]}x{X_train.shape[1]} "
                    f"and test shape {X_test.shape[0]}x{X_test.shape[1]}."
                ),
                details={
                    "feature_plan_keys": list(fe_plan.keys())[:10],
                    "feature_summary_keys": list(fe_summary.keys())[:10],
                },
            )

            push_stage(
                state,
                RunStage.MODELING,
                progress_callback=progress_callback,
                summary="Training the baseline model and generating predictions.",
            )
            model, predictions, model_record, training_summary = run_baseline_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                problem_type_hint=eda_results.get("problem_type_hint"),
                random_state=settings.random_seed,
            )
            state.metadata["model_object"] = model
            state.metadata["training_summary"] = training_summary
            complete_stage(
                state,
                RunStage.MODELING,
                progress_callback=progress_callback,
                summary=(
                    f"Trained baseline model '{model_record.get('model_name', 'unknown_model')}' "
                    f"from the {model_record.get('model_family', 'unknown')} family."
                ),
                details={
                    "training_summary": training_summary,
                    "model_family": model_record.get("model_family"),
                },
            )

            push_stage(
                state,
                RunStage.EVALUATION,
                progress_callback=progress_callback,
                summary="Computing evaluation metrics for the baseline model.",
            )
            evaluation = evaluate_predictions(
                task_type=eda_results.get("problem_type_hint"),
                X_test=X_test,
                y_test=y_test,
                predictions=predictions,
            )
            state.metadata["evaluation"] = evaluation
            update_state_model_result(
                state,
                evaluation=evaluation,
                model_record=model_record,
                problem_type=to_problem_type(eda_results.get("problem_type_hint")),
            )
            state.flags["can_train_model"] = True
            metric_snapshot = [
                f"{key}={value}"
                for key, value in list(evaluation.get("metrics", {}).items())[:5]
                if not isinstance(value, (dict, list))
            ]
            complete_stage(
                state,
                RunStage.EVALUATION,
                progress_callback=progress_callback,
                summary=(
                    "Evaluation completed."
                    + (f" Metric snapshot: {', '.join(metric_snapshot)}." if metric_snapshot else "")
                ),
                details={"metrics": evaluation.get("metrics", {})},
            )
        else:
            skip_stage(
                state,
                RunStage.SPLITTING,
                progress_callback=progress_callback,
                summary="Data splitting skipped because no supervised modeling path was selected.",
            )

            push_stage(
                state,
                RunStage.FEATURE_ENGINEERING,
                progress_callback=progress_callback,
                summary="Building a feature plan without a supervised modeling branch.",
            )
            fe_plan = plan_feature_engineering(
                cleaned_df,
                schema=cleaned_schema,
                target_column=plan.target_column,
                problem_type=eda_results.get("problem_type_hint"),
                outlier_summary=outlier_report,
            )
            state.metadata["feature_plan"] = fe_plan
            state.feature_summary = {
                "notes": ["modeling_skipped_due_to_missing_supervised_target_or_unsupervised_path"],
                "feature_plan": fe_plan,
            }
            complete_stage(
                state,
                RunStage.FEATURE_ENGINEERING,
                progress_callback=progress_callback,
                summary="Feature planning completed, but model training was skipped because no supervised target was confirmed.",
                details={"feature_plan_keys": list(fe_plan.keys())[:10]},
            )

            skip_stage(
                state,
                RunStage.MODELING,
                progress_callback=progress_callback,
                summary="Modeling skipped because the planner did not resolve a supervised task.",
            )
            skip_stage(
                state,
                RunStage.EVALUATION,
                progress_callback=progress_callback,
                summary="Evaluation skipped because no model was trained.",
            )

        push_stage(
            state,
            RunStage.INSIGHTS,
            progress_callback=progress_callback,
            summary="Drafting the executive insight narrative.",
        )
        insight_text, insight_msg = run_insight_agent(
            profile=model_to_dict(state.profile),
            target_analysis=eda_results.get("target_analysis", {}),
            eda_summary=eda_results,
            outlier_report=outlier_report,
            evaluation=evaluation,
        )
        state.insight_text = insight_text
        state.add_agent_message(insight_msg)
        complete_stage(
            state,
            RunStage.INSIGHTS,
            progress_callback=progress_callback,
            summary="Generated the executive insight summary.",
            warnings=insight_msg.warnings,
            details={"agent_warnings": insight_msg.warnings},
        )

        push_stage(
            state,
            RunStage.CRITIQUE,
            progress_callback=progress_callback,
            summary="Reviewing workflow risks and drafting recommendations.",
        )
        critique_text, critique_msg = run_critic_agent(
            profile=model_to_dict(state.profile),
            plan=model_to_dict(state.plan),
            feature_summary=state.feature_summary,
            evaluation=evaluation,
            outlier_report=outlier_report,
        )
        state.critique_text = critique_text
        state.add_agent_message(critique_msg)

        recommendation_text, recommendation_msg = run_recommendation_agent(
            profile=model_to_dict(state.profile),
            plan=model_to_dict(state.plan),
            evaluation=evaluation,
            critique_text=critique_text,
        )
        state.recommendation_text = recommendation_text
        state.add_agent_message(recommendation_msg)
        critique_warnings = critique_msg.warnings + recommendation_msg.warnings
        complete_stage(
            state,
            RunStage.CRITIQUE,
            progress_callback=progress_callback,
            summary="Generated critique and prioritized recommendations.",
            warnings=critique_warnings,
            details={"agent_warnings": critique_warnings},
        )

        if should_run_explainability(state):
            push_stage(
                state,
                RunStage.EXPLAINABILITY,
                progress_callback=progress_callback,
                summary="Explaining baseline model behavior and important features.",
            )
            explanation_text, explanation_msg = run_explanation_agent(
                model_record=model_record,
                evaluation=evaluation,
                feature_summary=state.feature_summary,
            )
            state.metadata["explanation_text"] = explanation_text
            state.add_agent_message(explanation_msg)
            state.explainability_summary = {
                "text": explanation_text,
                "feature_importance": model_record.get("feature_importance", {}),
            }
            complete_stage(
                state,
                RunStage.EXPLAINABILITY,
                progress_callback=progress_callback,
                summary="Explainability summary generated for the baseline model.",
                warnings=explanation_msg.warnings,
                details={"agent_warnings": explanation_msg.warnings},
            )
        else:
            skip_stage(
                state,
                RunStage.EXPLAINABILITY,
                progress_callback=progress_callback,
                summary="Explainability skipped because no trained model result was available.",
            )

        push_stage(
            state,
            RunStage.REPORTING,
            progress_callback=progress_callback,
            summary="Exporting markdown and HTML master reports.",
        )
        report_paths = export_reports(state)
        state.metadata["report_paths"] = report_paths
        complete_stage(
            state,
            RunStage.REPORTING,
            progress_callback=progress_callback,
            summary=f"Generated {len(report_paths)} report artifact(s).",
            details={"report_paths": report_paths},
        )

        push_stage(
            state,
            RunStage.ARTIFACTS,
            progress_callback=progress_callback,
            summary="Persisting downloadable datasets, charts, metadata, and model artifacts.",
        )
        artifact_paths = persist_pipeline_outputs(
            state=state,
            cleaned_df=cleaned_df,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            report_paths=report_paths,
        )
        state.metadata["artifact_paths"] = artifact_paths

        for key, path in artifact_paths.items():
            if not path:
                continue
            artifact_type = ArtifactType.METADATA
            if "report" in key:
                artifact_type = ArtifactType.REPORT
            elif "dataset" in key or "train" in key or "test" in key:
                artifact_type = ArtifactType.DATASET
            elif "model" in key:
                artifact_type = ArtifactType.MODEL
            elif "pipeline" in key or "preprocessor" in key:
                artifact_type = ArtifactType.PIPELINE

            state.add_artifact(
                ArtifactRecord(
                    artifact_type=artifact_type,
                    name=key,
                    path=str(path),
                    description=key,
                )
            )

        complete_stage(
            state,
            RunStage.ARTIFACTS,
            progress_callback=progress_callback,
            summary=f"Persisted {len(artifact_paths)} artifact path(s).",
            details={"artifact_keys": list(artifact_paths.keys())},
        )
        state.mark_completed("Pipeline completed successfully.")
        return state

    except (AgenticCSVError, ValidationError) as exc:
        failed_stage = state.current_stage if state.current_stage != RunStage.INITIALIZED else RunStage.SCHEMA
        fail_stage(
            state,
            failed_stage,
            progress_callback=progress_callback,
            summary=str(exc),
        )
        state.mark_failed(str(exc))
        _append_run_issue(
            state,
            stage=failed_stage,
            severity=SeverityLevel.CRITICAL,
            message=str(exc),
        )
        return state
    except Exception as exc:
        failed_stage = state.current_stage if state.current_stage != RunStage.INITIALIZED else RunStage.SCHEMA
        fail_stage(
            state,
            failed_stage,
            progress_callback=progress_callback,
            summary=f"Unhandled pipeline failure: {exc}",
        )
        state.mark_failed(str(exc))
        _append_run_issue(
            state,
            stage=failed_stage,
            severity=SeverityLevel.CRITICAL,
            message=f"Unhandled pipeline failure: {exc}",
            details={"exception_type": type(exc).__name__},
        )
        return state
