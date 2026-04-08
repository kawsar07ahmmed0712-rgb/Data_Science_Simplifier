from __future__ import annotations

from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st

from app.startup import ensure_project_directories
from config import get_package_flags
from core.executor import model_to_dict
from core.pipeline import run_pipeline_from_source
from integrations.ollama_client import check_ollama_health
from ui.components import (
    apply_theme,
    render_download_button,
    render_health_checks,
    render_hero,
    render_section_label,
    render_text_card,
    render_workflow_timeline,
    show_dataframe,
    show_json,
    show_kpi,
)
from ui.sidebar import RunUIOptions, render_sidebar_options


st.set_page_config(
    page_title="Agentic CSV Data Scientist",
    layout="wide",
)

ensure_project_directories()


def _safe_preview(uploaded_file) -> pd.DataFrame:
    preview_df = pd.read_csv(uploaded_file, sep=None, engine="python")
    uploaded_file.seek(0)
    return preview_df


def _build_column_audit_df(state) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": column.name,
                "role": column.role.value,
                "dtype": column.inferred_dtype,
                "nullable": column.nullable,
                "missing_count": column.missing_count,
                "unique_count": column.unique_count,
                "notes": ", ".join(column.notes),
            }
            for column in state.schema.columns
        ]
    )


def _build_missingness_df(state) -> pd.DataFrame:
    rows = [
        {
            "column": column,
            "missing_count": count,
            "missing_pct": state.profile.missing_percentages.get(column, 0.0),
        }
        for column, count in state.profile.missing_counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    return pd.DataFrame(rows).sort_values(by=["missing_count", "column"], ascending=[False, True])


def _build_uniqueness_df(state) -> pd.DataFrame:
    details = state.metadata.get("raw_profile_details", {})
    unique_ratios = details.get("unique_ratios", {})
    rows = [
        {
            "column": column,
            "unique_count": count,
            "unique_ratio": unique_ratios.get(column, 0.0),
        }
        for column, count in state.profile.unique_counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["column", "unique_count", "unique_ratio"])
    return pd.DataFrame(rows).sort_values(by=["unique_count", "column"], ascending=[False, True])


def _build_numeric_profile_df(state) -> pd.DataFrame:
    numeric_summary = state.metadata.get("raw_profile_details", {}).get("numeric_summary", {})
    rows = [{"column": column, **stats} for column, stats in numeric_summary.items()]
    return pd.DataFrame(rows)


def _build_categorical_profile_df(state) -> pd.DataFrame:
    categorical_summary = state.metadata.get("raw_profile_details", {}).get("categorical_summary", {})
    rows = []
    for column, stats in categorical_summary.items():
        rows.append(
            {
                "column": column,
                "count": stats.get("count"),
                "unique": stats.get("unique"),
                "mode": stats.get("mode"),
                "top_values": stats.get("top_values"),
            }
        )
    return pd.DataFrame(rows)


def _build_workflow_df(state) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": event.stage.value,
                "title": event.title,
                "status": event.status.value,
                "started_at": event.started_at,
                "ended_at": event.ended_at,
                "summary": event.summary,
                "warnings": ", ".join(event.warnings),
            }
            for event in sorted(state.workflow_events, key=lambda item: item.details.get("order", 999))
        ]
    )


def _build_metric_df(state) -> pd.DataFrame:
    metrics = state.metadata.get("evaluation", {}).get("metrics", {})
    rows = [
        {"metric": key, "value": value}
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    ]
    return pd.DataFrame(rows)


def _build_feature_importance_df(state) -> pd.DataFrame:
    importance = state.explainability_summary.get("feature_importance", {})
    return pd.DataFrame(
        [{"feature": feature, "importance": score} for feature, score in importance.items()]
    )


def _build_outlier_df(state) -> pd.DataFrame:
    outlier_report = state.metadata.get("outlier_report", {})
    return pd.DataFrame(outlier_report.get("top_columns", []))


def _build_anomaly_df(state) -> pd.DataFrame:
    anomaly_summary = state.metadata.get("anomaly_summary", {})
    return pd.DataFrame(anomaly_summary.get("methods_ranked", []))


def _build_cleaning_log_df(state) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action": entry.action,
                "column": entry.column,
                "affected_rows": entry.affected_rows,
                "details": entry.details,
            }
            for entry in state.cleaning.steps_applied
        ]
    )


def _agent_warning_lines(state) -> list[str]:
    warnings: list[str] = []
    for message in state.agent_messages:
        warnings.extend(message.warnings)
    return warnings


def _render_run_results(state, preview_df: pd.DataFrame, show_debug_json: bool) -> None:
    if state.flags.get("run_failed"):
        st.error("Pipeline failed. Review the issues and workflow timeline below.")
        if state.issues:
            for issue in state.issues:
                st.warning(f"{issue.stage.value}: {issue.message}")
    else:
        st.success("Pipeline completed successfully.")

    render_section_label("Run Snapshot")
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        show_kpi("Rows", state.profile.row_count or preview_df.shape[0])
    with kpi_cols[1]:
        show_kpi("Columns", state.profile.column_count or preview_df.shape[1])
    with kpi_cols[2]:
        show_kpi("Problem Type", state.problem_type.value)
    with kpi_cols[3]:
        show_kpi("Target", state.plan.target_column or "None")
    with kpi_cols[4]:
        show_kpi("Current Stage", state.current_stage.value)

    agent_warnings = _agent_warning_lines(state)
    if agent_warnings:
        st.info("Agent fallback/health notes: " + ", ".join(agent_warnings[:6]))

    tabs = st.tabs(
        [
            "Workflow",
            "Structural Audit",
            "Quality",
            "Profiles",
            "Modeling",
            "Narrative",
            "Downloads",
        ]
    )

    with tabs[0]:
        render_section_label("Live Workflow History")
        render_workflow_timeline(state.workflow_events)
        workflow_df = _build_workflow_df(state)
        if not workflow_df.empty:
            st.dataframe(workflow_df, use_container_width=True)

    with tabs[1]:
        render_section_label("Preview")
        show_dataframe("Uploaded Data", preview_df, rows=12)
        render_section_label("Column Audit")
        column_audit_df = _build_column_audit_df(state)
        st.dataframe(column_audit_df, use_container_width=True)
        audit_cols = st.columns(4)
        with audit_cols[0]:
            show_kpi("Numeric", len(state.schema.numeric_columns))
        with audit_cols[1]:
            show_kpi("Categorical", len(state.schema.categorical_columns))
        with audit_cols[2]:
            show_kpi("Datetime", len(state.schema.datetime_columns))
        with audit_cols[3]:
            show_kpi("ID-like", len(state.schema.id_like_columns))
        if show_debug_json:
            show_json("Schema JSON", model_to_dict(state.schema))

    with tabs[2]:
        render_section_label("Missingness")
        missingness_df = _build_missingness_df(state)
        st.dataframe(missingness_df, use_container_width=True)
        render_section_label("Uniqueness")
        uniqueness_df = _build_uniqueness_df(state)
        st.dataframe(uniqueness_df, use_container_width=True)
        render_section_label("Cleaning Log")
        cleaning_log_df = _build_cleaning_log_df(state)
        if cleaning_log_df.empty:
            st.info("No cleaning actions were recorded.")
        else:
            st.dataframe(cleaning_log_df, use_container_width=True)
        if show_debug_json:
            show_json("Profile JSON", model_to_dict(state.profile))
            show_json("Cleaning JSON", model_to_dict(state.cleaning))

    with tabs[3]:
        render_section_label("Categorical Profile")
        categorical_profile_df = _build_categorical_profile_df(state)
        if categorical_profile_df.empty:
            st.info("No categorical profile available.")
        else:
            st.dataframe(categorical_profile_df, use_container_width=True)

        render_section_label("Numeric Profile")
        numeric_profile_df = _build_numeric_profile_df(state)
        if numeric_profile_df.empty:
            st.info("No numeric profile available.")
        else:
            st.dataframe(numeric_profile_df, use_container_width=True)

        render_section_label("Outliers and Anomalies")
        outlier_df = _build_outlier_df(state)
        anomaly_df = _build_anomaly_df(state)
        profile_cols = st.columns(2)
        with profile_cols[0]:
            if outlier_df.empty:
                st.info("No outlier summary available.")
            else:
                st.dataframe(outlier_df, use_container_width=True)
        with profile_cols[1]:
            if anomaly_df.empty:
                st.info("No anomaly summary available.")
            else:
                st.dataframe(anomaly_df, use_container_width=True)

        chart_paths = {
            key: value
            for key, value in state.metadata.get("artifact_paths", {}).items()
            if key.startswith("chart_")
        }
        if chart_paths:
            render_section_label("Chart Gallery")
            for _, path in chart_paths.items():
                st.image(path, use_container_width=True)

        if show_debug_json:
            show_json("EDA JSON", state.eda_results)
            show_json("Outlier JSON", state.metadata.get("outlier_report", {}))
            show_json("Anomaly JSON", state.metadata.get("anomaly_summary", {}))

    with tabs[4]:
        render_section_label("Split and Feature Summary")
        model_cols = st.columns(2)
        with model_cols[0]:
            show_json("Split Summary", model_to_dict(state.split))
            show_json("Feature Summary", state.feature_summary)
        with model_cols[1]:
            metric_df = _build_metric_df(state)
            if metric_df.empty:
                st.info("No evaluation metrics available.")
            else:
                st.dataframe(metric_df, use_container_width=True)
            feature_importance_df = _build_feature_importance_df(state)
            if not feature_importance_df.empty:
                st.dataframe(feature_importance_df, use_container_width=True)
        if show_debug_json:
            show_json("Model Result JSON", model_to_dict(state.model_result))
            show_json("Explainability JSON", state.explainability_summary)

    with tabs[5]:
        render_section_label("Narrative Output")
        narrative_cols = st.columns(2)
        with narrative_cols[0]:
            render_text_card("Insights", state.insight_text or "N/A")
            render_text_card("Critique", state.critique_text or "N/A")
        with narrative_cols[1]:
            render_text_card("Recommendations", state.recommendation_text or "N/A")
            render_text_card("Explainability", state.metadata.get("explanation_text") or "N/A")
        if show_debug_json:
            show_json("Agent Messages", model_to_dict(state.agent_messages))

    with tabs[6]:
        render_section_label("Reports")
        for key, path in state.metadata.get("report_paths", {}).items():
            render_download_button(f"Download {key}", path, key=f"report::{key}")
        render_section_label("Artifacts")
        report_values = set(state.metadata.get("report_paths", {}).values())
        for key, path in state.metadata.get("artifact_paths", {}).items():
            if path in report_values:
                continue
            render_download_button(f"Download {key}", path, key=f"artifact::{key}")
        if show_debug_json:
            show_json("Artifacts JSON", model_to_dict(state.artifacts))


render_hero(
    "Agentic CSV Data Scientist",
    "Upload a CSV, inspect the structural audit, watch the workflow advance step by step, and export a richer master report.",
)

placeholder_state = st.session_state.get("analysis_state")
preview_df = st.session_state.get("preview_df")

uploaded_file = st.file_uploader("Upload CSV or delimited text", type=["csv", "txt", "tsv"])
preview_error = None

if uploaded_file is not None:
    try:
        preview_df = _safe_preview(uploaded_file)
        st.session_state["preview_df"] = preview_df
    except Exception as exc:
        preview_error = str(exc)
        preview_df = None

options = render_sidebar_options(list(preview_df.columns) if preview_df is not None else [])
apply_theme(options.theme_mode)

package_flags = get_package_flags()
required_packages = {
    "pandas": package_flags.pandas,
    "numpy": package_flags.numpy,
    "sklearn": package_flags.sklearn,
    "streamlit": package_flags.streamlit,
}
optional_packages = {
    "plotly": package_flags.plotly,
    "matplotlib": package_flags.matplotlib,
    "scipy": package_flags.scipy,
    "duckdb": package_flags.duckdb,
    "shap": package_flags.shap,
    "pyod": package_flags.pyod,
    "cleanlab": package_flags.cleanlab,
    "evidently": package_flags.evidently,
}
ollama_available = check_ollama_health()

render_section_label("Runtime Health")
render_health_checks(
    required_packages=required_packages,
    optional_packages=optional_packages,
    ollama_available=ollama_available,
)

workflow_placeholder = st.empty()

if preview_error:
    st.error(f"Could not preview the uploaded file: {preview_error}")
elif preview_df is not None:
    render_section_label("Upload Preview")
    show_dataframe("Dataset Preview", preview_df, rows=10)

    control_cols = st.columns([2, 1, 1])
    with control_cols[0]:
        st.caption(
            "Target selection, theme mode, and debug toggles are available in the sidebar. "
            "If Ollama is unavailable, the agent outputs fall back to deterministic summaries."
        )
    run_token = f"{uploaded_file.name}:{uploaded_file.size}:{options.target_column or 'AUTO'}"
    manual_run = False
    with control_cols[1]:
        manual_run = st.button("Run Full Pipeline", type="primary", use_container_width=True)
    with control_cols[2]:
        clear_state = st.button("Clear Results", use_container_width=True)

    if clear_state:
        st.session_state.pop("analysis_state", None)
        st.session_state.pop("last_run_token", None)
        placeholder_state = None

    should_auto_run = options.auto_run and st.session_state.get("last_run_token") != run_token
    should_run = manual_run or should_auto_run

    if should_run:
        def _progress_callback(event, live_state) -> None:
            with workflow_placeholder.container():
                render_section_label("Live Workflow")
                render_workflow_timeline(live_state.workflow_events)

        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        with st.spinner("Running full pipeline..."):
            state = run_pipeline_from_source(
                temp_path,
                target_column=options.target_column,
                progress_callback=_progress_callback,
            )

        st.session_state["analysis_state"] = state
        st.session_state["last_run_token"] = run_token
        placeholder_state = state

if placeholder_state is not None and preview_df is not None:
    with workflow_placeholder.container():
        render_section_label("Workflow")
        render_workflow_timeline(placeholder_state.workflow_events)
    _render_run_results(placeholder_state, preview_df, options.show_debug_json)
elif uploaded_file is None:
    with workflow_placeholder.container():
        render_section_label("Workflow")
        render_workflow_timeline([])
    st.info("Upload a CSV file to begin.")
