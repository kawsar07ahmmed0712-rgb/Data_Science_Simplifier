from __future__ import annotations

from core.enums import ProblemType, RunStage
from core.state import AnalysisState
from reporting.html_report import save_html_report
from reporting.master_report_builder import build_master_markdown_report


def test_master_report_contains_structural_audit_and_recommendations(tmp_path) -> None:
    state = AnalysisState(run_id="report-test")
    state.update_problem_type(ProblemType.CLASSIFICATION)
    state.plan.target_column = "target"
    state.raw_dataframe_shape = (10, 3)
    state.cleaned_dataframe_shape = (9, 3)
    state.profile.row_count = 10
    state.profile.column_count = 3
    state.profile.duplicate_count = 1
    state.profile.missing_counts = {"income": 1}
    state.profile.missing_percentages = {"income": 10.0}
    state.profile.unique_counts = {"target": 2, "income": 9, "segment": 3}
    state.profile.warnings = ["missing_income_values"]
    state.insight_text = "Income concentration differs across target classes."
    state.critique_text = "Baseline evaluation is still shallow."
    state.recommendation_text = "Benchmark a second classifier and review leakage risk."
    state.metadata["raw_profile_details"] = {
        "constant_columns": [],
        "unique_ratios": {"target": 0.2, "income": 0.9, "segment": 0.3},
        "column_memory_usage_bytes": {"income": 128, "segment": 96, "target": 64},
        "numeric_summary": {"income": {"mean": 2000.0, "std": 150.0, "min": 1500.0, "50%": 1980.0, "max": 2300.0, "skew": 0.3}},
        "categorical_summary": {"segment": {"count": 10, "unique": 3, "mode": "A", "top_values": {"A": 5, "B": 3}}},
    }
    state.eda_results = {
        "problem_type_hint": "classification",
        "target_analysis": {"summary": {"class_distribution": {"0": 6, "1": 4}}},
        "bivariate_analysis": {"categorical_vs_target": [], "numeric_vs_target": []},
        "correlation_analysis": {"top_pairs": [], "target_correlations": []},
        "segmentation_hints": ["segment"],
    }
    state.start_workflow_stage(RunStage.INGESTION)
    state.complete_workflow_stage(RunStage.INGESTION, summary="Loaded file")
    state.start_workflow_stage(RunStage.SCHEMA)
    state.complete_workflow_stage(RunStage.SCHEMA, summary="Schema inferred")

    markdown_text = build_master_markdown_report(state)
    html_path = save_html_report(markdown_text, tmp_path / "master_report.html")
    html_text = html_path.read_text(encoding="utf-8")

    assert "## Step 1 Structural Audit" in markdown_text
    assert "## Step 10 Recommendations" in markdown_text
    assert "Step 1 Structural Audit" in html_text
    assert "Recommendations" in html_text
