from __future__ import annotations

from core.state import AnalysisState
from reporting.critique_section import build_critique_section
from reporting.data_quality_section import build_data_quality_section
from reporting.eda_section import build_eda_section
from reporting.executive_summary_builder import build_executive_summary
from reporting.explainability_section import build_explainability_section
from reporting.feature_engineering_section import build_feature_engineering_section
from reporting.insights_section import build_insights_section
from reporting.model_section import build_model_section
from reporting.outlier_section import build_outlier_section
from reporting.recommendation_section import build_recommendation_section
from reporting.structural_audit_section import build_structural_audit_section


def build_master_markdown_report(state: AnalysisState) -> str:
    sections = [
        "# Master Report",
        "## Executive Summary",
        build_executive_summary(state),
        build_structural_audit_section(state),
        build_data_quality_section(state),
        build_eda_section(state),
        build_outlier_section(state),
        build_feature_engineering_section(state),
        build_model_section(state),
        build_explainability_section(state),
        build_insights_section(state),
        build_critique_section(state),
        build_recommendation_section(state),
    ]
    return "\n\n".join(section for section in sections if section.strip())
