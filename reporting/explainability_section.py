from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import markdown_table, paragraph


def build_explainability_section(state: AnalysisState) -> str:
    feature_importance = state.explainability_summary.get("feature_importance", {})
    importance_rows = [
        [feature, score]
        for feature, score in list(feature_importance.items())[:12]
    ]

    lines = [
        "## Step 7 Explainability",
        paragraph(state.metadata.get("explanation_text") or "Explainability was not generated for this run."),
        "",
        "### Top Feature Importance",
        markdown_table(["Feature", "Importance"], importance_rows),
    ]
    return "\n".join(lines)
