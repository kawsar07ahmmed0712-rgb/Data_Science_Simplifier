from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import bullet_lines, format_scalar, paragraph


def build_executive_summary(state: AnalysisState) -> str:
    evaluation = state.metadata.get("evaluation", {})
    metrics = evaluation.get("metrics", {})
    metric_snapshot = [
        f"{name}={value}"
        for name, value in list(metrics.items())[:5]
        if not isinstance(value, (dict, list))
    ]

    lines = [
        f"Run ID: {state.run_id}",
        f"Problem type: {state.problem_type.value}",
        f"Target column: {state.plan.target_column or 'None'}",
        f"Raw shape: {format_scalar(state.raw_dataframe_shape)}",
        f"Cleaned shape: {format_scalar(state.cleaned_dataframe_shape)}",
        f"Current stage: {state.current_stage.value}",
        "",
        "### Headline Findings",
        bullet_lines(
            [
                state.insight_text or "Insight narrative not available.",
                f"Planner risk flags: {format_scalar(state.plan.risk_flags[:6])}",
                (
                    f"Model metric snapshot: {', '.join(metric_snapshot)}"
                    if metric_snapshot
                    else "Model metric snapshot not available."
                ),
            ]
        ),
        "",
        "### Recommendation Snapshot",
        paragraph(state.recommendation_text or "Recommendations not available."),
    ]
    return "\n".join(lines)
