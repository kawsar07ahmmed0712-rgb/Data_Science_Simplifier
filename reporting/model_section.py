from __future__ import annotations

from core.state import AnalysisState
from core.executor import model_to_dict
from reporting.report_utils import format_scalar, markdown_table


def build_model_section(state: AnalysisState) -> str:
    model_result = model_to_dict(state.model_result)
    evaluation = state.metadata.get("evaluation", {})
    metrics = evaluation.get("metrics", {})

    workflow_rows = [
        [
            event.title,
            event.status.value,
            event.started_at.isoformat() if event.started_at else "N/A",
            event.ended_at.isoformat() if event.ended_at else "N/A",
            event.summary,
        ]
        for event in state.workflow_events
    ]

    metric_rows = [
        [name, value]
        for name, value in metrics.items()
        if not isinstance(value, (dict, list))
    ]

    lines = [
        "## Step 6 Workflow and Outcome",
        f"Run status: {state.current_stage.value}",
        f"Model name: {model_result.get('model_name') or 'N/A'}",
        f"Model family: {model_result.get('model_family') or 'N/A'}",
        f"Problem type: {model_result.get('problem_type') or state.problem_type.value}",
        "",
        "### Workflow Timeline",
        markdown_table(
            ["Stage", "Status", "Started", "Ended", "Summary"],
            workflow_rows,
        ),
        "",
        "### Evaluation Snapshot",
        markdown_table(["Metric", "Value"], metric_rows),
        "",
        f"Artifacts generated: {len(state.artifacts)}",
        f"Report paths: {format_scalar(list(state.metadata.get('report_paths', {}).values()))}",
    ]

    return "\n".join(lines)
