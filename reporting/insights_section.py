from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import paragraph


def build_insights_section(state: AnalysisState) -> str:
    return "\n".join(
        [
            "## Step 8 Insights",
            paragraph(state.insight_text or "No insight text generated."),
        ]
    )
