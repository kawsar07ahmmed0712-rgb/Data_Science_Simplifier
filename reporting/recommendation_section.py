from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import paragraph


def build_recommendation_section(state: AnalysisState) -> str:
    return "\n".join(
        [
            "## Step 10 Recommendations",
            paragraph(state.recommendation_text or "No recommendations generated."),
        ]
    )
