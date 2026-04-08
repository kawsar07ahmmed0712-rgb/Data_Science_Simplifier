from __future__ import annotations

from core.state import AnalysisState
from reporting.report_utils import paragraph


def build_critique_section(state: AnalysisState) -> str:
    return "\n".join(
        [
            "## Step 9 Critique",
            paragraph(state.critique_text or "No critique text generated."),
        ]
    )
