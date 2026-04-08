from __future__ import annotations

from pathlib import Path

from config import get_paths
from core.state import AnalysisState
from reporting.html_report import save_html_report
from reporting.markdown_report import save_markdown_report
from reporting.master_report_builder import build_master_markdown_report


def export_reports(state: AnalysisState) -> dict[str, str]:
    paths = get_paths()
    report_dir = Path(paths.reports_dir) / state.run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    markdown_text = build_master_markdown_report(state)

    md_path = save_markdown_report(markdown_text, report_dir / "master_report.md")
    html_path = save_html_report(markdown_text, report_dir / "master_report.html")

    return {
        "markdown_report": str(md_path),
        "html_report": str(html_path),
    }