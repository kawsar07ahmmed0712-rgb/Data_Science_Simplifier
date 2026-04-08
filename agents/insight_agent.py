from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.prompt_builder import (
    INSIGHT_SYSTEM_FALLBACK,
    join_sections,
    load_prompt_template,
    render_json_block,
)
from agents.prompt_guard import filter_large_payload
from core.enums import AgentName


def _heuristic_insights(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    profile = payload.get("profile", {})
    target_analysis = payload.get("target_analysis", {})
    outlier_report = payload.get("outlier_report", {})
    evaluation = payload.get("evaluation", {})

    row_count = profile.get("row_count")
    column_count = profile.get("column_count")
    duplicate_count = profile.get("duplicate_count")
    warnings = profile.get("warnings", [])

    lines.append(f"Dataset has {row_count} rows and {column_count} columns.")
    if duplicate_count:
        lines.append(f"Duplicate rows detected: {duplicate_count}.")
    if warnings:
        lines.append("Key data quality flags: " + ", ".join(map(str, warnings[:5])) + ".")

    if isinstance(target_analysis, dict) and target_analysis.get("has_target"):
        lines.append(
            f"Target column is '{target_analysis.get('target_column')}' with problem type hint "
            f"'{target_analysis.get('problem_type_hint')}'."
        )

    if isinstance(outlier_report, dict):
        top_columns = outlier_report.get("top_columns", [])
        if top_columns:
            lines.append(
                "Top outlier-heavy columns include: "
                + ", ".join(item["column"] for item in top_columns[:5])
                + "."
            )

    if isinstance(evaluation, dict):
        metrics = evaluation.get("metrics", {})
        if metrics:
            important_pairs = [f"{k}={v}" for k, v in list(metrics.items())[:5] if not isinstance(v, (dict, list))]
            if important_pairs:
                lines.append("Model metrics snapshot: " + ", ".join(important_pairs) + ".")

    return " ".join(lines)


def run_insight_agent(
    *,
    profile: dict[str, Any],
    target_analysis: dict[str, Any],
    eda_summary: dict[str, Any],
    outlier_report: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> tuple[str, object]:
    system_prompt = load_prompt_template(
        "insight_prompt.txt",
        INSIGHT_SYSTEM_FALLBACK,
    )

    payload = filter_large_payload(
        {
            "profile": profile,
            "target_analysis": target_analysis,
            "eda_summary": eda_summary,
            "outlier_report": outlier_report or {},
            "evaluation": evaluation or {},
        }
    )

    user_prompt = join_sections(
        "Write a grounded insight summary based only on the structured analytics below.",
        render_json_block("ANALYTICS_PAYLOAD", payload),
        (
            "Write 1 concise executive summary paragraph and then 4-8 bullet-style findings in plain text. "
            "Do not invent values or columns."
        ),
    )

    agent = BaseAgent(
        agent_name=AgentName.INSIGHT,
        system_prompt=system_prompt,
        model_name=model_name,
        use_json_mode=False,
    )

    warnings: list[str] = []
    try:
        content = agent.run(user_prompt=user_prompt)
        message = agent.build_message(content=content, warnings=warnings)
        return content, message
    except Exception as exc:
        warnings.append(f"insight_agent_fallback_due_to:{type(exc).__name__}")
        fallback = _heuristic_insights(payload)
        message = agent.build_message(content=fallback, warnings=warnings)
        return fallback, message