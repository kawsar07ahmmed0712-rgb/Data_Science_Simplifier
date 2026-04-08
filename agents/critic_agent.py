from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.prompt_builder import (
    CRITIC_SYSTEM_FALLBACK,
    join_sections,
    load_prompt_template,
    render_json_block,
)
from agents.prompt_guard import filter_large_payload
from core.enums import AgentName


def _heuristic_critique(payload: dict[str, Any]) -> str:
    critiques: list[str] = []

    profile = payload.get("profile", {})
    plan = payload.get("plan", {})
    evaluation = payload.get("evaluation", {})
    fe_summary = payload.get("feature_summary", {})

    missing_pct = profile.get("missing_percentages", {})
    if any(float(value) >= 40.0 for value in missing_pct.values()):
        critiques.append("High-missingness columns exist and may reduce model reliability.")

    if profile.get("duplicate_count", 0) > 0:
        critiques.append("Duplicate rows were present; verify whether business duplicates remain after cleaning.")

    leakage_columns = fe_summary.get("input_feature_groups", {}).get("leakage_risk_columns", [])
    if leakage_columns:
        critiques.append(
            "Potential leakage-risk columns were detected: " + ", ".join(leakage_columns[:5]) + "."
        )

    metrics = evaluation.get("metrics", {})
    if "accuracy" in metrics and "recall_macro" in metrics:
        accuracy = metrics.get("accuracy", 0)
        recall_macro = metrics.get("recall_macro", 0)
        if isinstance(accuracy, (int, float)) and isinstance(recall_macro, (int, float)):
            if accuracy - recall_macro >= 0.10:
                critiques.append("Accuracy may look stronger than macro recall; class imbalance might still matter.")

    if not critiques:
        critiques.append("No severe red flags were detected from the provided structured summary, but baseline-only evaluation is still limited.")

    return " ".join(critiques)


def run_critic_agent(
    *,
    profile: dict[str, Any],
    plan: dict[str, Any],
    feature_summary: dict[str, Any],
    evaluation: dict[str, Any],
    outlier_report: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> tuple[str, object]:
    system_prompt = load_prompt_template(
        "critic_prompt.txt",
        CRITIC_SYSTEM_FALLBACK,
    )

    payload = filter_large_payload(
        {
            "profile": profile,
            "plan": plan,
            "feature_summary": feature_summary,
            "evaluation": evaluation,
            "outlier_report": outlier_report or {},
        }
    )

    user_prompt = join_sections(
        "Review the following ML workflow summary and identify concrete weaknesses or risks.",
        render_json_block("CRITIQUE_PAYLOAD", payload),
        (
            "Return plain text with: 1 short overall verdict, then 4-8 focused critique points. "
            "Be critical but evidence-based."
        ),
    )

    agent = BaseAgent(
        agent_name=AgentName.CRITIC,
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
        warnings.append(f"critic_agent_fallback_due_to:{type(exc).__name__}")
        fallback = _heuristic_critique(payload)
        message = agent.build_message(content=fallback, warnings=warnings)
        return fallback, message