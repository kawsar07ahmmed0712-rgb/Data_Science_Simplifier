from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.prompt_builder import (
    RECOMMENDATION_SYSTEM_FALLBACK,
    join_sections,
    load_prompt_template,
    render_json_block,
)
from agents.prompt_guard import filter_large_payload
from core.enums import AgentName


def _heuristic_recommendations(payload: dict[str, Any]) -> str:
    recs: list[str] = []

    critique_text = str(payload.get("critique_text", "")).lower()
    evaluation = payload.get("evaluation", {})
    metrics = evaluation.get("metrics", {})

    if "leakage" in critique_text:
        recs.append("Recheck feature list and explicitly exclude any columns derived from the target.")
    if "imbalance" in critique_text:
        recs.append("Test class balancing strategies and compare recall/F1 against the current baseline.")
    if "missing" in critique_text:
        recs.append("Evaluate column-wise imputation strategy and consider dropping extremely sparse fields.")

    if "accuracy" in metrics and "f1_macro" in metrics:
        recs.append("Compare current baseline against at least one simpler and one stronger alternative model.")

    if not recs:
        recs.extend(
            [
                "Validate the detected target column and business objective before finalizing conclusions.",
                "Run a second benchmark model and compare metrics consistently.",
                "Add explainability and error-segment analysis before deployment decisions.",
            ]
        )

    return "\n".join(f"- {item}" for item in recs[:6])


def run_recommendation_agent(
    *,
    profile: dict[str, Any],
    plan: dict[str, Any],
    evaluation: dict[str, Any],
    critique_text: str,
    model_name: str | None = None,
) -> tuple[str, object]:
    system_prompt = load_prompt_template(
        "recommendation_prompt.txt",
        RECOMMENDATION_SYSTEM_FALLBACK,
    )

    payload = filter_large_payload(
        {
            "profile": profile,
            "plan": plan,
            "evaluation": evaluation,
            "critique_text": critique_text,
        }
    )

    user_prompt = join_sections(
        "Based on the analysis, critique, and metrics below, give practical next-step recommendations.",
        render_json_block("RECOMMENDATION_PAYLOAD", payload),
        "Return plain text with a prioritized recommendation list.",
    )

    agent = BaseAgent(
        agent_name=AgentName.RECOMMENDATION,
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
        warnings.append(f"recommendation_agent_fallback_due_to:{type(exc).__name__}")
        fallback = _heuristic_recommendations(payload)
        message = agent.build_message(content=fallback, warnings=warnings)
        return fallback, message