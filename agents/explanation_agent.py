from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.prompt_builder import (
    EXPLANATION_SYSTEM_FALLBACK,
    join_sections,
    load_prompt_template,
    render_json_block,
)
from agents.prompt_guard import filter_large_payload
from core.enums import AgentName


def _heuristic_explanation(payload: dict[str, Any]) -> str:
    model_record = payload.get("model_record", {})
    feature_importance = model_record.get("feature_importance", {})
    evaluation = payload.get("evaluation", {})

    lines: list[str] = []
    lines.append(
        f"Baseline model used: {model_record.get('model_name', 'unknown_model')} "
        f"({model_record.get('model_family', 'unknown_family')})."
    )

    if feature_importance:
        top_features = list(feature_importance.items())[:10]
        feature_text = ", ".join(f"{name} ({score})" for name, score in top_features)
        lines.append("Top important features: " + feature_text + ".")

    metrics = evaluation.get("metrics", {})
    metric_pairs = [f"{k}={v}" for k, v in list(metrics.items())[:5] if not isinstance(v, (dict, list))]
    if metric_pairs:
        lines.append("Key metric snapshot: " + ", ".join(metric_pairs) + ".")

    lines.append("Treat this explanation as baseline-level guidance until validated with deeper diagnostics such as SHAP or segment-level error analysis.")
    return " ".join(lines)


def run_explanation_agent(
    *,
    model_record: dict[str, Any],
    evaluation: dict[str, Any],
    feature_summary: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> tuple[str, object]:
    system_prompt = load_prompt_template(
        "explanation_prompt.txt",
        EXPLANATION_SYSTEM_FALLBACK,
    )

    payload = filter_large_payload(
        {
            "model_record": model_record,
            "evaluation": evaluation,
            "feature_summary": feature_summary or {},
        }
    )

    user_prompt = join_sections(
        "Explain the model behavior and important features using only the summary below.",
        render_json_block("EXPLANATION_PAYLOAD", payload),
        "Return plain text with a short explanation and caveats.",
    )

    agent = BaseAgent(
        agent_name=AgentName.EXPLANATION,
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
        warnings.append(f"explanation_agent_fallback_due_to:{type(exc).__name__}")
        fallback = _heuristic_explanation(payload)
        message = agent.build_message(content=fallback, warnings=warnings)
        return fallback, message