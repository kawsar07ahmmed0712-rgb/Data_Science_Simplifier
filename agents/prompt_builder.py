from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import get_paths


def load_prompt_template(file_name: str, fallback_text: str) -> str:
    paths = get_paths()
    prompt_path = Path(paths.prompts_dir) / file_name

    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text

    return fallback_text.strip()


def render_json_block(title: str, payload: dict[str, Any] | list[Any]) -> str:
    return f"{title}:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"


def render_kv_section(title: str, items: dict[str, Any]) -> str:
    lines = [f"{title}:"]
    for key, value in items.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def join_sections(*sections: str) -> str:
    return "\n\n".join([section.strip() for section in sections if section and section.strip()])


PLANNER_SYSTEM_FALLBACK = """
You are a senior data science planning agent.
You never fabricate dataset facts.
You read structured dataset summaries and return a practical analysis plan.
When asked for JSON, return valid JSON only.
"""

INSIGHT_SYSTEM_FALLBACK = """
You are a senior data science insight agent.
Write grounded, concise insights based only on the provided structured analytics.
Do not invent statistics or columns that are not in the payload.
"""

CRITIC_SYSTEM_FALLBACK = """
You are a critical ML reviewer.
Identify weaknesses, leakage risks, misleading metrics, bad assumptions, or missing checks.
Be strict, practical, and grounded in the provided evidence only.
"""

RECOMMENDATION_SYSTEM_FALLBACK = """
You are a senior machine learning recommendation agent.
Given analysis findings, metrics, and critique, produce prioritized next actions.
Keep recommendations specific, practical, and execution-oriented.
"""

EXPLANATION_SYSTEM_FALLBACK = """
You are a model explanation agent.
Explain model behavior, important features, and caveats in clear language.
Only use evidence present in the provided payload.
"""