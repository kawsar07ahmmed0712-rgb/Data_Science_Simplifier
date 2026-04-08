from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OllamaModelProfile:
    name: str
    family: str
    purpose: str
    temperature: float
    top_p: float
    num_ctx: int
    timeout_seconds: int
    supports_json_mode: bool
    recommended_for_planning: bool
    recommended_for_insights: bool
    recommended_for_critique: bool


MODEL_PROFILES: dict[str, OllamaModelProfile] = {
    "qwen2.5:7b": OllamaModelProfile(
        name="qwen2.5:7b",
        family="qwen",
        purpose="balanced_reasoning",
        temperature=0.2,
        top_p=0.9,
        num_ctx=8192,
        timeout_seconds=120,
        supports_json_mode=True,
        recommended_for_planning=True,
        recommended_for_insights=True,
        recommended_for_critique=True,
    ),
    "llama3.1:8b": OllamaModelProfile(
        name="llama3.1:8b",
        family="llama",
        purpose="general_reasoning",
        temperature=0.2,
        top_p=0.9,
        num_ctx=8192,
        timeout_seconds=120,
        supports_json_mode=True,
        recommended_for_planning=True,
        recommended_for_insights=True,
        recommended_for_critique=True,
    ),
    "mistral:7b": OllamaModelProfile(
        name="mistral:7b",
        family="mistral",
        purpose="lightweight_local",
        temperature=0.2,
        top_p=0.9,
        num_ctx=4096,
        timeout_seconds=120,
        supports_json_mode=False,
        recommended_for_planning=False,
        recommended_for_insights=True,
        recommended_for_critique=False,
    ),
}


def get_active_model_profile() -> OllamaModelProfile:
    selected = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    if selected in MODEL_PROFILES:
        return MODEL_PROFILES[selected]
    return OllamaModelProfile(
        name=selected,
        family="custom",
        purpose="custom_user_selected",
        temperature=0.2,
        top_p=0.9,
        num_ctx=8192,
        timeout_seconds=120,
        supports_json_mode=False,
        recommended_for_planning=True,
        recommended_for_insights=True,
        recommended_for_critique=True,
    )