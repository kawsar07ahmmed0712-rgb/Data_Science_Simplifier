from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.contracts import AgentMessage
from core.enums import AgentName
from integrations.ollama_client import generate_text


@dataclass(slots=True)
class BaseAgent:
    agent_name: AgentName
    system_prompt: str
    model_name: str | None = None
    use_json_mode: bool = False

    def run(self, *, user_prompt: str) -> str:
        response = generate_text(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            format_json=self.use_json_mode,
        )
        return response.response_text

    def build_message(
        self,
        *,
        content: str,
        structured_output: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> AgentMessage:
        return AgentMessage(
            agent=self.agent_name,
            content=content,
            structured_output=structured_output or {},
            warnings=warnings or [],
        )