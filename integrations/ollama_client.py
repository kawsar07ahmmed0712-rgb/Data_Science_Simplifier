from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import get_active_model_profile, get_settings
from core.exceptions import AgentExecutionError


@dataclass(frozen=True, slots=True)
class OllamaResponse:
    model: str
    response_text: str
    raw: dict[str, Any]


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise AgentExecutionError(
            f"Ollama HTTP error {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise AgentExecutionError(
            f"Could not connect to Ollama server: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AgentExecutionError("Ollama returned non-JSON response.") from exc


def generate_text(
    *,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    model_name: str | None = None,
    timeout_seconds: int | None = None,
    format_json: bool = False,
) -> OllamaResponse:
    settings = get_settings()
    profile = get_active_model_profile()

    model = model_name or settings.ollama_model
    timeout = timeout_seconds or profile.timeout_seconds
    endpoint = f"{settings.ollama_host.rstrip('/')}/api/generate"

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else profile.temperature,
            "top_p": top_p if top_p is not None else profile.top_p,
            "num_ctx": num_ctx if num_ctx is not None else profile.num_ctx,
        },
    }

    if system_prompt:
        payload["system"] = system_prompt

    if format_json and profile.supports_json_mode:
        payload["format"] = "json"

    raw = _post_json(endpoint, payload, timeout=timeout)
    response_text = str(raw.get("response", "")).strip()

    if not response_text:
        raise AgentExecutionError("Ollama returned an empty response.")

    return OllamaResponse(
        model=model,
        response_text=response_text,
        raw=raw,
    )


def check_ollama_health(timeout_seconds: int = 5) -> bool:
    settings = get_settings()
    url = f"{settings.ollama_host.rstrip('/')}/api/tags"
    request = urllib.request.Request(url=url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status == 200
    except Exception:
        return False