from __future__ import annotations

import json
from typing import Any


def to_safe_json(payload: Any, *, max_chars: int = 24000) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [TRUNCATED]"


def truncate_text(text: str, *, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [TRUNCATED]"


def compact_list(values: list[Any], *, max_items: int = 25) -> list[Any]:
    if len(values) <= max_items:
        return values
    return values[:max_items] + ["...TRUNCATED..."]


def filter_large_payload(payload: dict[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}

    for key, value in payload.items():
        if isinstance(value, str):
            filtered[key] = truncate_text(value)
        elif isinstance(value, list):
            filtered[key] = compact_list(value)
        else:
            filtered[key] = value

    return filtered