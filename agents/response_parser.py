from __future__ import annotations

import json
from typing import Any


def extract_first_json_block(text: str) -> str | None:
    if not text:
        return None

    start_positions = []
    for char in ("{", "["):
        idx = text.find(char)
        if idx != -1:
            start_positions.append(idx)

    if not start_positions:
        return None

    start = min(start_positions)
    opening = text[start]
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def parse_json_response(text: str) -> dict[str, Any] | list[Any] | None:
    candidate = extract_first_json_block(text)
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def clean_text_response(text: str) -> str:
    return str(text or "").strip()


def parse_agent_response(
    text: str,
    *,
    expect_json: bool = False,
) -> tuple[str, dict[str, Any] | list[Any] | None]:
    cleaned = clean_text_response(text)
    parsed_json = parse_json_response(cleaned)

    if expect_json and parsed_json is not None:
        return cleaned, parsed_json

    if expect_json:
        return cleaned, None

    return cleaned, parsed_json