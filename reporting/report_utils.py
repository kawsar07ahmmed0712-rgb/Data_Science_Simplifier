from __future__ import annotations

from typing import Any


def format_scalar(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(format_scalar(item) for item in list(value)[:8]) or "N/A"
    if isinstance(value, dict):
        items = list(value.items())[:6]
        return ", ".join(f"{key}={format_scalar(item)}" for key, item in items) or "N/A"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows available._"

    normalized_headers = [str(header) for header in headers]
    body_rows = [
        [str(format_scalar(cell)).replace("|", "\\|") for cell in row]
        for row in rows
    ]
    header_row = "| " + " | ".join(normalized_headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(normalized_headers)) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in body_rows]
    return "\n".join([header_row, separator_row, *data_rows])


def bullet_lines(items: list[Any], *, empty_message: str = "None") -> str:
    if not items:
        return f"- {empty_message}"
    return "\n".join(f"- {format_scalar(item)}" for item in items)


def paragraph(text: str) -> str:
    return text.strip() if text and text.strip() else "N/A"


def top_items(items: list[Any], *, limit: int) -> list[Any]:
    return list(items[:limit]) if items else []
