from __future__ import annotations

import re
from html import escape
from pathlib import Path


_ORDERED_LIST_RE = re.compile(r"^\d+\.\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\|\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?$")


def _render_inline(text: str) -> str:
    rendered = escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"_([^_]+)_", r"<em>\1</em>", rendered)
    return rendered


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _close_list(html_parts: list[str], open_list: str | None) -> str | None:
    if open_list is not None:
        html_parts.append(f"</{open_list}>")
    return None


def markdown_like_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Master Report</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f4f1ea; color: #18212c; }",
        ".page { max-width: 1120px; margin: 0 auto; padding: 32px 24px 64px; }",
        ".report-shell { background: #fffdf8; border: 1px solid #d8d1c2; border-radius: 24px; padding: 32px; box-shadow: 0 20px 48px rgba(24, 33, 44, 0.08); }",
        "h1, h2, h3 { color: #102033; margin-top: 1.4em; }",
        "h1 { font-size: 2rem; margin-top: 0; border-bottom: 2px solid #d7c39d; padding-bottom: 0.4rem; }",
        "h2 { font-size: 1.4rem; padding-top: 0.4rem; border-top: 1px solid #ece4d6; }",
        "h3 { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.04em; color: #705b2b; }",
        "p, li { font-size: 0.97rem; line-height: 1.65; }",
        "ul, ol { margin: 0.35rem 0 1rem 1.4rem; }",
        "table { width: 100%; border-collapse: collapse; margin: 0.75rem 0 1.25rem; font-size: 0.92rem; }",
        "thead th { background: #f1e7d2; color: #102033; }",
        "th, td { border: 1px solid #d8d1c2; padding: 0.6rem 0.7rem; text-align: left; vertical-align: top; }",
        "tbody tr:nth-child(even) { background: #faf6ee; }",
        "code { background: #efe8dc; padding: 0.12rem 0.35rem; border-radius: 6px; font-family: 'Consolas', monospace; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='page'><div class='report-shell'>",
    ]

    open_list: str | None = None
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            open_list = _close_list(html_parts, open_list)
            index += 1
            continue

        if stripped.startswith("### "):
            open_list = _close_list(html_parts, open_list)
            html_parts.append(f"<h3>{_render_inline(stripped[4:])}</h3>")
            index += 1
            continue

        if stripped.startswith("## "):
            open_list = _close_list(html_parts, open_list)
            html_parts.append(f"<h2>{_render_inline(stripped[3:])}</h2>")
            index += 1
            continue

        if stripped.startswith("# "):
            open_list = _close_list(html_parts, open_list)
            html_parts.append(f"<h1>{_render_inline(stripped[2:])}</h1>")
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[index + 1].strip()):
            open_list = _close_list(html_parts, open_list)
            header_cells = _split_table_row(stripped)
            data_rows: list[list[str]] = []
            index += 2
            while index < len(lines):
                row = lines[index].strip()
                if not row.startswith("|"):
                    break
                data_rows.append(_split_table_row(row))
                index += 1

            html_parts.append("<table><thead><tr>")
            html_parts.extend(f"<th>{_render_inline(cell)}</th>" for cell in header_cells)
            html_parts.append("</tr></thead><tbody>")
            for row in data_rows:
                html_parts.append("<tr>")
                html_parts.extend(f"<td>{_render_inline(cell)}</td>" for cell in row)
                html_parts.append("</tr>")
            html_parts.append("</tbody></table>")
            continue

        if stripped.startswith("- "):
            if open_list != "ul":
                open_list = _close_list(html_parts, open_list)
                html_parts.append("<ul>")
                open_list = "ul"
            html_parts.append(f"<li>{_render_inline(stripped[2:])}</li>")
            index += 1
            continue

        if _ORDERED_LIST_RE.match(stripped):
            if open_list != "ol":
                open_list = _close_list(html_parts, open_list)
                html_parts.append("<ol>")
                open_list = "ol"
            html_parts.append(f"<li>{_render_inline(_ORDERED_LIST_RE.sub('', stripped, count=1))}</li>")
            index += 1
            continue

        open_list = _close_list(html_parts, open_list)
        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if (
                not next_line
                or next_line.startswith("#")
                or next_line.startswith("- ")
                or next_line.startswith("|")
                or _ORDERED_LIST_RE.match(next_line)
            ):
                break
            paragraph_lines.append(next_line)
            index += 1
        html_parts.append(f"<p>{_render_inline(' '.join(paragraph_lines))}</p>")

    open_list = _close_list(html_parts, open_list)
    html_parts.extend(["</div></div>", "</body>", "</html>"])
    return "\n".join(html_parts)


def save_html_report(markdown_text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = markdown_like_to_html(markdown_text)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
