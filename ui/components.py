from __future__ import annotations

import hashlib
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#f5f1e8",
        "panel": "#fffdf8",
        "panel_alt": "#f1e6d1",
        "border": "#d7c39d",
        "text": "#162332",
        "muted": "#6b7280",
        "accent": "#915f1a",
        "accent_soft": "rgba(145, 95, 26, 0.16)",
        "success": "#1e7a57",
        "warning": "#a15c00",
        "danger": "#b42318",
    },
    "dark": {
        "bg": "#101821",
        "panel": "#18232f",
        "panel_alt": "#223244",
        "border": "#36516c",
        "text": "#edf3f9",
        "muted": "#a9bacb",
        "accent": "#f0ad4e",
        "accent_soft": "rgba(240, 173, 78, 0.18)",
        "success": "#33b07a",
        "warning": "#f0ad4e",
        "danger": "#ff6b6b",
    },
}


def apply_theme(theme_mode: str) -> None:
    palette = THEMES.get(theme_mode, THEMES["light"])
    css = f"""
    <style>
    :root {{
        --app-bg: {palette['bg']};
        --panel-bg: {palette['panel']};
        --panel-alt: {palette['panel_alt']};
        --panel-border: {palette['border']};
        --panel-text: {palette['text']};
        --panel-muted: {palette['muted']};
        --panel-accent: {palette['accent']};
        --panel-accent-soft: {palette['accent_soft']};
        --panel-success: {palette['success']};
        --panel-warning: {palette['warning']};
        --panel-danger: {palette['danger']};
    }}
    .stApp {{
        background:
            radial-gradient(circle at top left, var(--panel-accent-soft), transparent 34%),
            linear-gradient(180deg, var(--app-bg) 0%, var(--app-bg) 100%);
        color: var(--panel-text);
    }}
    .block-container {{
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }}
    .hero-card, .status-card, .timeline-card, .text-card {{
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: 22px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.08);
    }}
    .hero-card {{
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--panel-text);
        margin: 0 0 0.35rem 0;
    }}
    .hero-caption {{
        color: var(--panel-muted);
        font-size: 1rem;
        margin: 0;
        max-width: 72ch;
    }}
    .kpi-card {{
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        min-height: 102px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.06);
    }}
    .kpi-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--panel-muted);
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--panel-text);
        margin-top: 0.3rem;
    }}
    .kpi-help {{
        margin-top: 0.4rem;
        font-size: 0.88rem;
        color: var(--panel-muted);
    }}
    .section-label {{
        color: var(--panel-accent);
        font-weight: 700;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0.2rem 0 0.7rem 0;
    }}
    .status-card {{
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
    }}
    .status-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.8rem;
    }}
    .status-tile {{
        background: var(--panel-alt);
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
    }}
    .status-title {{
        font-size: 0.85rem;
        color: var(--panel-muted);
    }}
    .status-pill {{
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.45rem;
    }}
    .status-pill.ok {{
        background: rgba(30, 122, 87, 0.16);
        color: var(--panel-success);
    }}
    .status-pill.warn {{
        background: rgba(161, 92, 0, 0.16);
        color: var(--panel-warning);
    }}
    .status-pill.fail {{
        background: rgba(180, 35, 24, 0.16);
        color: var(--panel-danger);
    }}
    .timeline-wrap {{
        display: grid;
        gap: 0.85rem;
    }}
    .timeline-card {{
        padding: 0.95rem 1.1rem;
    }}
    .timeline-head {{
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
    }}
    .timeline-title {{
        font-weight: 700;
        color: var(--panel-text);
    }}
    .timeline-meta {{
        color: var(--panel-muted);
        font-size: 0.86rem;
        margin-top: 0.3rem;
    }}
    .timeline-summary {{
        margin-top: 0.55rem;
        color: var(--panel-text);
        font-size: 0.92rem;
    }}
    .timeline-warning {{
        color: var(--panel-warning);
        font-size: 0.85rem;
        margin-top: 0.45rem;
    }}
    .text-card {{
        padding: 1rem 1.15rem;
    }}
    .text-title {{
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        color: var(--panel-text);
    }}
    .text-body {{
        color: var(--panel-text);
        line-height: 1.65;
        white-space: pre-wrap;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{escape(title)}</div>
            <p class="hero-caption">{escape(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(title: str) -> None:
    st.markdown(f'<div class="section-label">{escape(title)}</div>', unsafe_allow_html=True)


def show_kpi(label: str, value: Any, help_text: str | None = None) -> None:
    value_text = escape(str(value))
    help_html = f'<div class="kpi-help">{escape(help_text)}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{value_text}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_json(title: str, payload: Any) -> None:
    with st.expander(title, expanded=False):
        st.json(payload)


def show_dataframe(title: str, df: pd.DataFrame, rows: int = 20) -> None:
    st.subheader(title)
    st.dataframe(df.head(rows), use_container_width=True)
    if len(df) > rows:
        st.caption(f"Showing the first {rows} rows of {len(df)}.")


def render_text_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="text-card">
            <div class="text-title">{escape(title)}</div>
            <div class="text-body">{escape(body or "N/A")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_class(ok: bool, warning: bool = False) -> str:
    if ok:
        return "ok"
    if warning:
        return "warn"
    return "fail"


def render_health_checks(
    *,
    required_packages: dict[str, bool],
    optional_packages: dict[str, bool],
    ollama_available: bool,
) -> None:
    tiles: list[str] = []

    required_ok = all(required_packages.values())
    tiles.append(
        _render_status_tile(
            "Required Packages",
            "Ready" if required_ok else "Missing packages",
            _status_class(required_ok),
            f"{sum(required_packages.values())}/{len(required_packages)} detected",
        )
    )

    optional_ok = sum(optional_packages.values())
    tiles.append(
        _render_status_tile(
            "Optional Packages",
            f"{optional_ok}/{len(optional_packages)} available",
            _status_class(optional_ok == len(optional_packages), warning=optional_ok > 0),
            "Optional analytics are feature-gated.",
        )
    )

    tiles.append(
        _render_status_tile(
            "Ollama",
            "Reachable" if ollama_available else "Fallback mode",
            _status_class(ollama_available, warning=not ollama_available),
            "LLM-backed summaries fall back deterministically if unavailable.",
        )
    )

    st.markdown(
        '<div class="status-card"><div class="status-grid">' + "".join(tiles) + "</div></div>",
        unsafe_allow_html=True,
    )


def _render_status_tile(title: str, label: str, css_class: str, detail: str) -> str:
    return (
        '<div class="status-tile">'
        f'<div class="status-title">{escape(title)}</div>'
        f'<div class="status-pill {css_class}">{escape(label)}</div>'
        f'<div class="timeline-meta">{escape(detail)}</div>'
        "</div>"
    )


def render_workflow_timeline(events: list[Any]) -> None:
    if not events:
        st.info("Workflow timeline will appear once a run starts.")
        return

    ordered_events = sorted(
        events,
        key=lambda event: event.details.get("order", 9999) if hasattr(event, "details") else 9999,
    )

    cards: list[str] = []
    for event in ordered_events:
        status = getattr(event, "status", None)
        status_value = getattr(status, "value", "pending")
        status_class = {
            "completed": "ok",
            "running": "warn",
            "pending": "warn",
            "skipped": "warn",
            "failed": "fail",
        }.get(status_value, "warn")
        started_at = getattr(event, "started_at", None)
        ended_at = getattr(event, "ended_at", None)
        warnings = getattr(event, "warnings", [])

        meta = []
        if started_at:
            meta.append(f"Started: {started_at.isoformat(timespec='seconds')}")
        if ended_at:
            meta.append(f"Ended: {ended_at.isoformat(timespec='seconds')}")
        if not meta:
            meta.append("Awaiting execution")

        warning_html = ""
        if warnings:
            warning_html = (
                f'<div class="timeline-warning">Warnings: {escape(", ".join(map(str, warnings[:4])))}'
                "</div>"
            )

        cards.append(
            (
                '<div class="timeline-card">'
                '<div class="timeline-head">'
                "<div>"
                f'<div class="timeline-title">{escape(getattr(event, "title", "Stage"))}</div>'
                f'<div class="timeline-meta">{escape(" | ".join(meta))}</div>'
                "</div>"
                f'<div class="status-pill {status_class}">{escape(status_value.replace("_", " ").title())}</div>'
                "</div>"
                f'<div class="timeline-summary">{escape(getattr(event, "summary", ""))}</div>'
                f"{warning_html}"
                "</div>"
            )
        )

    st.markdown('<div class="timeline-wrap">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_download_button(label: str, file_path: str, *, key: str | None = None) -> None:
    path = Path(file_path)
    if not path.exists():
        return
    mime = "text/plain"
    suffix = path.suffix.lower()
    if suffix == ".html":
        mime = "text/html"
    elif suffix == ".md":
        mime = "text/markdown"
    elif suffix == ".json":
        mime = "application/json"
    elif suffix == ".csv":
        mime = "text/csv"
    elif suffix == ".joblib":
        mime = "application/octet-stream"

    with path.open("rb") as f:
        resolved_key = key or hashlib.md5(f"{label}|{path.resolve()}".encode("utf-8")).hexdigest()
        st.download_button(
            label=label,
            data=f.read(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
            key=resolved_key,
        )
