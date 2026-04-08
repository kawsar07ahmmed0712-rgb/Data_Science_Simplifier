from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True, slots=True)
class RunUIOptions:
    target_column: str | None
    auto_run: bool
    theme_mode: str
    show_debug_json: bool


def render_sidebar_options(columns: list[str]) -> RunUIOptions:
    st.sidebar.header("Run Controls")

    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"

    target_options = ["AUTO"] + columns
    target_default_index = 0
    selected_target = st.session_state.get("selected_target", "AUTO")
    if selected_target in target_options:
        target_default_index = target_options.index(selected_target)

    target_choice = st.sidebar.selectbox(
        "Target Column",
        target_options,
        index=target_default_index,
        key="selected_target",
    )
    auto_run = st.sidebar.checkbox("Run immediately after upload", value=False)
    theme_mode = st.sidebar.radio(
        "Theme",
        options=["light", "dark"],
        index=0 if st.session_state.get("theme_mode", "light") == "light" else 1,
        horizontal=True,
    )
    show_debug_json = st.sidebar.checkbox("Show debug JSON", value=False)
    st.session_state["theme_mode"] = theme_mode

    return RunUIOptions(
        target_column=None if target_choice == "AUTO" else target_choice,
        auto_run=auto_run,
        theme_mode=theme_mode,
        show_debug_json=show_debug_json,
    )
