from __future__ import annotations

from core.enums import RunStage
from core.state import AnalysisState
from ui.components import render_workflow_timeline
from ui.sidebar import render_sidebar_options


class _FakeSidebar:
    def __init__(self, streamlit_stub) -> None:
        self._streamlit = streamlit_stub

    def header(self, *_args, **_kwargs) -> None:
        return None

    def selectbox(self, _label, options, index=0, key=None):
        value = "target"
        if key:
            self._streamlit.session_state[key] = value
        return value if value in options else options[index]

    def checkbox(self, label, value=False):
        return {
            "Run immediately after upload": True,
            "Show debug JSON": True,
        }.get(label, value)

    def radio(self, _label, options, index=0, horizontal=False):
        return "dark" if "dark" in options else options[index]


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.sidebar = _FakeSidebar(self)
        self.markdown_calls: list[str] = []
        self.info_calls: list[str] = []

    def markdown(self, body: str, **_kwargs) -> None:
        self.markdown_calls.append(body)

    def info(self, body: str) -> None:
        self.info_calls.append(body)


def test_sidebar_persists_theme_mode(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr("ui.sidebar.st", fake_st)

    options = render_sidebar_options(["target", "income"])

    assert options.target_column == "target"
    assert options.auto_run is True
    assert options.theme_mode == "dark"
    assert options.show_debug_json is True
    assert fake_st.session_state["theme_mode"] == "dark"


def test_workflow_timeline_renders_in_stage_order(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr("ui.components.st", fake_st)

    state = AnalysisState(run_id="ui-workflow")
    state.start_workflow_stage(RunStage.SCHEMA)
    state.complete_workflow_stage(RunStage.SCHEMA, summary="Schema done")
    state.skip_workflow_stage(RunStage.EXPLAINABILITY, summary="Skipped explainability")
    state.start_workflow_stage(RunStage.INGESTION)
    state.complete_workflow_stage(RunStage.INGESTION, summary="Loaded source")

    render_workflow_timeline(list(reversed(state.workflow_events)))

    html = "".join(fake_st.markdown_calls)
    assert "Loaded source" in html
    assert "Skipped explainability" in html
    assert html.index("Ingestion") < html.index("Structural Audit")
