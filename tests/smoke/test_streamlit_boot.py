from __future__ import annotations

import runpy
import sys
import types


class _FakeSidebar:
    def __init__(self, module) -> None:
        self._module = module

    def header(self, *_args, **_kwargs) -> None:
        return None

    def selectbox(self, _label, options, index=0, key=None):
        value = options[index]
        if key:
            self._module.session_state[key] = value
        return value

    def checkbox(self, _label, value=False):
        return value

    def radio(self, _label, options, index=0, horizontal=False):
        return options[index]


class _FakePlaceholder:
    def __init__(self, module) -> None:
        self._module = module

    def container(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_fake_streamlit() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    module.session_state = {}
    module.logged_info: list[str] = []
    module.logged_markdown: list[str] = []
    module.sidebar = _FakeSidebar(module)
    module.set_page_config = lambda **_kwargs: None
    module.markdown = lambda body, **_kwargs: module.logged_markdown.append(body)
    module.file_uploader = lambda *_args, **_kwargs: None
    module.empty = lambda: _FakePlaceholder(module)
    module.info = lambda body: module.logged_info.append(body)
    return module


def test_streamlit_app_boots_without_upload(monkeypatch) -> None:
    fake_streamlit = _build_fake_streamlit()

    sys.modules.pop("ui.components", None)
    sys.modules.pop("ui.sidebar", None)
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setattr(
        "config.get_package_flags",
        lambda: types.SimpleNamespace(
            pandas=True,
            numpy=True,
            sklearn=True,
            streamlit=True,
            plotly=True,
            matplotlib=True,
            scipy=True,
            duckdb=False,
            shap=False,
            pyod=False,
            cleanlab=False,
            evidently=False,
        ),
    )
    monkeypatch.setattr("integrations.ollama_client.check_ollama_health", lambda: False)

    runpy.run_path("ui/streamlit_app.py", run_name="__main__")

    assert any("Upload a CSV file to begin." in item for item in fake_streamlit.logged_info)
