from __future__ import annotations

import importlib.util
from dataclasses import dataclass


def _is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@dataclass(frozen=True, slots=True)
class PackageFlags:
    pandas: bool
    numpy: bool
    sklearn: bool
    streamlit: bool
    plotly: bool
    matplotlib: bool
    scipy: bool
    duckdb: bool
    ydata_profiling: bool
    feature_engine: bool
    pyod: bool
    imblearn: bool
    category_encoders: bool
    shap: bool
    cleanlab: bool
    evidently: bool


def get_package_flags() -> PackageFlags:
    return PackageFlags(
        pandas=_is_available("pandas"),
        numpy=_is_available("numpy"),
        sklearn=_is_available("sklearn"),
        streamlit=_is_available("streamlit"),
        plotly=_is_available("plotly"),
        matplotlib=_is_available("matplotlib"),
        scipy=_is_available("scipy"),
        duckdb=_is_available("duckdb"),
        ydata_profiling=_is_available("ydata_profiling"),
        feature_engine=_is_available("feature_engine"),
        pyod=_is_available("pyod"),
        imblearn=_is_available("imblearn"),
        category_encoders=_is_available("category_encoders"),
        shap=_is_available("shap"),
        cleanlab=_is_available("cleanlab"),
        evidently=_is_available("evidently"),
    )