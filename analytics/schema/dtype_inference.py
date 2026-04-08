from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def infer_dtype_label(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "boolean"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_integer_dtype(series):
        return "integer"
    if is_float_dtype(series):
        return "float"
    if is_numeric_dtype(series):
        return "numeric"
    if is_string_dtype(series):
        return "string"
    if is_object_dtype(series):
        return "object"
    return str(series.dtype)


def is_numeric_like(series: pd.Series) -> bool:
    return is_numeric_dtype(series)


def is_string_like(series: pd.Series) -> bool:
    return is_string_dtype(series) or is_object_dtype(series)