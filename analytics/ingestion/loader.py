from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.ingestion.delimiter_detector import detect_delimiter
from analytics.ingestion.encoding_detector import detect_encoding
from analytics.ingestion.file_validator import (
    extract_source_name,
    validate_file_size_bytes,
    validate_source,
)
from analytics.ingestion.header_normalizer import normalize_headers
from core.contracts import FileMeta
from core.exceptions import FileIngestionError


def _source_to_pandas_input(source: Any, encoding: str) -> Any:
    if isinstance(source, (str, Path)):
        return str(source)

    if hasattr(source, "getvalue"):
        data = source.getvalue()
        if isinstance(data, bytes):
            return BytesIO(data)
        if isinstance(data, str):
            return StringIO(data)

    if hasattr(source, "read"):
        current_position = None
        try:
            if hasattr(source, "tell"):
                current_position = source.tell()
        except Exception:
            current_position = None

        raw = source.read()
        if current_position is not None and hasattr(source, "seek"):
            try:
                source.seek(current_position)
            except Exception:
                pass

        if isinstance(raw, bytes):
            return BytesIO(raw)
        if isinstance(raw, str):
            return StringIO(raw)

    raise FileIngestionError("Unsupported input source for pandas loading.")


def _estimate_file_size_bytes(source: Any) -> int:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.stat().st_size if path.exists() else 0

    if hasattr(source, "getvalue"):
        data = source.getvalue()
        if isinstance(data, bytes):
            return len(data)
        if isinstance(data, str):
            return len(data.encode("utf-8"))

    return 0


def load_csv(
    source: Any,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    normalize_column_headers: bool = True,
    nrows: int | None = None,
    max_file_size_mb: int = 512,
) -> tuple[pd.DataFrame, FileMeta]:
    validate_source(source)

    file_name = extract_source_name(source)
    file_size_bytes = _estimate_file_size_bytes(source)
    if file_size_bytes > 0:
        validate_file_size_bytes(file_size_bytes=file_size_bytes, max_size_mb=max_file_size_mb)

    detected_encoding = encoding or detect_encoding(source)
    detected_delimiter = delimiter or detect_delimiter(source, encoding=detected_encoding)

    pandas_input = _source_to_pandas_input(source, encoding=detected_encoding)

    try:
        df = pd.read_csv(
            pandas_input,
            sep=detected_delimiter,
            encoding=detected_encoding,
            low_memory=False,
            nrows=nrows,
        )
    except Exception as exc:
        raise FileIngestionError(
            f"Failed to load CSV '{file_name}' with encoding={detected_encoding}, "
            f"delimiter='{detected_delimiter}'. Root cause: {exc}"
        ) from exc

    if df.empty and len(df.columns) == 0:
        raise FileIngestionError(f"The file '{file_name}' appears to be empty or unreadable.")

    if normalize_column_headers:
        df.columns = normalize_headers(df.columns)

    file_meta = FileMeta(
        file_name=file_name,
        file_path=str(source) if isinstance(source, (str, Path)) else "",
        file_size_bytes=file_size_bytes,
        delimiter=detected_delimiter,
        encoding=detected_encoding,
        row_count_estimate=int(df.shape[0]),
        column_count_estimate=int(df.shape[1]),
    )

    return df, file_meta