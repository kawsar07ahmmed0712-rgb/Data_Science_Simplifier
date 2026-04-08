from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.exceptions import FileIngestionError

DEFAULT_DELIMITERS = [",", ";", "\t", "|"]


def _read_text_sample(
    source: Any,
    encoding: str,
    sample_size: int = 65536,
) -> str:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileIngestionError(f"File not found: {path}")
        return path.read_text(encoding=encoding, errors="ignore")[:sample_size]

    if hasattr(source, "getvalue"):
        data = source.getvalue()
        if isinstance(data, bytes):
            return data.decode(encoding, errors="ignore")[:sample_size]
        if isinstance(data, str):
            return data[:sample_size]

    if hasattr(source, "read"):
        current_position = None
        try:
            if hasattr(source, "tell"):
                current_position = source.tell()
        except Exception:
            current_position = None

        raw = source.read(sample_size)
        if current_position is not None and hasattr(source, "seek"):
            try:
                source.seek(current_position)
            except Exception:
                pass

        if isinstance(raw, bytes):
            return raw.decode(encoding, errors="ignore")
        if isinstance(raw, str):
            return raw

    raise FileIngestionError("Could not read text sample from source for delimiter detection.")


def detect_delimiter(
    source: Any,
    encoding: str,
    candidate_delimiters: list[str] | None = None,
    sample_size: int = 65536,
) -> str:
    delimiters = candidate_delimiters or DEFAULT_DELIMITERS
    sample = _read_text_sample(source=source, encoding=encoding, sample_size=sample_size)

    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(delimiters))
        return dialect.delimiter
    except csv.Error:
        pass

    delimiter_counts = {delimiter: sample.count(delimiter) for delimiter in delimiters}
    best_delimiter = max(delimiter_counts, key=delimiter_counts.get)

    if delimiter_counts[best_delimiter] == 0:
        return ","
    return best_delimiter