from __future__ import annotations

from pathlib import Path
from typing import Any

from core.exceptions import FileIngestionError

DEFAULT_ENCODING_CANDIDATES = (
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin-1",
)


def _read_bytes_from_source(source: Any, sample_size: int = 65536) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileIngestionError(f"File not found: {path}")
        return path.read_bytes()[:sample_size]

    if hasattr(source, "getvalue"):
        data = source.getvalue()
        if isinstance(data, bytes):
            return data[:sample_size]
        if isinstance(data, str):
            return data.encode("utf-8", errors="ignore")[:sample_size]

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
            return raw
        if isinstance(raw, str):
            return raw.encode("utf-8", errors="ignore")

    raise FileIngestionError("Could not read bytes from source for encoding detection.")


def detect_encoding(
    source: Any,
    candidates: tuple[str, ...] = DEFAULT_ENCODING_CANDIDATES,
    sample_size: int = 65536,
) -> str:
    raw = _read_bytes_from_source(source=source, sample_size=sample_size)
    if not raw:
        return "utf-8"

    for encoding in candidates:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    return "latin-1"