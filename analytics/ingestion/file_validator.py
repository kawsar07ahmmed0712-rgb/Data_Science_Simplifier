from __future__ import annotations

from pathlib import Path
from typing import Any

from core.exceptions import FileIngestionError

_ALLOWED_SUFFIXES = {".csv", ".txt", ".tsv"}
_DEFAULT_MAX_FILE_SIZE_MB = 512


def validate_file_extension(file_name: str) -> None:
    suffix = Path(file_name).suffix.lower()
    if suffix and suffix not in _ALLOWED_SUFFIXES:
        raise FileIngestionError(
            f"Unsupported file extension '{suffix}'. Allowed: {sorted(_ALLOWED_SUFFIXES)}"
        )


def validate_file_size_bytes(
    file_size_bytes: int,
    max_size_mb: int = _DEFAULT_MAX_FILE_SIZE_MB,
) -> None:
    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise FileIngestionError(
            f"File size {file_size_bytes} bytes exceeds limit of {max_bytes} bytes."
        )


def extract_source_name(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    if hasattr(source, "name") and getattr(source, "name"):
        return str(getattr(source, "name"))
    return "uploaded.csv"


def validate_source(source: Any) -> None:
    if source is None:
        raise FileIngestionError("No file source was provided.")

    file_name = extract_source_name(source)
    validate_file_extension(file_name)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileIngestionError(f"File not found: {path}")
        if not path.is_file():
            raise FileIngestionError(f"Path is not a file: {path}")