from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppSettings:
    app_name: str
    app_env: str
    debug: bool
    ollama_host: str
    ollama_model: str
    random_seed: int
    default_test_size: float
    default_val_size: float
    max_rows_preview: int
    max_categories_preview: int
    save_intermediate_artifacts: bool
    strict_mode: bool


def get_settings() -> AppSettings:
    return AppSettings(
        app_name=os.getenv("APP_NAME", "Agentic CSV Data Scientist"),
        app_env=os.getenv("APP_ENV", "development"),
        debug=_get_bool("DEBUG", True),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        default_test_size=float(os.getenv("DEFAULT_TEST_SIZE", "0.2")),
        default_val_size=float(os.getenv("DEFAULT_VAL_SIZE", "0.1")),
        max_rows_preview=int(os.getenv("MAX_ROWS_PREVIEW", "20")),
        max_categories_preview=int(os.getenv("MAX_CATEGORIES_PREVIEW", "25")),
        save_intermediate_artifacts=_get_bool("SAVE_INTERMEDIATE_ARTIFACTS", True),
        strict_mode=_get_bool("STRICT_MODE", True),
    )