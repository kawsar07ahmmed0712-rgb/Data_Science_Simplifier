from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model_object(model: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path
