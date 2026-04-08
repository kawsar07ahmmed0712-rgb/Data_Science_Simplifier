from __future__ import annotations

from pathlib import Path

import joblib


def save_preprocessor(preprocessor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    return output_path