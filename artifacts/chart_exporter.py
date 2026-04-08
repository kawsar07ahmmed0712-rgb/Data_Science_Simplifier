from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.state import AnalysisState


def save_basic_charts(
    *,
    state: AnalysisState,
    cleaned_df: pd.DataFrame,
    charts_dir: Path,
) -> dict[str, Path]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    numeric_cols = [c for c in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[c])]
    if numeric_cols:
        col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(8, 4))
        cleaned_df[col].dropna().hist(ax=ax)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        path = charts_dir / "numeric_distribution.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved["numeric_distribution"] = path

    if state.plan.target_column and state.plan.target_column in cleaned_df.columns:
        target = state.plan.target_column
        fig, ax = plt.subplots(figsize=(8, 4))
        cleaned_df[target].astype(str).value_counts().head(15).plot(kind="bar", ax=ax)
        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        path = charts_dir / "target_distribution.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved["target_distribution"] = path

    return saved