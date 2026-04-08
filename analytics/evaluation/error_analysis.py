from __future__ import annotations

import pandas as pd


def analyze_classification_errors(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    max_examples: int = 20,
) -> dict[str, object]:
    comparison = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    comparison["is_error"] = comparison["y_true"].astype(str) != comparison["y_pred"].astype(str)

    errors = comparison[comparison["is_error"]].copy()
    error_pairs = (
        errors.groupby(["y_true", "y_pred"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top_error_pairs = [
        {
            "y_true": str(row["y_true"]),
            "y_pred": str(row["y_pred"]),
            "count": int(row["count"]),
        }
        for _, row in error_pairs.head(10).iterrows()
    ]

    error_examples = [
        {
            "index": str(idx),
            "y_true": str(row["y_true"]),
            "y_pred": str(row["y_pred"]),
        }
        for idx, row in errors.head(max_examples).iterrows()
    ]

    return {
        "error_count": int(errors.shape[0]),
        "error_rate": round(errors.shape[0] / max(comparison.shape[0], 1), 6),
        "top_error_pairs": top_error_pairs,
        "error_examples": error_examples,
    }


def analyze_regression_errors(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    max_examples: int = 20,
) -> dict[str, object]:
    comparison = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    comparison["residual"] = comparison["y_true"] - comparison["y_pred"]
    comparison["abs_error"] = comparison["residual"].abs()

    worst_cases = comparison.sort_values("abs_error", ascending=False).head(max_examples)

    return {
        "mean_residual": round(float(comparison["residual"].mean()), 6),
        "median_abs_error": round(float(comparison["abs_error"].median()), 6),
        "worst_cases": [
            {
                "index": str(idx),
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "residual": float(row["residual"]),
                "abs_error": float(row["abs_error"]),
            }
            for idx, row in worst_cases.iterrows()
        ],
    }