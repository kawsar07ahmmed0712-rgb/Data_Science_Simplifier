from __future__ import annotations

import pandas as pd


def build_anomaly_registry(
    *,
    results: list[dict[str, object]],
    combined_mask: pd.Series,
) -> dict[str, object]:
    methods = []
    per_method = {}

    for item in results:
        method = str(item["method"])
        methods.append(method)
        per_method[method] = {
            "flagged_count": int(item["flagged_count"]),
            "flagged_percentage": float(item["flagged_percentage"]),
        }

    return {
        "methods": methods,
        "per_method": per_method,
        "combined_flagged_rows": int(combined_mask.sum()),
        "combined_flagged_percentage": round(
            (int(combined_mask.sum()) / max(len(combined_mask), 1)) * 100.0,
            4,
        ),
    }