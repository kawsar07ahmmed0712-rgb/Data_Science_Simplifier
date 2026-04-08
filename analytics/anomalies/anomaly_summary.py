from __future__ import annotations


def summarize_anomaly_results(registry: dict[str, object]) -> dict[str, object]:
    per_method = registry.get("per_method", {})
    sorted_methods = sorted(
        [
            {
                "method": method,
                "flagged_count": stats["flagged_count"],
                "flagged_percentage": stats["flagged_percentage"],
            }
            for method, stats in per_method.items()
        ],
        key=lambda item: item["flagged_percentage"],
        reverse=True,
    )

    return {
        "combined_flagged_rows": registry.get("combined_flagged_rows", 0),
        "combined_flagged_percentage": registry.get("combined_flagged_percentage", 0.0),
        "methods_ranked": sorted_methods,
    }