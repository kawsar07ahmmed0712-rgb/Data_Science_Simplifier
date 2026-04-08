from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ReportConfig:
    include_executive_summary: bool = True
    include_data_quality: bool = True
    include_eda: bool = True
    include_outliers: bool = True
    include_feature_engineering: bool = True
    include_modeling: bool = True
    include_explainability: bool = True
    include_critique: bool = True
    include_recommendations: bool = True
    max_top_findings: int = 7
    max_top_features: int = 15
    max_top_outlier_columns: int = 15
    formats: tuple[str, ...] = field(default_factory=lambda: ("markdown", "html"))


def get_report_config() -> ReportConfig:
    return ReportConfig()