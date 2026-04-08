from __future__ import annotations

from enum import Enum


class ProblemType(str, Enum):
    UNKNOWN = "unknown"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    UNSUPERVISED = "unsupervised"


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    ID = "id"
    DATETIME = "datetime"
    TEXT = "text"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    CONSTANT = "constant"
    LEAKAGE_RISK = "leakage_risk"
    UNKNOWN = "unknown"


class RunStage(str, Enum):
    INITIALIZED = "initialized"
    INGESTION = "ingestion"
    SCHEMA = "schema"
    PROFILING = "profiling"
    PLANNING = "planning"
    CLEANING = "cleaning"
    EDA = "eda"
    OUTLIERS = "outliers"
    ANOMALIES = "anomalies"
    FEATURE_ENGINEERING = "feature_engineering"
    SPLITTING = "splitting"
    MODELING = "modeling"
    EVALUATION = "evaluation"
    EXPLAINABILITY = "explainability"
    INSIGHTS = "insights"
    CRITIQUE = "critique"
    REPORTING = "reporting"
    ARTIFACTS = "artifacts"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class SeverityLevel(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SplitStrategy(str, Enum):
    STANDARD = "standard"
    STRATIFIED = "stratified"
    TIME_BASED = "time_based"
    NONE = "none"


class OutlierAction(str, Enum):
    REPORT_ONLY = "report_only"
    CAP = "cap"
    WINSORIZE = "winsorize"
    REMOVE = "remove"
    FLAG = "flag"
    IGNORE = "ignore"


class ModelFamily(str, Enum):
    BASELINE = "baseline"
    LINEAR = "linear"
    TREE = "tree"
    ENSEMBLE = "ensemble"
    DISTANCE = "distance"
    ANOMALY = "anomaly"


class AgentName(str, Enum):
    PLANNER = "planner"
    INSIGHT = "insight"
    CRITIC = "critic"
    RECOMMENDATION = "recommendation"
    EXPLANATION = "explanation"


class ArtifactType(str, Enum):
    REPORT = "report"
    CHART = "chart"
    DATASET = "dataset"
    MODEL = "model"
    PIPELINE = "pipeline"
    METADATA = "metadata"
