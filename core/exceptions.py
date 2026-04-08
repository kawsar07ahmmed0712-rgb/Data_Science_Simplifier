from __future__ import annotations


class AgenticCSVError(Exception):
    """Base exception for the project."""


class ConfigurationError(AgenticCSVError):
    """Raised when config is missing or invalid."""


class FileIngestionError(AgenticCSVError):
    """Raised when CSV/file loading fails."""


class SchemaDetectionError(AgenticCSVError):
    """Raised when schema detection fails."""


class ProfilingError(AgenticCSVError):
    """Raised when profiling fails."""


class CleaningError(AgenticCSVError):
    """Raised when cleaning/transformation fails."""


class EDAError(AgenticCSVError):
    """Raised when EDA execution fails."""


class OutlierDetectionError(AgenticCSVError):
    """Raised when outlier detection fails."""


class FeatureEngineeringError(AgenticCSVError):
    """Raised when feature engineering fails."""


class SplitError(AgenticCSVError):
    """Raised when dataset split fails."""


class ModelingError(AgenticCSVError):
    """Raised when training/prediction fails."""


class EvaluationError(AgenticCSVError):
    """Raised when evaluation fails."""


class ExplainabilityError(AgenticCSVError):
    """Raised when explainability generation fails."""


class AgentExecutionError(AgenticCSVError):
    """Raised when Ollama/agent response handling fails."""


class ReportingError(AgenticCSVError):
    """Raised when report generation fails."""


class ArtifactExportError(AgenticCSVError):
    """Raised when artifact export fails."""


class ValidationError(AgenticCSVError):
    """Raised when state or contract validation fails."""