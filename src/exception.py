"""Project-specific exception types."""


class ProjectError(Exception):
    """Base exception for expected project failures."""


class DataIngestionError(ProjectError):
    """Raised when project data cannot be loaded."""


class DataTransformationError(ProjectError):
    """Raised when project data cannot be transformed."""


class ModelTrainingError(ProjectError):
    """Raised when model training or evaluation fails."""


class PredictionError(ProjectError):
    """Raised when a prediction request cannot be processed."""
