class AppError(Exception):
    """Base class for application-level exceptions."""


class DataLoadError(AppError):
    """Raised when data loading/parsing fails."""


class ConfigError(AppError):
    """Raised when configuration is invalid."""

