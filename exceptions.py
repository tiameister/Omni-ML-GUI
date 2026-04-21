class AppError(Exception):
    """Base class for all Omni-ML-GUI application-level exceptions.

    Catch ``AppError`` to handle any recoverable application error in a
    single except clause without swallowing unrelated errors:

        try:
            ...
        except AppError as exc:
            LOGGER.error("Application error: %s", exc)
    """


class DataLoadError(AppError):
    """Raised when a dataset cannot be loaded or parsed.

    Callers: ``data.loader`` — raised on unsupported file formats,
    encoding failures, or empty-file conditions.
    """


class ConfigError(AppError):
    """Raised when a configuration value is missing or invalid.

    Callers: ``config.__init__`` — raised on bad env-var types,
    out-of-range numeric options, or unrecognised enum values.
    """


class DataValidationError(AppError):
    """Raised when the training input fails the strict pre-flight check.

    Callers: ``core.data_validation`` — surfaced before the training
    worker thread is spawned so the user gets a friendly GUI warning
    instead of a cryptic stack trace from deep inside scikit-learn.
    """

