import logging
import os
from typing import Optional


def _parse_log_level(level_text: Optional[str]) -> int:
    if not level_text:
        return logging.INFO
    txt = str(level_text).strip().upper()
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }.get(txt, logging.INFO)


def configure_logging(app_name: str = "ml_trainer", level: Optional[str] = None) -> None:
    """Configure process-wide logging once.

    Environment overrides:
      - LOG_LEVEL: DEBUG/INFO/WARNING/ERROR/CRITICAL
      - LOG_FILE: optional file path for duplicate log output
    """
    root = logging.getLogger()
    if root.handlers:
        return

    effective_level = _parse_log_level(level or os.environ.get("LOG_LEVEL"))

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.environ.get("LOG_FILE", "").strip()
    if log_file:
        try:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError:
            # Fall back to console-only logging.
            pass

    logging.basicConfig(level=effective_level, format=fmt, datefmt=datefmt, handlers=handlers)
    logging.getLogger(app_name).debug("Logging initialized with level %s", logging.getLevelName(effective_level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
