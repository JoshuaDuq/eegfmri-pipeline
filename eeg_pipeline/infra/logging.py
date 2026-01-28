"""Logging utilities for the EEG pipeline."""

from __future__ import annotations

import logging
from typing import Optional

_configured_loggers: set[str] = set()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOGGER_NAME = "eeg_pipeline"
DEFAULT_LOG_LEVEL = logging.INFO


def _create_log_formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)


def _add_console_handler(logger: logging.Logger, formatter: logging.Formatter) -> None:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _is_logger_configured(logger_name: str) -> bool:
    return logger_name in _configured_loggers


def _mark_logger_configured(logger_name: str) -> None:
    _configured_loggers.add(logger_name)


def _configure_logger_handlers(logger: logging.Logger) -> None:
    if _is_logger_configured(logger.name):
        return

    logger.handlers.clear()
    logger.propagate = False

    formatter = _create_log_formatter()
    _add_console_handler(logger, formatter)

    logger.setLevel(DEFAULT_LOG_LEVEL)
    _mark_logger_configured(logger.name)


def get_logger(name: str) -> logging.Logger:
    if not name or not isinstance(name, str):
        raise ValueError("Logger name must be a non-empty string")
    
    logger = logging.getLogger(name)
    _configure_logger_handlers(logger)
    return logger


def get_module_logger(logger: Optional[logging.Logger] = None, module_name: Optional[str] = None) -> logging.Logger:
    if logger is not None:
        return logger
    return get_logger(module_name or __name__)


def get_subject_logger(
    module_name: str,
    subject: str,
) -> logging.Logger:
    if not module_name or not isinstance(module_name, str):
        raise ValueError("Module name must be a non-empty string")
    if not subject or not isinstance(subject, str):
        raise ValueError("Subject must be a non-empty string")
    
    logger_name = f"{module_name}.sub-{subject}"
    return get_logger(logger_name)


def get_default_logger() -> logging.Logger:
    return get_logger(DEFAULT_LOGGER_NAME)


__all__ = [
    "get_logger",
    "get_module_logger",
    "get_subject_logger",
    "get_default_logger",
]
