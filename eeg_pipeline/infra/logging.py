"""Logging utilities for the EEG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from eeg_pipeline.utils.config.loader import ConfigDict

from .paths import ensure_dir

EEGConfig = ConfigDict

_configured_loggers: set[str] = set()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_EXTENSION = ".log"
LOGS_DIRECTORY_NAME = "logs"
DEFAULT_LOGGER_NAME = "eeg_pipeline"
DEFAULT_LOG_LEVEL = logging.INFO


def _resolve_log_dir(config: Optional[EEGConfig] = None) -> Optional[Path]:
    if config is None:
        return None

    log_dir = config.get("logging.log_dir")
    if log_dir:
        return Path(log_dir)

    deriv_root = config.get("paths.deriv_root")
    if deriv_root:
        return Path(deriv_root) / LOGS_DIRECTORY_NAME

    return None


def _get_log_file_path(logger_name: str, log_file_name: Optional[str], config: Optional[EEGConfig]) -> Optional[Path]:
    has_file_name = log_file_name is not None
    has_config = config is not None
    
    if not has_file_name and not has_config:
        return None
    
    log_dir = _resolve_log_dir(config)
    if log_dir is None:
        return None
    
    filename = log_file_name if has_file_name else f"{logger_name}{LOG_FILE_EXTENSION}"
    return log_dir / filename


def _create_log_formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)


def _add_console_handler(logger: logging.Logger, formatter: logging.Formatter) -> None:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _add_file_handler(logger: logging.Logger, log_file_path: Path, formatter: logging.Formatter) -> None:
    ensure_dir(log_file_path.parent)
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _is_logger_configured(logger_name: str) -> bool:
    return logger_name in _configured_loggers


def _mark_logger_configured(logger_name: str) -> None:
    _configured_loggers.add(logger_name)


def _configure_logger_handlers(logger: logging.Logger, log_file_path: Optional[Path] = None) -> None:
    if _is_logger_configured(logger.name):
        return

    logger.handlers.clear()
    logger.propagate = False

    formatter = _create_log_formatter()
    _add_console_handler(logger, formatter)

    if log_file_path is not None:
        _add_file_handler(logger, log_file_path, formatter)

    logger.setLevel(DEFAULT_LOG_LEVEL)
    _mark_logger_configured(logger.name)


def get_logger(name: str, log_file_path: Optional[Path] = None) -> logging.Logger:
    if not name or not isinstance(name, str):
        raise ValueError("Logger name must be a non-empty string")
    
    logger = logging.getLogger(name)
    _configure_logger_handlers(logger, log_file_path)
    return logger


def get_module_logger(logger: Optional[logging.Logger] = None, module_name: Optional[str] = None) -> logging.Logger:
    if logger is not None:
        return logger
    return get_logger(module_name or __name__)


def log_and_raise_error(logger: logging.Logger, error_msg: str, exception_class=ValueError) -> None:
    logger.error(error_msg)
    raise exception_class(error_msg)


def reset_logging() -> None:
    global _configured_loggers
    for name in list(_configured_loggers):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    _configured_loggers.clear()


def get_subject_logger(
    module_name: str,
    subject: str,
    log_file_name: Optional[str] = None,
    config: Optional[EEGConfig] = None,
) -> logging.Logger:
    if not module_name or not isinstance(module_name, str):
        raise ValueError("Module name must be a non-empty string")
    if not subject or not isinstance(subject, str):
        raise ValueError("Subject must be a non-empty string")
    
    logger_name = f"{module_name}.sub-{subject}"
    log_file_path = _get_log_file_path(logger_name, log_file_name, config)
    return get_logger(logger_name, log_file_path)


def get_pipeline_logger(module_name: Optional[str] = None, config: Optional[EEGConfig] = None) -> logging.Logger:
    logger_name = module_name or DEFAULT_LOGGER_NAME
    log_file_path = _get_log_file_path(logger_name, None, config) if config else None
    return get_logger(logger_name, log_file_path)


def setup_pipeline_logger(config: Optional[EEGConfig] = None) -> logging.Logger:
    return get_pipeline_logger(config=config)


def setup_subject_logger(subject: str, config: Optional[EEGConfig] = None) -> logging.Logger:
    return get_subject_logger("pipeline", subject, config=config)


def setup_logger(config: Optional[EEGConfig] = None, subject: Optional[str] = None) -> logging.Logger:
    if subject:
        return setup_subject_logger(subject, config=config)
    return setup_pipeline_logger(config=config)


def get_default_logger() -> logging.Logger:
    return logging.getLogger(__name__)


def get_group_logger(
    module_name: str,
    log_file_name: Optional[str] = None,
    config: Optional[EEGConfig] = None,
) -> logging.Logger:
    if not module_name or not isinstance(module_name, str):
        raise ValueError("Module name must be a non-empty string")
    
    logger_name = f"{module_name}.group"
    log_file_path = _get_log_file_path(logger_name, log_file_name, config)
    return get_logger(logger_name, log_file_path)


__all__ = [
    "_resolve_log_dir",
    "_configure_logger_handlers",
    "get_logger",
    "get_module_logger",
    "log_and_raise_error",
    "reset_logging",
    "get_subject_logger",
    "get_group_logger",
    "get_pipeline_logger",
    "setup_logger",
    "setup_pipeline_logger",
    "setup_subject_logger",
    "get_default_logger",
]
