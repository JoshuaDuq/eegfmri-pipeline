"""
Logging utilities for the EEG pipeline.

This module provides functions for creating and configuring loggers
with consistent formatting and file handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

try:
    from ..config.loader import ConfigDict
except ImportError:
    ConfigDict = dict

EEGConfig = ConfigDict

from .paths import ensure_dir


_configured_loggers: set = set()


def _resolve_log_dir(config: Optional[EEGConfig] = None) -> Optional[Path]:
    if config is None:
        return None
    
    log_dir = config.get("logging.log_dir")
    if log_dir:
        return Path(log_dir)
    
    deriv_root = config.get("paths.deriv_root")
    if deriv_root:
        return Path(deriv_root) / "logs"
    
    return None


def _configure_logger_handlers(
    logger: logging.Logger,
    log_file_path: Optional[Path] = None
) -> None:
    """
    Configure handlers for a logger, preventing duplicates.
    
    This function:
    1. Tracks configured loggers to prevent duplicate handler setup
    2. Disables propagation to prevent duplicate messages from parent loggers
    3. Uses a consistent format across all loggers
    """
    # Skip if already configured
    if logger.name in _configured_loggers:
        return
    
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Disable propagation to parent loggers to prevent duplicate messages
    logger.propagate = False
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file_path is not None:
        ensure_dir(log_file_path.parent)
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(logging.INFO)
    _configured_loggers.add(logger.name)


def get_logger(
    name: str,
    log_file_path: Optional[Path] = None
) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Ensures no duplicate handlers are added and propagation is disabled.
    """
    logger = logging.getLogger(name)
    _configure_logger_handlers(logger, log_file_path)
    return logger


def get_module_logger(logger: Optional[logging.Logger] = None, module_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a module, using provided logger if available.
    
    This is the preferred way to get a logger within functions that accept
    an optional logger parameter.
    """
    if logger is not None:
        return logger
    return get_logger(module_name or __name__)


def log_and_raise_error(logger: logging.Logger, error_msg: str, exception_class=ValueError) -> None:
    logger.error(error_msg)
    raise exception_class(error_msg)


def reset_logging() -> None:
    """
    Reset the logging configuration.
    
    Clears all configured loggers and their handlers. Useful for testing
    or when reconfiguring logging at runtime.
    """
    global _configured_loggers
    for name in list(_configured_loggers):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    _configured_loggers.clear()


def _get_log_file_path(logger_name: str, log_file_name: Optional[str], config: Optional[EEGConfig]) -> Optional[Path]:
    if not log_file_name and not config:
        return None
    log_dir = _resolve_log_dir(config)
    if not log_dir:
        return None
    filename = log_file_name or f"{logger_name}.log"
    return log_dir / filename


def get_subject_logger(
    module_name: str,
    subject: str,
    log_file_name: Optional[str] = None,
    config: Optional[EEGConfig] = None
) -> logging.Logger:
    logger_name = f"{module_name}.sub-{subject}"
    log_file_path = _get_log_file_path(logger_name, log_file_name, config)
    return get_logger(logger_name, log_file_path)


def get_group_logger(
    module_name: str,
    log_file_name: Optional[str] = None,
    config: Optional[EEGConfig] = None
) -> logging.Logger:
    logger_name = f"{module_name}.group"
    log_file_path = _get_log_file_path(logger_name, log_file_name, config)
    return get_logger(logger_name, log_file_path)


def get_pipeline_logger(
    module_name: Optional[str] = None,
    config: Optional[EEGConfig] = None
) -> logging.Logger:
    logger_name = module_name or "eeg_pipeline"
    log_file_path = _get_log_file_path(logger_name, None, config) if config else None
    return get_logger(logger_name, log_file_path)


def setup_logger(
    config: Optional[EEGConfig] = None,
    subject: Optional[str] = None
) -> logging.Logger:
    if subject:
        return get_subject_logger("pipeline", subject, config=config)
    return get_pipeline_logger(config=config)


def get_default_logger() -> logging.Logger:
    return logging.getLogger(__name__)


__all__ = [
    "_resolve_log_dir",
    "_configure_logger_handlers",
    "get_logger",
    "get_module_logger",
    "log_and_raise_error",
    "reset_logging",
    "_get_log_file_path",
    "get_subject_logger",
    "get_group_logger",
    "get_pipeline_logger",
    "setup_logger",
    "get_default_logger",
]










