#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

LOG_LEVEL_MAP: Dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_CONFIGURED_LOGGERS: Dict[str, Tuple[logging.Logger, Path]] = {}


def _level_from_string(level_name: str) -> int:
    return LOG_LEVEL_MAP.get(level_name.upper(), logging.INFO)


def _resolve_log_dir(log_dir: Optional[str] = None) -> Path:
    if log_dir:
        return Path(log_dir).expanduser()
    env_dir = os.getenv("NPS_LOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.cwd() / "logs"


def configure_logging(
    name: str,
    log_dir: Optional[str] = None,
    console_level: Optional[str] = None,
) -> Tuple[logging.Logger, Path]:
    cache_key = name
    if cache_key in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[cache_key]

    log_directory = _resolve_log_dir(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_directory / f"{name}_{timestamp}.log"

    logger_name = f"NPS.{name}"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    resolved_console_level = os.getenv("NPS_CONSOLE_LOG_LEVEL", console_level or "INFO")
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_level_from_string(resolved_console_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED_LOGGERS[cache_key] = (logger, log_path)
    return logger, log_path


def get_log_function(
    name: str,
    log_dir: Optional[str] = None,
    console_level: Optional[str] = None,
) -> Tuple[Callable[[str, str], None], Path]:
    logger, log_path = configure_logging(name, log_dir=log_dir, console_level=console_level)

    def _log(message: str, level: str = "INFO") -> None:
        logger.log(_level_from_string(level), message)

    return _log, log_path

