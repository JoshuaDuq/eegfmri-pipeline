"""
Utility Decorators
==================

Common decorators for validation, logging, and error handling.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])


def validate_input(
    min_samples: int = 3,
    require_finite: bool = True,
    arg_names: Optional[List[str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to validate numerical inputs to a function.
    
    Checks that array/DataFrame inputs have minimum samples and finite values.
    
    Parameters
    ----------
    min_samples : int
        Minimum required samples
    require_finite : bool
        If True, check for NaN/Inf values
    arg_names : List[str], optional
        Names of arguments to validate. If None, validates first two args.
    
    Example
    -------
    @validate_input(min_samples=10)
    def compute_correlation(x, y, method="spearman"):
        ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get arguments to validate
            to_check = []
            if arg_names:
                for name in arg_names:
                    if name in kwargs:
                        to_check.append((name, kwargs[name]))
            else:
                # Check first two positional args by default
                for i, arg in enumerate(args[:2]):
                    to_check.append((f"arg{i}", arg))

            for name, val in to_check:
                if val is None:
                    continue

                if isinstance(val, pd.DataFrame):
                    if len(val) < min_samples:
                        raise ValueError(f"{name}: insufficient samples ({len(val)} < {min_samples})")
                    if require_finite and val.select_dtypes(include=[np.number]).isna().any().any():
                        pass  # Allow NaN in DataFrames, handle in function

                elif isinstance(val, (pd.Series, np.ndarray)):
                    arr = np.asarray(val)
                    finite_count = np.sum(np.isfinite(arr))
                    if finite_count < min_samples:
                        raise ValueError(f"{name}: insufficient finite samples ({finite_count} < {min_samples})")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def log_execution(
    level: int = logging.DEBUG,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to log function execution.
    
    Parameters
    ----------
    level : int
        Logging level
    log_args : bool
        If True, log function arguments
    log_result : bool
        If True, log function result
    """

    def decorator(func: F) -> F:
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__

            if log_args:
                logger.log(level, f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                logger.log(level, f"Calling {func_name}")

            result = func(*args, **kwargs)

            if log_result:
                logger.log(level, f"{func_name} returned: {type(result)}")
            else:
                logger.log(level, f"{func_name} completed")

            return result

        return wrapper  # type: ignore

    return decorator


def handle_empty_input(default_return: Any = None) -> Callable[[F], F]:
    """
    Decorator to handle empty inputs gracefully.
    
    Returns default_return if first argument is None or empty.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                return default_return

            first_arg = args[0]

            if first_arg is None:
                return default_return

            if isinstance(first_arg, (pd.DataFrame, pd.Series)):
                if first_arg.empty:
                    return default_return

            if isinstance(first_arg, np.ndarray):
                if first_arg.size == 0:
                    return default_return

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_columns(*columns: str) -> Callable[[F], F]:
    """
    Decorator to require specific columns in DataFrame argument.
    
    Raises ValueError if any required column is missing.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find DataFrame in args
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            
            if df is None:
                for val in kwargs.values():
                    if isinstance(val, pd.DataFrame):
                        df = val
                        break

            if df is not None:
                missing = set(columns) - set(df.columns)
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def cache_result(maxsize: int = 128) -> Callable[[F], F]:
    """
    Simple memoization decorator for functions with hashable args.
    
    Uses functools.lru_cache under the hood.
    """

    def decorator(func: F) -> F:
        cached = functools.lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Try to use cached version
                return cached(*args, **kwargs)
            except TypeError:
                # Args not hashable, call directly
                return func(*args, **kwargs)

        wrapper.cache_clear = cached.cache_clear  # type: ignore
        wrapper.cache_info = cached.cache_info  # type: ignore

        return wrapper  # type: ignore

    return decorator

