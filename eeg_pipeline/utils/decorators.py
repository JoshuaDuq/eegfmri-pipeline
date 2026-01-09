"""
Utility Decorators
==================

Common decorators for validation, logging, and error handling.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])


def _find_dataframe_in_args(args: tuple, kwargs: dict) -> Optional[pd.DataFrame]:
    """Find first DataFrame in function arguments."""
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            return arg
    for val in kwargs.values():
        if isinstance(val, pd.DataFrame):
            return val
    return None


def _validate_dataframe(
    df: pd.DataFrame,
    name: str,
    min_samples: int,
    require_finite: bool,
) -> None:
    """Validate DataFrame has sufficient samples."""
    if len(df) < min_samples:
        raise ValueError(
            f"{name}: insufficient samples ({len(df)} < {min_samples})"
        )
    if require_finite:
        numeric_cols = df.select_dtypes(include=[np.number])
        has_nan = numeric_cols.isna().any().any()
        if has_nan:
            raise ValueError(f"{name}: contains NaN values")


def _validate_array(
    arr: np.ndarray,
    name: str,
    min_samples: int,
) -> None:
    """Validate array has sufficient finite samples."""
    finite_count = np.sum(np.isfinite(arr))
    if finite_count < min_samples:
        raise ValueError(
            f"{name}: insufficient finite samples "
            f"({finite_count} < {min_samples})"
        )


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
            arguments_to_validate = []
            if arg_names:
                for name in arg_names:
                    if name in kwargs:
                        arguments_to_validate.append((name, kwargs[name]))
            else:
                for i, arg in enumerate(args[:2]):
                    arguments_to_validate.append((f"arg{i}", arg))

            for name, value in arguments_to_validate:
                if value is None:
                    continue

                if isinstance(value, pd.DataFrame):
                    _validate_dataframe(value, name, min_samples, require_finite)
                elif isinstance(value, (pd.Series, np.ndarray)):
                    array = np.asarray(value)
                    _validate_array(array, name, min_samples)

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
        function_name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if log_args:
                logger.log(
                    level,
                    f"Calling {function_name} with args={args}, kwargs={kwargs}",
                )
            else:
                logger.log(level, f"Calling {function_name}")

            result = func(*args, **kwargs)

            if log_result:
                result_type = type(result).__name__
                logger.log(level, f"{function_name} returned: {result_type}")
            else:
                logger.log(level, f"{function_name} completed")

            return result

        return wrapper  # type: ignore

    return decorator


def _is_empty(value: Any) -> bool:
    """Check if value is None or empty."""
    if value is None:
        return True
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.empty
    if isinstance(value, np.ndarray):
        return value.size == 0
    return False


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

            first_argument = args[0]
            if _is_empty(first_argument):
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
            dataframe = _find_dataframe_in_args(args, kwargs)
            if dataframe is not None:
                required_columns = set(columns)
                existing_columns = set(dataframe.columns)
                missing_columns = required_columns - existing_columns
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def cache_result(maxsize: int = 128) -> Callable[[F], F]:
    """
    Simple memoization decorator for functions with hashable args.
    
    Uses functools.lru_cache under the hood. Falls back to direct call
    if arguments are not hashable.
    """

    def decorator(func: F) -> F:
        cached_func = functools.lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return cached_func(*args, **kwargs)
            except TypeError as error:
                error_message = str(error)
                is_unhashable_error = (
                    "unhashable type" in error_message
                    or "not hashable" in error_message.lower()
                )
                if is_unhashable_error:
                    return func(*args, **kwargs)
                raise

        wrapper.cache_clear = cached_func.cache_clear  # type: ignore
        wrapper.cache_info = cached_func.cache_info  # type: ignore

        return wrapper  # type: ignore

    return decorator

