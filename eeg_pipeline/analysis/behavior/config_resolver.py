"""Behavior config resolvers with canonical keys only."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from eeg_pipeline.utils.analysis.stats.correlation import normalize_correlation_method
from eeg_pipeline.utils.config.loader import get_config_value


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    out: List[str] = []
    for item in items:
        s = str(item).strip().lower()
        if s:
            out.append(s)
    return out


def _warn(logger: Any, message: str, *args: Any) -> None:
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(message, *args)


def resolve_correlation_method(
    config: Any,
    *,
    logger: Any = None,
    default: str = "spearman",
) -> str:
    """Resolve correlation method from canonical key only."""
    canonical_raw = get_config_value(
        config, "behavior_analysis.statistics.correlation_method", None
    )

    canonical = (
        normalize_correlation_method(canonical_raw, default=default)
        if canonical_raw is not None
        else None
    )

    if canonical is not None:
        return canonical

    return normalize_correlation_method(default, default=default)


def resolve_correlation_targets(
    config: Any,
    *,
    logger: Any = None,
    default_targets: Optional[List[str]] = None,
) -> List[str]:
    """Resolve correlation targets from defaults only."""
    _ = config
    _ = logger
    return list(default_targets or [])
