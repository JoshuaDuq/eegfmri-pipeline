"""Behavior config resolvers with canonical key precedence and legacy fallbacks."""

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
    """Resolve correlation method using canonical key with strict conflict checks."""
    canonical_raw = get_config_value(
        config, "behavior_analysis.statistics.correlation_method", None
    )
    legacy_raw = get_config_value(config, "behavior_analysis.correlation_method", None)

    canonical = (
        normalize_correlation_method(canonical_raw, default=default)
        if canonical_raw is not None
        else None
    )
    legacy = (
        normalize_correlation_method(legacy_raw, default=default)
        if legacy_raw is not None
        else None
    )

    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError(
            "Conflicting correlation method config keys: "
            "behavior_analysis.statistics.correlation_method "
            f"({canonical}) != behavior_analysis.correlation_method ({legacy}). "
            "Use only behavior_analysis.statistics.correlation_method."
        )

    if canonical is not None:
        return canonical

    if legacy is not None:
        _warn(
            logger,
            "behavior_analysis.correlation_method is deprecated; use "
            "behavior_analysis.statistics.correlation_method instead.",
        )
        return legacy

    return normalize_correlation_method(default, default=default)


def resolve_correlation_targets(
    config: Any,
    *,
    logger: Any = None,
    default_targets: Optional[List[str]] = None,
) -> List[str]:
    """Resolve correlation targets from canonical key with legacy fallback."""
    canonical = _normalize_string_list(
        get_config_value(config, "behavior_analysis.correlations.targets", None)
    )
    legacy = _normalize_string_list(get_config_value(config, "behavior_analysis.targets", None))

    if canonical and legacy and canonical != legacy:
        raise ValueError(
            "Conflicting target config keys: behavior_analysis.correlations.targets "
            f"({canonical}) != behavior_analysis.targets ({legacy}). "
            "Use only behavior_analysis.correlations.targets."
        )

    if canonical:
        return canonical

    if legacy:
        _warn(
            logger,
            "behavior_analysis.targets is deprecated; use "
            "behavior_analysis.correlations.targets instead.",
        )
        return legacy

    return list(default_targets or [])
