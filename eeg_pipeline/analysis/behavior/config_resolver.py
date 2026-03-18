"""Behavior config resolvers with canonical keys only."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from eeg_pipeline.utils.analysis.stats.correlation import normalize_correlation_method
from eeg_pipeline.utils.config.behavior_loader import ensure_behavior_config
from eeg_pipeline.utils.config.loader import get_config_value, require_config_value


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


def resolve_correlation_method(
    config: Any,
    *,
    logger: Any = None,
) -> str:
    """Resolve correlation method from canonical behavior statistics config."""
    _ = logger
    config = ensure_behavior_config(config)
    canonical_raw = require_config_value(
        config, "behavior_analysis.statistics.correlation_method"
    )
    canonical = normalize_correlation_method(canonical_raw, default="")
    if not canonical:
        raise ValueError(
            "Unsupported behavior_analysis.statistics.correlation_method "
            f"value: {canonical_raw!r}."
        )
    return canonical


def resolve_correlation_targets(
    config: Any,
    *,
    logger: Any = None,
    default_targets: Optional[List[str]] = None,
) -> List[str]:
    """Resolve correlation targets from canonical behavior correlations config."""
    _ = logger
    config = ensure_behavior_config(config)
    canonical = _normalize_string_list(
        get_config_value(config, "behavior_analysis.correlations.targets", None)
    )
    if canonical:
        return canonical

    return list(default_targets or [])
