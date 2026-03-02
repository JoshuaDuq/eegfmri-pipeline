"""Source-localization feature path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from eeg_pipeline.utils.config.loader import get_config_value


SOURCE_LOCALIZATION_METHODS: tuple[str, str] = ("lcmv", "eloreta")
_DEFAULT_SOURCE_LOCALIZATION_METHOD = "lcmv"


def normalize_source_localization_method(method: Any) -> str:
    """Normalize and validate source localization method name."""
    normalized = str(method or "").strip().lower()
    if normalized not in SOURCE_LOCALIZATION_METHODS:
        raise ValueError(
            "feature_engineering.sourcelocalization.method must be one of "
            f"{SOURCE_LOCALIZATION_METHODS} (got {method!r})."
        )
    return normalized


def resolve_source_localization_method(
    config: Optional[Any],
    *,
    default: str = _DEFAULT_SOURCE_LOCALIZATION_METHOD,
) -> str:
    """Resolve source-localization method from config."""
    method_raw = get_config_value(
        config,
        "feature_engineering.sourcelocalization.method",
        default,
    )
    return normalize_source_localization_method(method_raw)


def resolve_source_localization_method_from_attrs(
    attrs: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Resolve source-localization method from DataFrame attrs if present."""
    if attrs is None:
        return None
    method_raw = attrs.get("method")
    if method_raw is None:
        return None
    return normalize_source_localization_method(method_raw)


def source_localization_folder(method: str) -> str:
    """Return source-localization subfolder for a given method."""
    return f"sourcelocalization/{normalize_source_localization_method(method)}"


def source_localization_candidate_paths(
    *,
    features_dir: Path,
    filename: str,
    config: Optional[Any] = None,
) -> list[Path]:
    """Return candidate source-localization feature paths."""
    if config is not None:
        method = resolve_source_localization_method(config)
        return [features_dir / source_localization_folder(method) / filename]

    return [
        features_dir / source_localization_folder(method) / filename
        for method in SOURCE_LOCALIZATION_METHODS
    ]


def source_localization_estimates_dir(
    *,
    features_dir: Path,
    method: str,
) -> Path:
    """Return directory for source-estimate STC artifacts."""
    return features_dir / source_localization_folder(method) / "source_estimates"
