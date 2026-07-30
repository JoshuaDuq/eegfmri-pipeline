"""Study 1 feature-family contract."""

from __future__ import annotations

from typing import Any

from eeg_pipeline.utils.config.loader import get_config_value

PRIMARY_FEATURE_FAMILY = "power"
SUPPORTED_EXPLORATORY_FEATURE_FAMILIES = (
    "spectral",
    "aperiodic",
    "erds",
    "ratios",
    "asymmetry",
    "complexity",
    "bursts",
)
DEFAULT_EXPLORATORY_FEATURE_FAMILIES = SUPPORTED_EXPLORATORY_FEATURE_FAMILIES


def _coerce_family_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "study1.features.exploratory_feature_families must be a list of supported family names."
        )

    resolved: list[str] = []
    seen: set[str] = set()
    for raw_family in value:
        family = str(raw_family).strip().lower()
        if not family:
            raise ValueError(
                "study1.features.exploratory_feature_families contains an empty family name."
            )
        if family in seen:
            continue
        seen.add(family)
        resolved.append(family)
    return resolved


def resolve_exploratory_feature_families(config: Any) -> list[str]:
    configured = get_config_value(
        config,
        "study1.features.exploratory_feature_families",
        list(DEFAULT_EXPLORATORY_FEATURE_FAMILIES),
    )
    resolved = _coerce_family_list(configured)
    unsupported = [
        family for family in resolved if family not in SUPPORTED_EXPLORATORY_FEATURE_FAMILIES
    ]
    if unsupported:
        raise ValueError(
            "Unsupported Study 1 exploratory feature families: "
            f"{unsupported}. Supported families are {list(SUPPORTED_EXPLORATORY_FEATURE_FAMILIES)}."
        )
    return resolved


def resolve_study1_feature_families(config: Any) -> list[str]:
    return [PRIMARY_FEATURE_FAMILY, *resolve_exploratory_feature_families(config)]


__all__ = [
    "DEFAULT_EXPLORATORY_FEATURE_FAMILIES",
    "PRIMARY_FEATURE_FAMILY",
    "SUPPORTED_EXPLORATORY_FEATURE_FAMILIES",
    "resolve_exploratory_feature_families",
    "resolve_study1_feature_families",
]
