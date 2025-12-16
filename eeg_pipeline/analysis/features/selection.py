from __future__ import annotations

import logging
from typing import Any, List, Optional

from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES, PRECOMPUTED_GROUP_CHOICES


def resolve_precomputed_groups(config: Any, override: Optional[List[str]] = None) -> List[str]:
    groups = override if override is not None else config.get("feature_engineering.precomputed_groups")
    if not groups:
        raise ValueError(
            "Precomputed groups not specified; set feature_engineering.precomputed_groups "
            "in eeg_config.yaml or provide via CLI."
        )

    groups = [str(g) for g in groups]
    groups = list(dict.fromkeys(groups))
    if not groups:
        raise ValueError("No precomputed groups specified; provide at least one.")

    unknown = [g for g in groups if g not in PRECOMPUTED_GROUP_CHOICES]
    if unknown:
        logging.getLogger(__name__).warning(
            "Skipping unsupported precomputed groups (not implemented/allowed): %s",
            ", ".join(unknown),
        )

    supported = [g for g in groups if g in PRECOMPUTED_GROUP_CHOICES]
    if not supported:
        raise ValueError(
            "No supported precomputed groups requested after filtering; requested="
            f"{groups}. Supported={PRECOMPUTED_GROUP_CHOICES}"
        )
    return supported


def resolve_feature_categories(config: Any, requested: Optional[List[str]]) -> List[str]:
    if requested is not None:
        categories = list(requested)
    else:
        from_config = config.get("feature_engineering.feature_categories")
        categories = list(from_config) if from_config else list(FEATURE_CATEGORIES)

    categories = list(dict.fromkeys(str(cat) for cat in categories))
    if not categories:
        raise ValueError("No feature categories specified; provide at least one.")

    invalid = [c for c in categories if c not in FEATURE_CATEGORIES]
    if invalid:
        raise ValueError(
            f"Invalid feature categories {invalid}. "
            f"Valid options: {', '.join(FEATURE_CATEGORIES)}"
        )

    return categories
