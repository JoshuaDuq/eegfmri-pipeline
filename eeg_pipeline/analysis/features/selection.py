from __future__ import annotations

from typing import Any, List, Optional

from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES


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

