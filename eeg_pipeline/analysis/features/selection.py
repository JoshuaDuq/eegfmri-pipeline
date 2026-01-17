from __future__ import annotations

from typing import Any, List, Optional

from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES


def resolve_feature_categories(config: Any, requested: Optional[List[str]]) -> List[str]:
    """
    Resolve feature categories from user request, configuration, and defaults.

    Precedence:
    - If `requested` is provided, use it directly.
    - Otherwise, read `feature_engineering.feature_categories` from `config`.
    - If neither is available, fall back to all entries in `FEATURE_CATEGORIES`.

    The resulting categories are:
    - Cast to strings
    - De-duplicated while preserving order
    - Validated to be non-empty and all members of `FEATURE_CATEGORIES`

    Raises
    ------
    ValueError
        If no categories are specified or any category is invalid.
    """
    if requested is not None:
        categories = list(requested)
    else:
        configured_categories = config.get("feature_engineering.feature_categories")
        if configured_categories:
            categories = list(configured_categories)
        else:
            categories = list(FEATURE_CATEGORIES)

    categories = list(dict.fromkeys(str(cat) for cat in categories))
    if not categories:
        raise ValueError("No feature categories specified; provide at least one.")

    invalid_categories = [c for c in categories if c not in FEATURE_CATEGORIES]
    if invalid_categories:
        raise ValueError(
            f"Invalid feature categories {invalid_categories}. "
            f"Valid options: {', '.join(FEATURE_CATEGORIES)}"
        )

    return categories

