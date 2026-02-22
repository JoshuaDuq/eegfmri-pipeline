"""Shared utilities and constants for CLI commands."""

from eeg_pipeline.cli.commands.base_discovery import (
    discover_condition_effects_columns,
    discover_event_columns,
    discover_trial_table_columns,
)
from eeg_pipeline.cli.commands.base_feature_availability import (
    BEHAVIOR_COMPUTATIONS,
    FREQUENCY_BANDS,
    _empty_feature_availability,
    detect_available_bands,
    detect_feature_availability,
)
from eeg_pipeline.pipelines.constants import (
    BEHAVIOR_VISUALIZE_CATEGORIES,
    FEATURE_VISUALIZE_CATEGORIES,
)

__all__ = [
    "detect_available_bands",
    "detect_feature_availability",
    "_empty_feature_availability",
    "discover_event_columns",
    "discover_trial_table_columns",
    "discover_condition_effects_columns",
    "BEHAVIOR_COMPUTATIONS",
    "FEATURE_VISUALIZE_CATEGORIES",
    "BEHAVIOR_VISUALIZE_CATEGORIES",
    "FREQUENCY_BANDS",
]
