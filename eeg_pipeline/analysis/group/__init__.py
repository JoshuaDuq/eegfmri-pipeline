"""
Group-level analysis modules.

Submodules:
- features: Feature aggregation across subjects
- behavior: Behavioral correlation aggregation
- statistics: Group statistical utilities
"""

from eeg_pipeline.analysis.group.features import aggregate_feature_stats
from eeg_pipeline.analysis.group.behavior import aggregate_behavior_correlations
from eeg_pipeline.analysis.group.statistics import (
    parse_pow_column,
    extract_bin_token,
    validate_subjects_for_group_analysis,
)

__all__ = [
    "aggregate_feature_stats",
    "aggregate_behavior_correlations",
    "parse_pow_column",
    "extract_bin_token",
    "validate_subjects_for_group_analysis",
]

