"""Pipeline-level visualization entrypoints.

Grouped orchestration functions for behavior, decoding, and TFR.
"""

from .behavior import visualize_subject_behavior, visualize_behavior_for_subjects
from .decoding import (
    visualize_regression_results,
    visualize_time_generalization,
    visualize_model_comparisons,
    visualize_riemann_analysis,
    visualize_incremental_validity,
)
from .tfr import visualize_subject_tfr, visualize_tfr_for_subjects
from .features import visualize_features, visualize_features_for_subjects

__all__ = [
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
    "visualize_regression_results",
    "visualize_time_generalization",
    "visualize_model_comparisons",
    "visualize_riemann_analysis",
    "visualize_incremental_validity",
    "visualize_subject_tfr",
    "visualize_tfr_for_subjects",
    "visualize_features",
    "visualize_features_for_subjects",
]
