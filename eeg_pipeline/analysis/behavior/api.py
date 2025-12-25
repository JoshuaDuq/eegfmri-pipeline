"""
Canonical behavior analysis API consolidating stats entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eeg_pipeline.analysis.behavior.feature_correlator import (
    CorrelationConfig,
    FeatureBehaviorCorrelator,
    FeatureCorrelationResult,
    correlate_pain_relevant_features,
    run_unified_feature_correlations,
)
from eeg_pipeline.domain.features.registry import (
    FeatureRegistry,
    FeatureRule,
    classify_feature,
    get_feature_registry,
)
from eeg_pipeline.utils.analysis.stats.correlation import (
    CorrelationResult,
    compute_change_features,
    compute_pain_sensitivity_index,
    interpret_correlation,
    interpret_effect_size,
    run_pain_sensitivity_correlations,
)
from eeg_pipeline.utils.analysis.stats.effect_size import (
    compute_condition_effects,
    split_by_condition,
)
from eeg_pipeline.utils.analysis.stats.mixed_effects import (
    MixedEffectsResult,
    compute_icc,
    fit_mixed_effects_model,
    run_mediation_analysis,
    run_multilevel_correlation_analysis,
)
from eeg_pipeline.utils.analysis.stats.reliability import (
    compute_correlation_split_half_reliability as compute_split_half_reliability,
)
from eeg_pipeline.utils.analysis.stats.temporal import (
    compute_time_frequency_correlations,
    compute_temporal_correlations_by_condition,
    compute_time_frequency_from_context,
    compute_temporal_from_context,
)
from eeg_pipeline.utils.analysis.stats.cluster import (
    compute_pain_nonpain_time_cluster_test,
    _run_cluster_test_core,
)
from eeg_pipeline.utils.analysis.stats.topomaps import run_power_topomap_correlations
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.infra.paths import deriv_stats_path

if TYPE_CHECKING:
    from pathlib import Path
    import logging
    from eeg_pipeline.context.behavior import BehaviorContext


def run_cluster_test_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Run pain vs non-pain cluster test using pre-loaded context data."""
    return _run_cluster_test_core(
        ctx.subject,
        ctx.epochs,
        ctx.aligned_events,
        ctx.stats_dir,
        ctx.config,
        ctx.logger,
        ctx.n_perm,
    )


def run_pain_nonpain_cluster_test(
    subject: str,
    task: str,
    deriv_root: "Path",
    config,
    logger: "logging.Logger",
) -> None:
    """
    Load epochs/events and run the pain vs non-pain cluster test.
    """
    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=True,
        deriv_root=deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )
    default_n_perm = int(
        get_config_value(
            config,
            "behavior_analysis.cluster_correction.default_n_permutations",
            get_config_value(config, "behavior_analysis.statistics.default_n_permutations", 100),
        )
    )
    _run_cluster_test_core(
        subject,
        epochs,
        aligned_events,
        deriv_stats_path(deriv_root, subject),
        config,
        logger,
        n_perm=default_n_perm,
    )


__all__ = [
    # Correlation helpers
    "compute_pain_sensitivity_index",
    "compute_change_features",
    "compute_split_half_reliability",
    "interpret_effect_size",
    "interpret_correlation",
    "CorrelationResult",
    # Feature correlator
    "FeatureBehaviorCorrelator",
    "CorrelationConfig",
    "FeatureCorrelationResult",
    "correlate_pain_relevant_features",
    "run_unified_feature_correlations",
    # Feature registry (backward-compatible re-exports)
    "FeatureRegistry",
    "FeatureRule",
    # Cluster tests
    "compute_pain_nonpain_time_cluster_test",
    "run_cluster_test_from_context",
    "run_pain_nonpain_cluster_test",
    # Mixed-effects
    "MixedEffectsResult",
    "fit_mixed_effects_model",
    "compute_icc",
    "run_multilevel_correlation_analysis",
    "run_mediation_analysis",
    # Condition
    "split_by_condition",
    "compute_condition_effects",
    # Temporal
    "compute_time_frequency_from_context",
    "compute_temporal_from_context",
    "compute_time_frequency_correlations",
    "compute_temporal_correlations_by_condition",
    # Topomaps
    "run_power_topomap_correlations",
]







