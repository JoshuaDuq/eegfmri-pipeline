"""
Canonical behavior analysis API consolidating stats entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any

from eeg_pipeline.analysis.behavior.feature_correlator import (
    CorrelationConfig,
    FeatureBehaviorCorrelator,
    FeatureCorrelationResult,
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
    compute_pain_sensitivity_index,
    interpret_correlation,
    interpret_effect_size,
    run_pain_sensitivity_correlations,
)
from eeg_pipeline.utils.analysis.stats.transforms import compute_change_features
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

if TYPE_CHECKING:
    from pathlib import Path
    import logging
    from eeg_pipeline.context.behavior import BehaviorContext


def run_cluster_test_from_context(ctx: "BehaviorContext") -> Optional[Dict[str, Any]]:
    """Run a two-condition cluster test (configurable) using pre-loaded context data."""
    if ctx.computation_features and "cluster" in ctx.computation_features:
        allowed = ctx.computation_features["cluster"]
        if "power" not in allowed and "spectral" not in allowed:
            ctx.logger.info("Skipping cluster test: feature filter %s excludes 'power'/'spectral'", allowed)
            return None

    from eeg_pipeline.infra.paths import ensure_dir
    
    # Create cluster subdirectory for organized output
    cluster_dir = ctx.stats_dir / "cluster"
    ensure_dir(cluster_dir)

    return _run_cluster_test_core(
        ctx.subject,
        ctx.epochs,
        ctx.aligned_events,
        cluster_dir,
        ctx.config,
        ctx.logger,
        ctx.n_perm,
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
    "run_unified_feature_correlations",
    # Feature registry (backward-compatible re-exports)
    "FeatureRegistry",
    "FeatureRule",
    # Cluster tests
    "compute_pain_nonpain_time_cluster_test",
    "run_cluster_test_from_context",
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



