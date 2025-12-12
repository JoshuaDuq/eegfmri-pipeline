"""
Decoding plotting module.

Low-level plotting primitives live here. High-level orchestration/IO is defined in
`pipelines.viz.decoding` to keep responsibilities separated.
"""

from __future__ import annotations

# Time-generalization plotting
from .time_generalization import (
    plot_time_generalization_matrix,
    plot_time_generalization_with_null,
)

# Performance metric plotting
from .performance import (
    plot_prediction_scatter,
    plot_per_subject_performance,
    plot_decoding_null_hist,
    plot_calibration_curve,
    plot_bootstrap_distributions,
    plot_permutation_null,
)

# Residual diagnostics plotting
from .diagnostics import (
    plot_residual_diagnostics,
)

# Model comparison plotting
from .comparisons import (
    plot_model_comparison,
    plot_riemann_band_comparison,
    plot_riemann_sliding_window,
    plot_incremental_validity,
)

# Feature importance plotting
from .importance import (
    plot_feature_importance_top_n,
    plot_feature_importance_stability,
)

__all__ = [
    # Time-generalization plotting
    "plot_time_generalization_matrix",
    "plot_time_generalization_with_null",
    # Performance metric plotting
    "plot_prediction_scatter",
    "plot_per_subject_performance",
    "plot_decoding_null_hist",
    "plot_calibration_curve",
    "plot_bootstrap_distributions",
    "plot_permutation_null",
    # Residual diagnostics plotting
    "plot_residual_diagnostics",
    # Model comparison plotting
    "plot_model_comparison",
    "plot_riemann_band_comparison",
    "plot_riemann_sliding_window",
    "plot_incremental_validity",
    # Feature importance plotting
    "plot_feature_importance_top_n",
    "plot_feature_importance_stability",
]
