"""Compatibility wrapper for decoding visualization entrypoints.

`eeg_pipeline.plotting.decoding.viz` historically contained orchestration code.
The canonical entrypoints now live in `eeg_pipeline.pipelines.viz.decoding`.

This wrapper preserves public imports.
"""

from __future__ import annotations

from eeg_pipeline.pipelines.viz.decoding import (  # noqa: F401
    visualize_regression_results,
    visualize_time_generalization,
    visualize_model_comparisons,
    visualize_riemann_analysis,
    visualize_incremental_validity,
)

__all__ = [
    "visualize_regression_results",
    "visualize_time_generalization",
    "visualize_model_comparisons",
    "visualize_riemann_analysis",
    "visualize_incremental_validity",
]
