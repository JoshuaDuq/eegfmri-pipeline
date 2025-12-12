"""Compatibility wrapper for TFR visualization entrypoints.

`eeg_pipeline.plotting.tfr.viz` historically contained orchestration code.
The canonical entrypoints now live in `eeg_pipeline.pipelines.viz.tfr`.

This wrapper preserves public imports.
"""

from __future__ import annotations

from .. import visualize_subject_tfr, visualize_tfr_for_subjects  # noqa: F401

__all__ = [
    "visualize_subject_tfr",
    "visualize_tfr_for_subjects",
]
