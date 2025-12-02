"""
Data loading utilities for decoding (Re-exports).

.. deprecated::
    This module re-exports from eeg_pipeline.utils.data.loading.
    Import directly from there for new code.
"""

from eeg_pipeline.utils.data.loading import (
    load_plateau_matrix,
    load_epoch_windows,
)

__all__ = [
    "load_plateau_matrix",
    "load_epoch_windows",
]
