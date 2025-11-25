# Import only the most commonly used shared helpers
# Most code imports directly from specific modules (e.g., utils.analysis.tfr, utils.analysis.stats)
# This __init__ provides a minimal convenience surface for truly shared utilities

from .tfr import (
    compute_adaptive_n_cycles,
    validate_baseline_indices,
)
from .windowing import (
    sliding_window_centers,
)

__all__ = [
    # Core TFR utilities used across multiple modules
    "compute_adaptive_n_cycles",
    "validate_baseline_indices",
    "sliding_window_centers",
]
