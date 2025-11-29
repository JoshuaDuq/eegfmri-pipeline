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
from .signal_metrics import (
    compute_zero_crossings,
    compute_rms,
    compute_peak_to_peak,
    compute_line_length,
    compute_permutation_entropy,
    compute_hjorth_parameters,
    compute_lempel_ziv_complexity,
    compute_spectral_entropy,
    compute_band_power,
    compute_peak_frequency,
)
from .graph_metrics import (
    symmetrize_adjacency,
    threshold_adjacency,
    compute_global_efficiency_weighted,
    compute_small_world_sigma,
    compute_participation_coefficient,
    compute_clustering_coefficient,
)

__all__ = [
    # Core TFR utilities
    "compute_adaptive_n_cycles",
    "validate_baseline_indices",
    "sliding_window_centers",
    # Signal metrics
    "compute_zero_crossings",
    "compute_rms",
    "compute_peak_to_peak",
    "compute_line_length",
    "compute_permutation_entropy",
    "compute_hjorth_parameters",
    "compute_lempel_ziv_complexity",
    "compute_spectral_entropy",
    "compute_band_power",
    "compute_peak_frequency",
    # Graph metrics
    "symmetrize_adjacency",
    "threshold_adjacency",
    "compute_global_efficiency_weighted",
    "compute_small_world_sigma",
    "compute_participation_coefficient",
    "compute_clustering_coefficient",
]
