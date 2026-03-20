from __future__ import annotations

from typing import Any, Tuple

###################################################################
# Feature Categories and Groups
###################################################################

FEATURE_CATEGORIES = [
    # Core spectral features
    "power",          # Band power (log-ratio normalized)
    "spectral",       # Peak frequency, IAF, spectral edge
    "aperiodic",      # 1/f spectral slope (E/I balance)
    "erp",            # Time-domain evoked features (ERP/LEP)
    "erds",           # Event-related (de)synchronization
    "ratios",         # Band power ratios (theta/beta, etc.)
    "asymmetry",      # Hemispheric power asymmetry
    "microstates",    # EEG microstate dynamics (A-D)
    # Connectivity & phase
    "connectivity",   # Functional connectivity (wPLI, AEC)
    "directedconnectivity",  # Directed connectivity (PSI, DTF, PDC)
    "itpc",           # Inter-trial phase coherence
    "pac",            # Phase-amplitude coupling
    # Source localization
    "sourcelocalization",  # Source-space ROI features (LCMV, eLORETA)
    # Exploratory & QC
    "complexity",     # Signal complexity (exploratory)
    "bursts",         # Burst dynamics (beta/gamma)
    "quality",        # Trial quality metrics
]

SPATIAL_MODES = [
    "roi",       # Aggregate by ROI (mean across channels in each ROI)
    "channels",  # Per-channel features (no aggregation)
    "global",    # Global mean across all channels
]

###################################################################
# Numerical Constants
###################################################################

EPSILON_STD = 1e-12       # For standard deviation floor (avoid div-by-zero)
EPSILON_PSD = 1e-20       # For PSD floor (avoid log of zero)
MIN_SAMPLES_DEFAULT = 10  # Minimum samples for most feature extraction


###################################################################
# Validation Helpers
###################################################################


def validate_precomputed(
    precomputed: Any,
    *,
    require_windows: bool = False,
    require_bands: bool = False,
    require_psd: bool = False,
) -> Tuple[bool, str]:
    """Validate precomputed data structure.
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message). If valid, error_message is empty string.
    """
    if precomputed is None:
        return False, "precomputed is None"
    
    if not hasattr(precomputed, "data") or precomputed.data is None:
        return False, "precomputed.data is None"
    
    if precomputed.data.size == 0:
        return False, "precomputed.data is empty"
    
    if require_windows and precomputed.windows is None:
        return False, "precomputed.windows is None"
    
    if require_bands and not precomputed.band_data:
        return False, "precomputed.band_data is empty"
    
    if require_psd and precomputed.psd_data is None:
        return False, "precomputed.psd_data is None"
    
    return True, ""


def validate_extractor_inputs(
    ctx: Any,
    extractor_name: str,
    *,
    require_epochs: bool = True,
    require_windows: bool = True,
    min_epochs: int = 2,
) -> Tuple[bool, str]:
    """Validate common inputs for feature extractors.
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message). If valid, error_message is empty string.
    """
    if ctx is None:
        return False, f"{extractor_name}: context is None"
    
    if require_epochs:
        if not hasattr(ctx, "epochs") or ctx.epochs is None:
            return False, f"{extractor_name}: epochs is None"
        if len(ctx.epochs) < min_epochs:
            return False, f"{extractor_name}: insufficient epochs ({len(ctx.epochs)} < {min_epochs})"
    
    if require_windows:
        if not hasattr(ctx, "windows") or ctx.windows is None:
            return False, f"{extractor_name}: windows is None"
    
    return True, ""
