from __future__ import annotations

from typing import Tuple, Any, Dict, List, Callable, Optional
import numpy as np

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
    # Connectivity & phase
    "connectivity",   # Functional connectivity (wPLI, AEC)
    "itpc",           # Inter-trial phase coherence
    "pac",            # Phase-amplitude coupling
    # Exploratory & QC
    "complexity",     # Signal complexity (exploratory)
    "bursts",         # Burst dynamics (beta/gamma)
    "quality",        # Trial quality metrics
    "temporal",       # Time-resolved (binned) features
]

SPATIAL_MODES = [
    "roi",       # Aggregate by ROI (mean across channels in each ROI)
    "channels",  # Per-channel features (no aggregation)
    "global",    # Global mean across all channels
]

PRECOMPUTED_GROUP_CHOICES = [
    "erds",
    "spectral",
    "ratios",
    "asymmetry",
]


###################################################################
# Numerical Constants
###################################################################

EPSILON_STD = 1e-12       # For standard deviation floor (avoid div-by-zero)
EPSILON_PSD = 1e-20       # For PSD floor (avoid log of zero)
EPSILON_FANO = 1e-12      # For Fano factor denominator
MIN_SAMPLES_DEFAULT = 10  # Minimum samples for most feature extraction
MIN_SAMPLES_COMPLEXITY = 100  # Minimum samples for complexity metrics
MIN_SAMPLES_BURST = 10    # Minimum samples for burst detection


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
    
    Returns (is_valid, error_message).
    """
    if precomputed is None:
        return False, "precomputed is None"
    
    if not hasattr(precomputed, "data") or precomputed.data is None:
        return False, "precomputed.data is None"
    
    if precomputed.data.size == 0:
        return False, "precomputed.data is empty"
    
    if require_windows:
        if precomputed.windows is None:
            return False, "precomputed.windows is None"
    
    if require_bands:
        if not precomputed.band_data:
            return False, "precomputed.band_data is empty"
    
    if require_psd:
        if precomputed.psd_data is None:
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
    
    Returns (is_valid, error_message).
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


def get_segment_mask(windows: Any, segment_name: str) -> Optional[np.ndarray]:
    """Get segment mask using consistent access pattern.
    
    Standardizes access to window masks across all feature modules.
    """
    if windows is None or not segment_name:
        return None

    # 1. Preferred method
    if hasattr(windows, "get_mask"):
        return windows.get_mask(segment_name)

    # 2. Generic dictionary
    if hasattr(windows, "masks") and isinstance(windows.masks, dict):
        return windows.masks.get(segment_name)

    # 3. Legacy attribute access
    mask_attr = f"{segment_name}_mask"
    if hasattr(windows, mask_attr):
        return getattr(windows, mask_attr)

    return None
