from __future__ import annotations

from typing import Tuple, Any, Dict, List, Callable, Optional
import numpy as np

###################################################################
# Feature Categories and Groups
###################################################################

FEATURE_CATEGORIES = [
    "power",
    "connectivity",
    "microstates",
    "aperiodic",
    "itpc",
    "pac",
    "precomputed",
    "cfc",
    "dynamics_advanced",
    "complexity",
    "quality",
]

PRECOMPUTED_GROUP_CHOICES = [
    "erds",
    "spectral",
    "gfp",
    "roi",
    "temporal",
    "ratios",
    "complexity",
    "asymmetry",
    "aperiodic",
    "connectivity",
    "microstates",
    "pac",
    "cfc",
    "dynamics_advanced",
    "itpc",
    "quality",
]

###################################################################
# Standard Segments for Feature Extraction
###################################################################

STANDARD_SEGMENTS = ["baseline", "ramp", "plateau"]
SEGMENT_BASELINE = "baseline"
SEGMENT_RAMP = "ramp"
SEGMENT_PLATEAU = "plateau"

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
    if windows is None:
        return None
    if hasattr(windows, "get_mask"):
        return windows.get_mask(segment_name)
    mask_attr = f"{segment_name}_mask"
    if hasattr(windows, mask_attr):
        return getattr(windows, mask_attr)
    if hasattr(windows, "masks") and isinstance(windows.masks, dict):
        return windows.masks.get(segment_name)
    return None


def process_standard_segments(
    windows: Any,
    processor_fn: Callable[[str, np.ndarray], Dict[str, Any]],
    min_samples: int = MIN_SAMPLES_DEFAULT,
    segments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Process standard segments (baseline, ramp, plateau) with a processor function.
    
    Parameters
    ----------
    windows : Any
        Time window specification with get_mask() method
    processor_fn : Callable[[str, np.ndarray], Dict[str, Any]]
        Function that takes (segment_name, mask) and returns dict of results
    min_samples : int
        Minimum samples required for processing
    segments : Optional[List[str]]
        Segments to process (default: STANDARD_SEGMENTS)
        
    Returns
    -------
    Dict[str, Any]
        Combined results from all processed segments
    """
    if segments is None:
        segments = STANDARD_SEGMENTS
    
    results: Dict[str, Any] = {}
    for seg_name in segments:
        mask = get_segment_mask(windows, seg_name)
        if mask is None or not np.any(mask):
            continue
        if np.sum(mask) < min_samples:
            continue
        seg_results = processor_fn(seg_name, mask)
        if seg_results:
            results.update(seg_results)
    return results
