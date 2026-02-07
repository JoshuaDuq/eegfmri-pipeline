"""
Signal Metrics Utilities
=========================

Low-level signal processing functions for computing metrics on 1D arrays.
These are pure numpy functions with no EEG-specific dependencies.

Categories:
- GFP: global field power (spatial std over channels)
- Complexity: permutation entropy, sample entropy, multiscale entropy, Lempel-Ziv
"""

from __future__ import annotations

from math import factorial
from typing import Dict, Iterable, Optional

import numpy as np


def compute_gfp(data: np.ndarray) -> np.ndarray:
    """
    Compute Global Field Power (GFP) of multichannel data.
    
    GFP is defined as the spatial standard deviation across all channels 
    at each time point. It represents the global activity level of the 
    brain at that instant.
    
    Parameters
    ----------
    data : np.ndarray
        Multichannel data of shape (..., n_channels, n_times)
    
    Returns
    -------
    np.ndarray
        GFP time series of shape (..., n_times)
    """
    if data.size == 0:
        return np.array([])
    return np.nanstd(data, axis=-2)


# =============================================================================
# Complexity Metrics
# =============================================================================

def _embed_time_series(x: np.ndarray, order: int, delay: int) -> np.ndarray:
    """Create time-delay embedding of a signal using vectorized operations."""
    n = len(x)
    if n < (order - 1) * delay + 1:
        return np.array([])
    
    n_vectors = n - (order - 1) * delay
    indices = np.arange(order) * delay + np.arange(n_vectors)[:, np.newaxis]
    return x[indices]


def compute_permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Compute permutation entropy of a signal.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    order : int
        Embedding dimension (pattern length)
    delay : int
        Time delay for embedding
    normalize : bool
        If True, normalize by maximum entropy
    
    Returns
    -------
    float
        Permutation entropy value
    """
    if len(x) < (order - 1) * delay + 1:
        return np.nan
    
    embedded = _embed_time_series(x, order, delay)
    if embedded.size == 0:
        return np.nan
    
    n_vectors = embedded.shape[0]
    permutation_counts: Dict[tuple, int] = {}
    
    for i in range(n_vectors):
        sorted_indices = np.argsort(embedded[i])
        pattern = tuple(sorted_indices)
        permutation_counts[pattern] = permutation_counts.get(pattern, 0) + 1
    
    probabilities = np.array(list(permutation_counts.values())) / n_vectors
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    if normalize:
        max_entropy = np.log2(factorial(order))
        if max_entropy > 0:
            entropy = entropy / max_entropy
    
    return float(entropy)


def _sample_entropy_fallback(x: np.ndarray, order: int, tolerance: float) -> float:
    """Fallback sample entropy implementation when antropy is unavailable."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < order + 2:
        return np.nan

    embedded_m = _embed_time_series(x, order=order, delay=1)
    embedded_m1 = _embed_time_series(x, order=order + 1, delay=1)
    if embedded_m.size == 0 or embedded_m1.size == 0:
        return np.nan

    def _count_matches(vectors: np.ndarray) -> int:
        n = vectors.shape[0]
        count = 0
        for i in range(n - 1):
            dmax = np.max(np.abs(vectors[i + 1 :] - vectors[i]), axis=1)
            count += int(np.sum(dmax <= tolerance))
        return count

    b_count = _count_matches(embedded_m)
    a_count = _count_matches(embedded_m1)
    if b_count <= 0 or a_count <= 0:
        return np.nan
    return float(-np.log(a_count / b_count))


def compute_sample_entropy(
    x: np.ndarray,
    order: int = 2,
    r: float = 0.2,
    tolerance: Optional[float] = None,
) -> float:
    """Compute sample entropy using antropy when available."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < order + 2:
        return np.nan

    if tolerance is None:
        std = float(np.std(x, ddof=0))
        tolerance = float(r) * std
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        tolerance = max(float(np.finfo(float).eps), float(r) * float(np.nanstd(x)))

    try:
        from antropy import sample_entropy as _antropy_sample_entropy

        value = _antropy_sample_entropy(x, order=int(order), tolerance=tolerance, metric="chebyshev")
        return float(value) if np.isfinite(value) else np.nan
    except ImportError:
        return _sample_entropy_fallback(x, order=int(order), tolerance=tolerance)
    except Exception:
        return _sample_entropy_fallback(x, order=int(order), tolerance=tolerance)


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """Coarse-grain a signal by non-overlapping averaging."""
    if scale <= 1:
        return np.asarray(x, dtype=float)

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_complete = x.size // scale
    if n_complete <= 0:
        return np.array([], dtype=float)
    trimmed = x[: n_complete * scale]
    return np.mean(trimmed.reshape(n_complete, scale), axis=1)


def compute_multiscale_entropy(
    x: np.ndarray,
    scales: Iterable[int] = range(1, 21),
    order: int = 2,
    r: float = 0.2,
) -> Dict[int, float]:
    """
    Compute multiscale entropy (MSE) across coarse-graining scales.

    Uses non-overlapping averaging for coarse-graining and sample entropy
    at each scale.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out: Dict[int, float] = {}
    min_len = int(order) + 2

    for raw_scale in scales:
        scale = int(raw_scale)
        if scale <= 0:
            continue
        coarse = _coarse_grain(x, scale)
        if coarse.size < min_len:
            out[scale] = np.nan
            continue
        out[scale] = float(compute_sample_entropy(coarse, order=int(order), r=float(r)))

    return out


def compute_lempel_ziv_complexity(x: np.ndarray, threshold: Optional[float] = None) -> float:
    """
    Compute Lempel-Ziv complexity of a binarized signal.
    
    Uses an optimized incremental LZ76 algorithm.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    threshold : float, optional
        Binarization threshold. If None, uses median.
    
    Returns
    -------
    float
        Normalized Lempel-Ziv complexity
    """
    if len(x) < 2:
        return np.nan
    
    if threshold is None:
        threshold = np.median(x)
    
    binary = (x > threshold).astype(np.uint8)
    n = len(binary)
    
    vocabulary = set()
    complexity_count = 0
    position = 0
    
    while position < n:
        phrase_length = 1
        current_phrase = tuple(binary[position:position + phrase_length])
        
        while position + phrase_length <= n and current_phrase in vocabulary:
            phrase_length += 1
            current_phrase = tuple(binary[position:position + phrase_length])
        
        if position + phrase_length <= n:
            vocabulary.add(current_phrase)
        
        complexity_count += 1
        position += phrase_length
    
    if n > 1:
        theoretical_max = n / np.log2(n)
        return float(complexity_count / theoretical_max)
    
    return np.nan
