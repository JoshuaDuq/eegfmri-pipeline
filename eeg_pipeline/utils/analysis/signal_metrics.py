"""
Signal Metrics Utilities
=========================

Low-level signal processing functions for computing metrics on 1D arrays.
These are pure numpy functions with no EEG-specific dependencies.

Categories:
- GFP: global field power (spatial std over channels)
- Complexity: permutation entropy, Lempel-Ziv
"""

from __future__ import annotations

from math import factorial
from typing import Dict, Optional

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
