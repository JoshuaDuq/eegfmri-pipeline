"""
Signal Metrics Utilities
=========================

Low-level signal processing functions for computing metrics on 1D arrays.
These are pure numpy functions with no EEG-specific dependencies.

Categories:
- Time-domain: zero crossings, RMS, peak-to-peak, line length
- Complexity: permutation entropy, sample entropy, Hjorth, Lempel-Ziv
- Statistical: MAD, nonlinear energy
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from math import factorial

import numpy as np


# =============================================================================
# Time-Domain Metrics
# =============================================================================

def compute_zero_crossings(x: np.ndarray) -> int:
    """Count zero crossings in a 1D signal."""
    if len(x) < 2:
        return 0
    signs = np.sign(x)
    signs[signs == 0] = 1
    return int(np.sum(np.diff(signs) != 0))


def compute_rms(x: np.ndarray) -> float:
    """Compute root mean square of a signal."""
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def compute_peak_to_peak(x: np.ndarray) -> float:
    """Compute peak-to-peak amplitude."""
    if len(x) == 0:
        return np.nan
    return float(np.max(x) - np.min(x))


def compute_line_length(x: np.ndarray) -> float:
    """Compute line length (sum of absolute differences)."""
    if len(x) < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(x))))


def compute_mean_absolute_deviation(x: np.ndarray) -> float:
    """Compute mean absolute deviation from the mean."""
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x - np.mean(x))))


def compute_nonlinear_energy(x: np.ndarray) -> float:
    """Compute Teager-Kaiser nonlinear energy operator."""
    if len(x) < 3:
        return np.nan
    nle = x[1:-1] ** 2 - x[:-2] * x[2:]
    return float(np.mean(nle))


# =============================================================================
# Complexity Metrics
# =============================================================================

def embed_time_series(x: np.ndarray, order: int, delay: int) -> np.ndarray:
    """Create time-delay embedding of a signal using vectorized operations."""
    n = len(x)
    if n < (order - 1) * delay + 1:
        return np.array([])
    
    n_vectors = n - (order - 1) * delay
    
    # Vectorized embedding using advanced indexing
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
    
    embedded = embed_time_series(x, order, delay)
    if embedded.size == 0:
        return np.nan
    
    n_vectors = embedded.shape[0]
    perm_counts: Dict[tuple, int] = {}
    
    for i in range(n_vectors):
        pattern = tuple(np.argsort(embedded[i]))
        perm_counts[pattern] = perm_counts.get(pattern, 0) + 1
    
    probs = np.array(list(perm_counts.values())) / n_vectors
    probs = probs[probs > 0]
    pe = -np.sum(probs * np.log2(probs))
    
    if normalize:
        max_entropy = np.log2(factorial(order))
        if max_entropy > 0:
            pe = pe / max_entropy
    
    return float(pe)


def compute_sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    r_multiplier: float = 0.2,
) -> float:
    """
    Compute sample entropy using vectorized Chebyshev distance.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    m : int
        Embedding dimension
    r : float, optional
        Tolerance threshold. If None, uses r_multiplier * std(x)
    r_multiplier : float
        Multiplier for automatic r calculation
    
    Returns
    -------
    float
        Sample entropy value
    """
    n = len(x)
    if n < m + 2:
        return np.nan
    
    if r is None:
        std = np.std(x, ddof=1)
        if std == 0:
            return np.nan
        r = r_multiplier * std
    
    def count_matches_vectorized(template_length: int) -> int:
        n_templates = n - template_length
        if n_templates < 2:
            return 0
        
        templates = np.array([x[i:i + template_length] for i in range(n_templates)])
        diff = templates[:, np.newaxis, :] - templates[np.newaxis, :, :]
        chebyshev = np.max(np.abs(diff), axis=2)
        
        triu_idx = np.triu_indices(n_templates, k=1)
        return int(np.sum(chebyshev[triu_idx] < r))
    
    a = count_matches_vectorized(m + 1)
    b = count_matches_vectorized(m)
    
    if b == 0:
        return np.nan
    
    return float(-np.log(a / b)) if a > 0 else np.nan


def compute_hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Hjorth parameters: activity, mobility, complexity.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    
    Returns
    -------
    Tuple[float, float, float]
        (activity, mobility, complexity)
    """
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    
    var_x = np.var(x, ddof=1) if len(x) > 1 else np.nan
    var_d1 = np.var(diff1, ddof=1) if len(diff1) > 1 else np.nan
    var_d2 = np.var(diff2, ddof=1) if len(diff2) > 1 else np.nan
    
    if var_x == 0:
        return np.nan, np.nan, np.nan
    
    activity = float(var_x)
    mobility = float(np.sqrt(var_d1 / var_x)) if var_x > 0 else np.nan
    
    if var_d1 > 0:
        mobility_d1 = np.sqrt(var_d2 / var_d1)
        complexity = float(mobility_d1 / mobility) if mobility > 0 else np.nan
    else:
        complexity = np.nan
    
    return activity, mobility, complexity


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
    
    # Standard LZ76 implementation
    # We maintain seen substrings in a set for O(1) lookup
    vocabulary = set()
    c = 0  # Complexity count
    
    i = 0  # Start of current substring
    while i < n:
        # Try to extend the current substring as long as it exists in vocabulary
        length = 1
        while i + length <= n and tuple(binary[i:i + length]) in vocabulary:
            length += 1
        
        # Add the new pattern to vocabulary (includes the novel extension)
        if i + length <= n:
            vocabulary.add(tuple(binary[i:i + length]))
        
        c += 1
        i += length  # Move past the current phrase
    
    # Normalize by theoretical maximum
    if n > 1:
        b = n / np.log2(n)
        return float(c / b)
    return np.nan


# =============================================================================
# Spectral Metrics
# =============================================================================

def compute_peak_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    min_prominence: float = 0.1,
) -> Tuple[float, float]:
    """
    Find peak frequency and power within a frequency band.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array
    psd : np.ndarray
        Power spectral density array
    fmin, fmax : float
        Frequency band limits
    min_prominence : float
        Minimum prominence for peak detection (0-1)
    
    Returns
    -------
    Tuple[float, float]
        (peak_frequency, peak_power)
    """
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return np.nan, np.nan

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    if band_psd.size == 0 or not np.any(np.isfinite(band_psd)):
        return np.nan, np.nan

    peak_idx = np.nanargmax(band_psd)
    peak_freq = float(band_freqs[peak_idx])
    peak_power = float(band_psd[peak_idx])

    psd_range = np.nanmax(band_psd) - np.nanmin(band_psd)
    if psd_range > 0:
        prominence = (peak_power - np.nanmin(band_psd)) / psd_range
        if prominence < min_prominence:
            return np.nan, np.nan

    return peak_freq, peak_power


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Compute absolute power within a frequency band by integrating the PSD.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array
    psd : np.ndarray
        Power spectral density array
    fmin, fmax : float
        Frequency band limits
    
    Returns
    -------
    float
        Band-limited power (area under PSD curve)
    """
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return np.nan

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    finite_mask = np.isfinite(band_freqs) & np.isfinite(band_psd)
    if not np.any(finite_mask):
        return np.nan

    band_freqs = band_freqs[finite_mask]
    band_psd = band_psd[finite_mask]

    if band_freqs.size < 2:
        return np.nan

    return float(np.trapz(band_psd, band_freqs))


def compute_spectral_entropy(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    normalize: bool = True,
) -> float:
    """
    Compute spectral entropy of a PSD.
    
    Spectral entropy measures the flatness/peakedness of the spectrum.
    High entropy = flat spectrum (white noise-like)
    Low entropy = peaked spectrum (dominant frequency)
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array
    psd : np.ndarray
        Power spectral density array
    fmin, fmax : float, optional
        Frequency limits for computation
    normalize : bool
        If True, normalize by maximum entropy
    
    Returns
    -------
    float
        Spectral entropy value
    """
    if fmin is not None and fmax is not None:
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[mask]
        psd = psd[mask]

    if psd.size == 0 or not np.any(np.isfinite(psd)) or freqs.size == 0:
        return np.nan
    if freqs.size < 2:
        return np.nan

    freqs_clean = np.asarray(freqs, dtype=float)
    psd_clean = np.asarray(psd, dtype=float)

    finite_mask = np.isfinite(freqs_clean) & np.isfinite(psd_clean)
    freqs_clean = freqs_clean[finite_mask]
    psd_clean = psd_clean[finite_mask]

    if freqs_clean.size < 2 or not np.any(psd_clean > 0):
        return np.nan

    # Convert density to power by integrating over frequency bins
    freq_weights = np.gradient(freqs_clean)
    power = psd_clean * freq_weights
    total_power = np.nansum(power)
    if not np.isfinite(total_power) or total_power <= 0:
        return np.nan

    p = power / total_power
    p = p[p > 0]
    if p.size == 0:
        return np.nan

    se = float(-np.sum(p * np.log2(p)))

    if normalize and p.size > 1:
        se = se / np.log2(p.size)

    return se


def compute_gfp(data: np.ndarray) -> np.ndarray:
    """
    Compute Global Field Power for all epochs.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (epochs, channels, times)
        
    Returns
    -------
    np.ndarray
        GFP (epochs, times)
    """
    return np.std(data, axis=1)
