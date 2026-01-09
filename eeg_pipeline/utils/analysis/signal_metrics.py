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

from math import factorial
from typing import Dict, Optional, Tuple

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
    sign_changes = np.diff(signs) != 0
    return int(np.sum(sign_changes))


def compute_rms(x: np.ndarray) -> float:
    """Compute root mean square of a signal."""
    if len(x) == 0:
        return np.nan
    
    mean_squared = np.mean(x ** 2)
    return float(np.sqrt(mean_squared))


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
    
    current_squared = x[1:-1] ** 2
    product_neighbors = x[:-2] * x[2:]
    nonlinear_energy = current_squared - product_neighbors
    return float(np.mean(nonlinear_energy))


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
    # Standard deviation over the channel axis (assumed to be the second to last)
    return np.nanstd(data, axis=-2)


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


def _count_template_matches(
    x: np.ndarray,
    template_length: int,
    tolerance: float,
) -> int:
    """Count matching templates using Chebyshev distance."""
    n = len(x)
    n_templates = n - template_length
    if n_templates < 2:
        return 0
    
    templates = np.array([x[i:i + template_length] for i in range(n_templates)])
    pairwise_diff = templates[:, np.newaxis, :] - templates[np.newaxis, :, :]
    chebyshev_distances = np.max(np.abs(pairwise_diff), axis=2)
    
    upper_triangle_indices = np.triu_indices(n_templates, k=1)
    distances_upper = chebyshev_distances[upper_triangle_indices]
    matches = distances_upper < tolerance
    return int(np.sum(matches))


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
        signal_std = np.std(x, ddof=1)
        if signal_std == 0:
            return np.nan
        r = r_multiplier * signal_std
    
    matches_m_plus_1 = _count_template_matches(x, m + 1, r)
    matches_m = _count_template_matches(x, m, r)
    
    if matches_m == 0:
        return np.nan
    
    if matches_m_plus_1 == 0:
        return np.nan
    
    ratio = matches_m_plus_1 / matches_m
    return float(-np.log(ratio))


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
    
    first_diff = np.diff(x)
    second_diff = np.diff(first_diff)
    
    variance_signal = np.var(x, ddof=1) if len(x) > 1 else np.nan
    variance_first_diff = np.var(first_diff, ddof=1) if len(first_diff) > 1 else np.nan
    variance_second_diff = np.var(second_diff, ddof=1) if len(second_diff) > 1 else np.nan
    
    if variance_signal == 0:
        return np.nan, np.nan, np.nan
    
    activity = float(variance_signal)
    
    if variance_signal > 0:
        mobility = float(np.sqrt(variance_first_diff / variance_signal))
    else:
        mobility = np.nan
    
    if variance_first_diff > 0 and mobility > 0:
        mobility_first_diff = np.sqrt(variance_second_diff / variance_first_diff)
        complexity = float(mobility_first_diff / mobility)
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

    psd_min = np.nanmin(band_psd)
    psd_max = np.nanmax(band_psd)
    psd_range = psd_max - psd_min
    
    if psd_range > 0:
        peak_prominence = (peak_power - psd_min) / psd_range
        if peak_prominence < min_prominence:
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

    frequency_bin_widths = np.gradient(freqs_clean)
    power_per_bin = psd_clean * frequency_bin_widths
    total_power = np.nansum(power_per_bin)
    
    if not np.isfinite(total_power) or total_power <= 0:
        return np.nan

    probabilities = power_per_bin / total_power
    probabilities = probabilities[probabilities > 0]
    
    if probabilities.size == 0:
        return np.nan

    spectral_entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    if normalize and probabilities.size > 1:
        max_entropy = np.log2(probabilities.size)
        spectral_entropy = spectral_entropy / max_entropy

    return spectral_entropy


