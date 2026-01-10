from __future__ import annotations

from typing import List, Tuple, Any, Optional, Dict
import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import EPSILON_PSD
from eeg_pipeline.utils.analysis.tfr import extract_tfr_object
from eeg_pipeline.utils.analysis.windowing import make_mask_for_times


def _extract_tfr_components(tfr: Any) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Extract TFR object and its core components.
    
    Returns:
        Tuple of (tfr_obj, data, freqs, times, channel_names) or None values if extraction fails.
    """
    tfr_obj = extract_tfr_object(tfr)
    if tfr_obj is None:
        return None, None, None, None, None
    
    data = tfr_obj.data
    freqs = tfr_obj.freqs
    times = tfr_obj.times
    channel_names = tfr_obj.info["ch_names"]
    
    return tfr_obj, data, freqs, times, channel_names


def _parse_baseline_column_name(column_name: str) -> Optional[Tuple[str, str]]:
    """Parse baseline column name to extract band and channel.
    
    Returns:
        (band, channel) tuple if valid baseline power column, None otherwise.
    """
    parsed = NamingSchema.parse(str(column_name))
    if not parsed.get("valid"):
        return None
    
    is_power_baseline = (
        parsed.get("group") == "power" and
        parsed.get("segment") == "baseline" and
        parsed.get("scope") == "ch"
    )
    if not is_power_baseline:
        return None
    
    band = parsed.get("band")
    channel = parsed.get("identifier")
    if not band or not channel:
        return None
    
    return band, channel


def _build_baseline_arrays(
    baseline_df: Optional[pd.DataFrame],
    channel_names: List[str],
) -> Dict[str, np.ndarray]:
    """Build baseline power arrays organized by frequency band.
    
    Args:
        baseline_df: DataFrame with baseline power features (may be None/empty).
        channel_names: Ordered list of channel names to match against.
    
    Returns:
        Dictionary mapping band names to (n_epochs, n_channels) arrays.
    """
    if baseline_df is None or baseline_df.empty:
        return {}

    baseline_map: Dict[str, Dict[str, np.ndarray]] = {}
    for column_name in baseline_df.columns:
        parsed_result = _parse_baseline_column_name(column_name)
        if parsed_result is None:
            continue
        
        band, channel = parsed_result
        values = baseline_df[column_name].to_numpy(dtype=float)
        baseline_map.setdefault(band, {})[channel] = values

    n_epochs = len(baseline_df)
    baseline_arrays: Dict[str, np.ndarray] = {}
    
    for band, channel_map in baseline_map.items():
        band_matrix = np.full((n_epochs, len(channel_names)), np.nan)
        for channel_idx, channel_name in enumerate(channel_names):
            channel_values = channel_map.get(channel_name)
            if channel_values is not None and len(channel_values) == n_epochs:
                band_matrix[:, channel_idx] = channel_values
        baseline_arrays[band] = band_matrix
    
    return baseline_arrays

def _compute_frequency_weights(band_frequencies: np.ndarray) -> np.ndarray:
    """Compute frequency weights for weighted averaging.
    
    Uses gradient to account for non-uniform frequency spacing (e.g., log-spaced).
    This prevents bias toward frequency bins with higher density.
    
    Args:
        band_frequencies: Array of frequencies within the band.
    
    Returns:
        Array of weights, same length as band_frequencies.
    """
    if band_frequencies.size >= 2 and np.all(np.isfinite(band_frequencies)):
        weights = np.gradient(band_frequencies).astype(float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, np.nan)
    else:
        weights = np.ones((band_frequencies.size,), dtype=float)
    
    return weights


def _compute_frequency_weighted_power(
    tfr_data: np.ndarray,
    frequency_mask: np.ndarray,
    time_mask: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Compute frequency-weighted mean power for a band and time window.
    
    Args:
        tfr_data: TFR data array (n_epochs, n_channels, n_freqs, n_times).
        frequency_mask: Boolean mask for frequencies in the band.
        time_mask: Boolean mask for time points in the segment.
        frequencies: Full frequency array.
    
    Returns:
        Array of shape (n_epochs, n_channels) with weighted mean power.
    """
    band_data = tfr_data[:, :, frequency_mask, :][:, :, :, time_mask]
    power_freq_time = np.nanmean(band_data, axis=3)
    
    band_frequencies = np.asarray(frequencies[frequency_mask], dtype=float)
    frequency_weights = _compute_frequency_weights(band_frequencies)
    
    weights_3d = frequency_weights[None, None, :]
    finite_mask = np.isfinite(power_freq_time) & np.isfinite(weights_3d)
    
    numerator = np.nansum(np.where(finite_mask, power_freq_time * weights_3d, 0.0), axis=2)
    denominator = np.nansum(np.where(finite_mask, weights_3d, 0.0), axis=2)
    
    weighted_power = np.where(denominator > 0, numerator / denominator, np.nan)
    return weighted_power


def _normalize_power(
    raw_power: np.ndarray,
    band: str,
    baseline_arrays: Dict[str, np.ndarray],
    is_tfr_baselined: bool,
    require_baseline: bool,
    epsilon_psd: float,
) -> Tuple[np.ndarray, str]:
    """Normalize raw power values.
    
    Args:
        raw_power: Raw power array (n_epochs, n_channels).
        band: Frequency band name.
        baseline_arrays: Dictionary of baseline arrays by band.
        is_tfr_baselined: Whether TFR was already baseline-corrected.
        require_baseline: Whether baseline is required for normalization.
        epsilon_psd: Epsilon value for PSD floor.
    
    Returns:
        Tuple of (normalized_values, statistic_name).
    """
    if is_tfr_baselined:
        return raw_power, "baselined"
    
    power_floor = np.maximum(raw_power, epsilon_psd)
    baseline_array = baseline_arrays.get(band)
    
    if baseline_array is None or not np.isfinite(baseline_array).any():
        if require_baseline:
            raise ValueError(
                f"Missing baseline power for band '{band}'; "
                "set feature_engineering.power.require_baseline=false to allow raw log power."
            )
        normalized = np.log10(power_floor)
        return normalized, "log10raw"
    
    baseline_denominator = baseline_array.copy()
    baseline_denominator[baseline_denominator <= 0] = np.nan
    normalized = np.log10(power_floor / baseline_denominator)
    return normalized, "logratio"


def _extract_channel_features(
    normalized_power: np.ndarray,
    segment_name: str,
    band: str,
    statistic_name: str,
    channel_names: List[str],
) -> Dict[str, np.ndarray]:
    """Extract per-channel power features.
    
    Args:
        normalized_power: Normalized power array (n_epochs, n_channels).
        segment_name: Name of the time segment.
        band: Frequency band name.
        statistic_name: Name of the normalization statistic.
        channel_names: List of channel names.
    
    Returns:
        Dictionary mapping column names to feature arrays.
    """
    features = {}
    for channel_idx, channel_name in enumerate(channel_names):
        column_name = NamingSchema.build(
            "power", segment_name, band, "ch", statistic_name, channel=channel_name
        )
        features[column_name] = normalized_power[:, channel_idx]
    
    return features


def _extract_global_features(
    normalized_power: np.ndarray,
    segment_name: str,
    band: str,
    statistic_name: str,
) -> Dict[str, np.ndarray]:
    """Extract global (across-channel) mean power features.
    
    Args:
        normalized_power: Normalized power array (n_epochs, n_channels).
        segment_name: Name of the time segment.
        band: Frequency band name.
        statistic_name: Name of the normalization statistic.
    
    Returns:
        Dictionary mapping column name to feature array.
    """
    global_mean = np.nanmean(normalized_power, axis=1)
    column_name = NamingSchema.build(
        "power", segment_name, band, "global", f"{statistic_name}_mean"
    )
    return {column_name: global_mean}


def _extract_roi_features(
    normalized_power: np.ndarray,
    segment_name: str,
    band: str,
    statistic_name: str,
    roi_map: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """Extract ROI (region of interest) mean power features.
    
    Args:
        normalized_power: Normalized power array (n_epochs, n_channels).
        segment_name: Name of the time segment.
        band: Frequency band name.
        statistic_name: Name of the normalization statistic.
        roi_map: Dictionary mapping ROI names to channel indices.
    
    Returns:
        Dictionary mapping column names to feature arrays.
    """
    features = {}
    for roi_name, channel_indices in roi_map.items():
        if len(channel_indices) == 0:
            continue
        
        roi_mean = np.nanmean(normalized_power[:, channel_indices], axis=1)
        column_name = NamingSchema.build(
            "power", segment_name, band, "roi", f"{statistic_name}_mean", channel=roi_name
        )
        features[column_name] = roi_mean
    
    return features


def _build_band_frequency_masks(
    bands: List[str],
    frequency_bands: Dict[str, Tuple[float, float]],
    frequencies: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build frequency masks for each band.
    
    Args:
        bands: List of band names to process.
        frequency_bands: Dictionary mapping band names to (fmin, fmax) tuples.
        frequencies: Full frequency array.
    
    Returns:
        Dictionary mapping band names to boolean frequency masks.
    """
    band_masks = {}
    for band in bands:
        if band not in frequency_bands:
            continue
        
        fmin, fmax = frequency_bands[band]
        frequency_mask = (frequencies >= fmin) & (frequencies <= fmax)
        if np.any(frequency_mask):
            band_masks[band] = frequency_mask
    
    return band_masks


def _build_roi_map_if_needed(
    spatial_modes: List[str],
    channel_names: List[str],
    config: Any,
) -> Dict[str, List[int]]:
    """Build ROI map if ROI features are requested.
    
    Args:
        spatial_modes: List of spatial aggregation modes.
        channel_names: List of channel names.
        config: Configuration object.
    
    Returns:
        Dictionary mapping ROI names to lists of channel indices.
    """
    if 'roi' not in spatial_modes:
        return {}
    
    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
    from eeg_pipeline.utils.analysis.channels import build_roi_map
    
    roi_definitions = get_roi_definitions(config)
    if not roi_definitions:
        return {}
    
    roi_index_map = build_roi_map(channel_names, roi_definitions)
    return roi_index_map


def _validate_baseline_requirements(
    baseline_df: Optional[pd.DataFrame],
    n_epochs: int,
    is_tfr_baselined: bool,
    require_baseline: bool,
) -> None:
    """Validate baseline requirements for power normalization.
    
    Args:
        baseline_df: Baseline features DataFrame.
        n_epochs: Number of epochs in TFR data.
        is_tfr_baselined: Whether TFR was already baseline-corrected.
        require_baseline: Whether baseline is required.
    
    Raises:
        ValueError: If baseline requirements are not met.
    """
    if require_baseline and not is_tfr_baselined:
        if baseline_df is None or baseline_df.empty:
            raise ValueError("Power features require baseline_df for log-ratio normalization.")
        if len(baseline_df) != n_epochs:
            raise ValueError(
                f"Baseline feature length mismatch for power normalization: "
                f"{len(baseline_df)} vs {n_epochs}"
            )


def _check_tfr_baselined(tfr_obj: Any) -> bool:
    """Check if TFR object has been baseline-corrected.
    
    Args:
        tfr_obj: TFR object to check.
    
    Returns:
        True if TFR comment indicates baseline correction.
    """
    comment = getattr(tfr_obj, "comment", None)
    return isinstance(comment, str) and "BASELINED:" in comment


def extract_power_features(
    ctx: Any,  # FeatureContext
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract power features for defined time segments.
    
    Computes:
    - Raw power for baseline (if available) - internalized for normalization
    - Log-ratio power for active segments relative to baseline
    - Global mean power per band
    - ROI mean power per band (if spatial_modes includes 'roi')
    
    Args:
        ctx: FeatureContext with TFR data, windows, and configuration.
        bands: List of frequency band names to extract.
    
    Returns:
        Tuple of (features_dataframe, column_names_list).
    """
    if not bands:
        return pd.DataFrame(), []

    tfr = ctx.results.get("tfr")
    if tfr is None:
        return pd.DataFrame(), []

    tfr_obj, tfr_data, freqs, times, channel_names = _extract_tfr_components(tfr)
    if tfr_data is None:
        return pd.DataFrame(), []

    is_tfr_baselined = _check_tfr_baselined(tfr_obj)

    from eeg_pipeline.utils.config.loader import get_frequency_bands
    frequency_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(ctx.config)
    
    spatial_modes = getattr(ctx, 'spatial_modes', ['roi', 'global'])
    roi_map = _build_roi_map_if_needed(spatial_modes, channel_names, ctx.config)
    
    band_frequency_masks = _build_band_frequency_masks(bands, frequency_bands, freqs)
    if not band_frequency_masks:
        return pd.DataFrame(), []

    n_epochs = len(tfr_data)
    
    baseline_df = getattr(ctx, "baseline_df", None)
    baseline_arrays = _build_baseline_arrays(baseline_df, channel_names)
    
    require_baseline = bool(ctx.config.get("feature_engineering.power.require_baseline", True))
    _validate_baseline_requirements(baseline_df, n_epochs, is_tfr_baselined, require_baseline)
    
    segment_name = getattr(ctx, "name", None)
    ctx.logger.info(f"Computing power features for segment: {segment_name or 'unnamed'}")
    
    time_mask = make_mask_for_times(ctx.windows, segment_name, times)
    if not np.any(time_mask):
        ctx.logger.error(
            f"Time window '{segment_name}' is undefined or empty. "
            f"Available windows: {list(ctx.windows.ranges.keys()) if ctx.windows else 'none'}. "
            "Skipping power feature extraction for this segment."
        )
        return pd.DataFrame(), []
    
    epsilon_psd = float(ctx.config.get("feature_engineering.constants.epsilon_psd", EPSILON_PSD))
    output_features = {}
    
    for band, frequency_mask in band_frequency_masks.items():
        try:
            raw_power = _compute_frequency_weighted_power(
                tfr_data, frequency_mask, time_mask, freqs
            )
            
            normalized_power, statistic_name = _normalize_power(
                raw_power,
                band,
                baseline_arrays,
                is_tfr_baselined,
                require_baseline,
                epsilon_psd,
            )
            
            if 'channels' in spatial_modes:
                channel_features = _extract_channel_features(
                    normalized_power, segment_name, band, statistic_name, channel_names
                )
                output_features.update(channel_features)
            
            if 'global' in spatial_modes:
                global_features = _extract_global_features(
                    normalized_power, segment_name, band, statistic_name
                )
                output_features.update(global_features)
            
            if 'roi' in spatial_modes and roi_map:
                roi_features = _extract_roi_features(
                    normalized_power, segment_name, band, statistic_name, roi_map
                )
                output_features.update(roi_features)
                
        except Exception as e:
            ctx.logger.error(f"Error computing power features for {segment_name} {band}: {e}")

    if not output_features:
        return pd.DataFrame(), []
        
    features_df = pd.DataFrame(output_features)
    return features_df, list(features_df.columns)
