"""
Spectral Feature Extraction
============================

Consolidated module for all spectral power and descriptor features:
- TFR-based power extraction with baseline normalization
- Precomputed-based power extraction (log-ratio, slope)
- Spectral descriptor extraction (IAF, peak power, center, bandwidth, entropy, edge)

This module consolidates power.py, spectral.py, and precomputed/spectral.py
to eliminate duplicated spatial aggregation logic and provide a unified interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import EPSILON_PSD, validate_precomputed
from eeg_pipeline.utils.analysis.tfr import extract_tfr_object
from eeg_pipeline.utils.analysis.windowing import make_mask_for_times, get_segment_masks
from eeg_pipeline.utils.analysis.channels import pick_eeg_channels, build_roi_map
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
from eeg_pipeline.utils.config.loader import get_frequency_bands, get_feature_constant
from eeg_pipeline.utils.analysis.arrays import nanmean_with_fraction
from eeg_pipeline.types import PrecomputedData


###################################################################
# TFR-BASED POWER EXTRACTION
###################################################################


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
    """Extract power features for defined time segments using TFR data.
    
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


###################################################################
# PRECOMPUTED-BASED POWER EXTRACTION
###################################################################


def extract_power_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Extract power features from precomputed band data.
    
    Computes log-ratio power relative to baseline, plus temporal slope features.
    
    Args:
        precomputed: PrecomputedData with band power and window masks.
        bands: List of frequency band names to extract.
    
    Returns:
        Tuple of (features_dataframe, column_names_list, qc_dict).
    """
    # Import validate_window_masks from precomputed.extras
    from eeg_pipeline.analysis.features.precomputed.extras import validate_window_masks
    
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("Power (precomputed): %s; skipping extraction.", err_msg)
        return pd.DataFrame(), [], {}

    n_epochs = precomputed.data.shape[0]
    min_epochs = get_feature_constant(precomputed.config, "MIN_EPOCHS_FOR_FEATURES", 10)
    if n_epochs < min_epochs:
        if precomputed.logger:
            precomputed.logger.warning(
                "Power extraction skipped: only %d epochs available (min=%d). "
                "Insufficient trials for stable power estimation.",
                n_epochs,
                min_epochs,
            )
        return pd.DataFrame(), [], {"skipped_reason": "insufficient_epochs", "n_epochs": n_epochs}

    if not validate_window_masks(precomputed, precomputed.logger):
        return pd.DataFrame(), [], {}

    epsilon = float(get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12))
    min_valid_fraction = float(get_feature_constant(precomputed.config, "MIN_VALID_FRACTION", 0.5))
    windows = precomputed.windows

    # Determine which segments to process
    target_name = getattr(windows, "name", None)
    if target_name and target_name in windows.masks:
        segments_to_process = [(target_name, windows.get_mask(target_name))]
    else:
        # Process all non-baseline segments
        segments_to_process = [
            (name, mask) for name, mask in windows.masks.items() 
            if name.lower() != "baseline" and np.any(mask)
        ]
    
    spatial_modes = getattr(precomputed, "spatial_modes", ["roi", "global"])
    roi_map = {}
    if "roi" in spatial_modes:
        roi_defs = get_roi_definitions(precomputed.config)
        if roi_defs:
            roi_map = build_roi_map(precomputed.ch_names, roi_defs)

    times = precomputed.times
    baseline_mask = windows.baseline_mask
    if baseline_mask is None:
        baseline_mask = np.zeros_like(times, dtype=bool)
    records: List[Dict[str, Any]] = [{} for _ in range(n_epochs)]

    if not segments_to_process:
        if precomputed.logger:
            precomputed.logger.warning("Power extraction: No user-defined segments to process.")
        return pd.DataFrame(), [], {}

    qc_payload: Dict[str, Any] = {
        "baseline_valid_fractions": [[] for _ in range(n_epochs)],
        "min_valid_fraction": min_valid_fraction,
    }

    for seg_label, active_mask in segments_to_process:
        active_times = times[active_mask] if np.any(active_mask) else np.array([])
        
        for ep_idx in range(n_epochs):
            record = records[ep_idx]

            for band in bands:
                if band not in precomputed.band_data:
                    continue

                power = precomputed.band_data[band].power[ep_idx]
                baseline_valid_count = 0
                total_channels = len(precomputed.ch_names)
                all_power_full: List[float] = []

                for ch_idx, ch_name in enumerate(precomputed.ch_names):
                    baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                        power[ch_idx], baseline_mask
                    )
                    active_power, _, _, _ = nanmean_with_fraction(power[ch_idx], active_mask)
                    baseline_valid = (
                        baseline_power > epsilon
                        and baseline_frac >= min_valid_fraction
                        and baseline_total > 0
                    )
                    if baseline_valid:
                        baseline_valid_count += 1

                    if baseline_valid and active_power > 0:
                        logratio = np.log10(active_power / baseline_power)
                    else:
                        logratio = np.nan
                    
                    record[
                        NamingSchema.build("spectral", seg_label, band, "ch", "logratio", channel=ch_name)
                    ] = float(logratio)
                    all_power_full.append(float(logratio) if np.isfinite(logratio) else np.nan)

                    if len(active_times) > 2:
                        active_power_trace = power[ch_idx, active_mask]
                        if baseline_valid:
                            logratio_trace = np.log10(np.maximum(active_power_trace / baseline_power, epsilon))
                            valid_mask = np.isfinite(logratio_trace)
                            if np.sum(valid_mask) > 2:
                                slope, _ = np.polyfit(
                                    active_times[valid_mask], logratio_trace[valid_mask], 1
                                )
                                record[
                                    NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                                ] = float(slope)
                            else:
                                record[
                                    NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                                ] = np.nan
                        else:
                            record[
                                NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                            ] = np.nan
                    else:
                        record[
                            NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                        ] = np.nan

                baseline_valid_fraction = (
                    baseline_valid_count / total_channels if total_channels > 0 else 0.0
                )
                
                # Store valid fraction in QC instead of columns
                qc_payload["baseline_valid_fractions"][ep_idx].append(float(baseline_valid_fraction))

                if "global" in spatial_modes:
                    if baseline_valid_fraction < min_valid_fraction:
                        record[
                            NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                        ] = np.nan
                        record[
                            NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                        ] = np.nan
                    else:
                        valid_powers = [p for p in all_power_full if np.isfinite(p)]
                        if valid_powers:
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                            ] = float(np.mean(valid_powers))
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                            ] = float(np.std(valid_powers))
                        else:
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                            ] = np.nan
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                            ] = np.nan
                
                if "roi" in spatial_modes and roi_map:
                    for roi_name, roi_indices in roi_map.items():
                        roi_powers = [all_power_full[idx] for idx in roi_indices if np.isfinite(all_power_full[idx])]
                        if roi_powers:
                            record[NamingSchema.build("spectral", seg_label, band, "roi", "logratio_mean", channel=roi_name)] = float(np.mean(roi_powers))
                        else:
                            record[NamingSchema.build("spectral", seg_label, band, "roi", "logratio_mean", channel=roi_name)] = np.nan

                if "channels" not in spatial_modes:
                    for ch_name in precomputed.ch_names:
                        record.pop(NamingSchema.build("spectral", seg_label, band, "ch", "logratio", channel=ch_name), None)
                        record.pop(NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name), None)

    if not records or all(not r for r in records):
        return pd.DataFrame(), [], {}

    # Summarize QC
    all_fractions = [f for ep_list in qc_payload["baseline_valid_fractions"] for f in ep_list]
    if all_fractions:
        qc_payload["mean_baseline_valid_fraction"] = float(np.mean(all_fractions))
        qc_payload["min_baseline_valid_fraction"] = float(np.min(all_fractions))
    
    # Remove large per-trial list from final QC to keep it small
    qc_payload.pop("baseline_valid_fractions", None)

    df = pd.DataFrame(records)
    return df, list(df.columns), qc_payload


###################################################################
# SPECTRAL DESCRIPTOR EXTRACTION
###################################################################


def remove_aperiodic_component(
    psd: np.ndarray,
    freqs: np.ndarray,
    fit_range: Tuple[float, float] = (2.0, 40.0),
) -> np.ndarray:
    """
    Remove 1/f aperiodic component from PSD using robust linear fit in log-log space.
    
    This addresses the scientific validity concern that raw PSD peak detection
    is biased by 1/f changes (common in pain/arousal/alertness states).
    
    Parameters
    ----------
    psd : np.ndarray
        Power spectral density (1D array)
    freqs : np.ndarray
        Frequency values
    fit_range : tuple
        Frequency range for fitting 1/f model (Hz)
        
    Returns
    -------
    residual : np.ndarray
        Aperiodic-adjusted PSD (residual in log space)
    """
    if psd.size == 0 or freqs.size == 0:
        return psd.copy()
    
    log_f = np.log10(np.maximum(freqs, 1e-6))
    log_p = np.log10(np.maximum(psd, 1e-20))
    
    fit_mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & np.isfinite(log_p)
    if np.sum(fit_mask) < 5:
        return psd.copy()
    
    try:
        slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
        aperiodic_fit = intercept + slope * log_f
        residual = log_p - aperiodic_fit
        return 10 ** residual
    except (np.linalg.LinAlgError, ValueError):
        return psd.copy()


def compute_peak_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    aperiodic_adjusted: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Compute peak frequency and peak power within a frequency range.
    
    Parameters
    ----------
    psd : np.ndarray
        Power spectral density (1D array)
    freqs : np.ndarray
        Frequency values
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    aperiodic_adjusted : bool
        If True, remove 1/f component before peak detection to avoid
        bias from aperiodic power changes (default: True)
    
    Returns
    -------
    peak_freq : float
        Frequency of maximum power (on aperiodic-adjusted spectrum if enabled)
    peak_power : float
        Power at peak frequency (from original PSD)
    peak_ratio : float
        Ratio of raw power to aperiodic fit at peak frequency (oscillatory 
        prominence relative to 1/f background). Values > 1 indicate a peak
        above the aperiodic floor.
    peak_residual : float
        Log10(power) - log10(aperiodic_fit) at peak frequency. Positive values
        indicate oscillatory power above the aperiodic background.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    if len(psd_band) == 0 or np.all(np.isnan(psd_band)):
        return np.nan, np.nan, np.nan, np.nan
    
    # Always compute aperiodic fit for prominence metrics
    log_f = np.log10(np.maximum(freqs, 1e-6))
    log_p = np.log10(np.maximum(psd, 1e-20))
    
    fit_range = (2.0, 40.0)  # Standard fit range
    fit_mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & np.isfinite(log_p)
    
    aperiodic_fit = None
    if np.sum(fit_mask) >= 5:
        try:
            slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
            aperiodic_fit = 10 ** (intercept + slope * log_f)  # Linear in log-log space
        except (np.linalg.LinAlgError, ValueError):
            pass
    
    if aperiodic_adjusted and aperiodic_fit is not None:
        residual = log_p - (np.log10(aperiodic_fit))
        psd_for_peak = (10 ** residual)[mask]
    else:
        psd_for_peak = psd_band
    
    if np.all(np.isnan(psd_for_peak)):
        psd_for_peak = psd_band
    
    peak_idx = np.nanargmax(psd_for_peak)
    peak_freq = float(freqs_band[peak_idx])
    peak_power = float(psd_band[peak_idx])
    
    # Compute peak prominence metrics
    peak_ratio = np.nan
    peak_residual = np.nan
    
    if aperiodic_fit is not None:
        # Find the global index for the peak frequency
        global_peak_idx = np.where(mask)[0][peak_idx]
        aperiodic_at_peak = aperiodic_fit[global_peak_idx]
        
        if np.isfinite(aperiodic_at_peak) and aperiodic_at_peak > 0:
            peak_ratio = float(peak_power / aperiodic_at_peak)
            peak_residual = float(np.log10(peak_power) - np.log10(aperiodic_at_peak))
    
    return peak_freq, peak_power, peak_ratio, peak_residual


def compute_spectral_center(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Compute spectral center of gravity (centroid) within a frequency range.
    
    Uses Δf weighting for non-uniform frequency grids (e.g., log-spaced).
    Formula: Σ(f * P * Δf) / Σ(P * Δf)
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan
    
    center = float(np.nansum(freqs_band * mass) / total_mass)
    return center


def compute_spectral_bandwidth(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Compute spectral bandwidth (standard deviation of frequency distribution).
    
    Uses Δf weighting for non-uniform frequency grids.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan
    
    center = np.nansum(freqs_band * mass) / total_mass
    variance = np.nansum(mass * (freqs_band - center) ** 2) / total_mass
    bandwidth = float(np.sqrt(variance))
    
    return bandwidth


def compute_spectral_edge(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    percentile: float = 0.95,
) -> float:
    """
    Compute spectral edge frequency (frequency below which X% of power lies).
    
    Uses Δf weighting for non-uniform frequency grids.
    
    Parameters
    ----------
    percentile : float
        Cumulative power threshold (default 0.95 = 95%)
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan
    
    cumsum = np.nancumsum(mass) / total_mass
    edge_idx = np.searchsorted(cumsum, percentile)
    edge_idx = min(edge_idx, len(freqs_band) - 1)
    
    return float(freqs_band[edge_idx])


def compute_spectral_entropy(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Compute normalized spectral entropy within a frequency range.
    
    Uses Δf weighting for non-uniform frequency grids.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan

    psd_band = psd[mask]
    freqs_band = freqs[mask]
    if len(psd_band) == 0 or np.all(np.isnan(psd_band)):
        return np.nan

    psd_band = np.maximum(psd_band, 0)
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan

    probs = mass / total_mass
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if probs.size == 0:
        return np.nan

    entropy = -np.sum(probs * np.log(probs))
    norm = np.log(float(probs.size))
    if norm > 0:
        entropy /= norm
    return float(entropy)


def extract_spectral_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract spectral descriptor features including IAF (Individual Alpha Frequency).
    
    Features extracted:
    - Peak frequency per band (IAF for alpha band)
    - Peak power per band
    - Spectral center of gravity
    - Spectral bandwidth
    - Spectral entropy (normalized)
    - Spectral edge frequency (broadband, 95%)
    """
    if not bands:
        return pd.DataFrame(), []
    
    epochs = ctx.epochs
    config = ctx.config
    logger = ctx.logger
    
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("Spectral: No EEG channels available; skipping.")
        return pd.DataFrame(), []
    
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)
    spatial_modes = getattr(ctx, "spatial_modes", ["roi", "global"])
    
    roi_map = {}
    if "roi" in spatial_modes:
        roi_defs = get_roi_definitions(config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)
    
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data(picks=picks)
    n_epochs = data.shape[0]
    n_channels = data.shape[1]

    spec_cfg = config.get("feature_engineering.spectral", {}) if hasattr(config, "get") else {}
    psd_method = str(spec_cfg.get("psd_method", "multitaper")).strip().lower()
    if psd_method not in {"welch", "multitaper"}:
        psd_method = "multitaper"

    fmin_psd = float(spec_cfg.get("fmin", 1.0))
    fmax_psd = float(spec_cfg.get("fmax", min(80.0, float(sfreq) / 2.0 - 0.5)))

    exclude_line = bool(spec_cfg.get("exclude_line_noise", True))
    line_freqs = spec_cfg.get("line_noise_freqs", [50.0])
    try:
        line_freqs = [float(f) for f in line_freqs]
    except Exception:
        line_freqs = [50.0]
    line_width = float(spec_cfg.get("line_noise_width_hz", 1.0))
    n_harm = int(spec_cfg.get("line_noise_harmonics", 3))
    
    # Determine which segments to process
    target_name = getattr(ctx, "name", None)
    configured_segments = spec_cfg.get("segments")
    
    # When a specific time range is targeted (ctx.name is set), the epochs have 
    # already been cropped to that range. Use all available data with that segment name.
    if target_name:
        segments = [target_name]
        # Create a mask covering all data since epochs are pre-cropped
        segment_masks = {target_name: np.ones(data.shape[2], dtype=bool)}
    else:
        segment_masks = get_segment_masks(epochs.times, ctx.windows, config)
        if configured_segments:
            if isinstance(configured_segments, str):
                configured_segments = [configured_segments]
            segments = [s for s in configured_segments if s in segment_masks]
        else:
            segments = list(segment_masks.keys())
    
    if not segments:
        logger.warning("Spectral: No valid segments found; returning empty DataFrame.")
        return pd.DataFrame(), []
    
    # Segment duration validation parameters
    min_segment_sec = float(spec_cfg.get("min_segment_sec", 2.0))
    min_cycles_at_fmin = float(spec_cfg.get("min_cycles_at_fmin", 3.0))

    records = [dict() for _ in range(n_epochs)]

    for segment_name in segments:
        mask = segment_masks.get(segment_name)
        if mask is None or not np.any(mask):
            continue

        seg_data = data[:, :, mask]
        seg_duration_sec = float(seg_data.shape[2]) / float(sfreq)
        
        # Validate minimum segment duration
        if seg_duration_sec < min_segment_sec:
            logger.warning(
                "Spectral: segment '%s' duration (%.2fs) is shorter than min_segment_sec (%.2fs); "
                "skipping to ensure reliable spectral estimation.",
                segment_name, seg_duration_sec, min_segment_sec
            )
            continue
        
        if seg_data.shape[2] < 2:
            continue

        try:
            import mne
            if psd_method == "multitaper":
                psds, freqs = mne.time_frequency.psd_array_multitaper(
                    seg_data,
                    sfreq=float(sfreq),
                    fmin=fmin_psd,
                    fmax=fmax_psd,
                    adaptive=True,
                    normalization="full",
                    verbose=False,
                )
            else:
                n_times = int(seg_data.shape[2])
                n_per_seg = min(int(float(sfreq) * 2.0), n_times)
                psds, freqs = mne.time_frequency.psd_array_welch(
                    seg_data,
                    sfreq=float(sfreq),
                    fmin=fmin_psd,
                    fmax=fmax_psd,
                    n_fft=min(n_times, max(64, n_per_seg)),
                    n_overlap=0,
                    verbose=False,
                )
        except Exception as exc:
            logger.warning("Spectral: PSD computation failed for segment '%s' (%s); skipping.", segment_name, exc)
            continue

        freqs = np.asarray(freqs, dtype=float)
        psds = np.asarray(psds, dtype=float)
        if psds.ndim != 3:
            continue

        freq_keep_mask = np.ones_like(freqs, dtype=bool)
        if exclude_line and freqs.size > 0 and line_width > 0 and n_harm > 0:
            for base in line_freqs:
                if not np.isfinite(base) or base <= 0:
                    continue
                for h in range(1, n_harm + 1):
                    f0 = base * h
                    freq_keep_mask &= ~(
                        (freqs >= (f0 - line_width)) & (freqs <= (f0 + line_width))
                    )

        freqs_use = freqs[freq_keep_mask] if np.any(~freq_keep_mask) else freqs
        psds_use = psds[:, :, freq_keep_mask] if np.any(~freq_keep_mask) else psds

        for ep_idx in range(n_epochs):
            record = records[ep_idx]
            channel_psd = psds_use[ep_idx]

            for band in bands:
                if band not in freq_bands:
                    continue
                fmin, fmax = freq_bands[band]

                # Require enough cycles at the lowest frequency in the band.
                if min_cycles_at_fmin > 0:
                    try:
                        fmin_hz = float(fmin)
                    except Exception:
                        fmin_hz = np.nan
                    if np.isfinite(fmin_hz) and fmin_hz > 0:
                        min_req_sec = float(min_cycles_at_fmin) / fmin_hz
                        if seg_duration_sec < min_req_sec:
                            continue

                if "channels" in spatial_modes:
                    for ch_idx, ch_name in enumerate(ch_names):
                        psd = channel_psd[ch_idx]
                        peak_freq, peak_power, peak_ratio, peak_residual = compute_peak_frequency(
                            psd, freqs_use, fmin, fmax, aperiodic_adjusted=True
                        )
                        center_freq = compute_spectral_center(psd, freqs_use, fmin, fmax)
                        bandwidth = compute_spectral_bandwidth(psd, freqs_use, fmin, fmax)
                        entropy = compute_spectral_entropy(psd, freqs_use, fmin, fmax)

                        record[NamingSchema.build("spectral", segment_name, band, "ch", "peak_freq", channel=ch_name)] = peak_freq
                        record[NamingSchema.build("spectral", segment_name, band, "ch", "peak_power", channel=ch_name)] = peak_power
                        record[NamingSchema.build("spectral", segment_name, band, "ch", "peak_ratio", channel=ch_name)] = peak_ratio
                        record[NamingSchema.build("spectral", segment_name, band, "ch", "peak_residual", channel=ch_name)] = peak_residual
                        record[NamingSchema.build("spectral", segment_name, band, "ch", "center_freq", channel=ch_name)] = center_freq
                        record[NamingSchema.build("spectral", segment_name, band, "ch", "bandwidth", channel=ch_name)] = bandwidth
                        record[NamingSchema.build("spectral", segment_name, band, "ch", "entropy", channel=ch_name)] = entropy

                if "global" in spatial_modes:
                    global_psd = np.nanmean(channel_psd, axis=0)
                    g_peak_freq, g_peak_power, g_peak_ratio, g_peak_residual = compute_peak_frequency(
                        global_psd, freqs_use, fmin, fmax, aperiodic_adjusted=True
                    )
                    g_center = compute_spectral_center(global_psd, freqs_use, fmin, fmax)
                    g_bandwidth = compute_spectral_bandwidth(global_psd, freqs_use, fmin, fmax)
                    g_entropy = compute_spectral_entropy(global_psd, freqs_use, fmin, fmax)

                    record[NamingSchema.build("spectral", segment_name, band, "global", "peak_freq")] = g_peak_freq
                    record[NamingSchema.build("spectral", segment_name, band, "global", "peak_power")] = g_peak_power
                    record[NamingSchema.build("spectral", segment_name, band, "global", "peak_ratio")] = g_peak_ratio
                    record[NamingSchema.build("spectral", segment_name, band, "global", "peak_residual")] = g_peak_residual
                    record[NamingSchema.build("spectral", segment_name, band, "global", "center_freq")] = g_center
                    record[NamingSchema.build("spectral", segment_name, band, "global", "bandwidth")] = g_bandwidth
                    record[NamingSchema.build("spectral", segment_name, band, "global", "entropy")] = g_entropy

                if "roi" in spatial_modes and roi_map:
                    for roi_name, roi_indices in roi_map.items():
                        if not roi_indices:
                            continue
                        roi_psd = np.nanmean(channel_psd[roi_indices], axis=0)
                        r_peak_freq, r_peak_power, r_peak_ratio, r_peak_residual = compute_peak_frequency(
                            roi_psd, freqs_use, fmin, fmax, aperiodic_adjusted=True
                        )
                        r_center = compute_spectral_center(roi_psd, freqs_use, fmin, fmax)
                        r_bandwidth = compute_spectral_bandwidth(roi_psd, freqs_use, fmin, fmax)
                        r_entropy = compute_spectral_entropy(roi_psd, freqs_use, fmin, fmax)

                        record[NamingSchema.build("spectral", segment_name, band, "roi", "peak_freq", channel=roi_name)] = r_peak_freq
                        record[NamingSchema.build("spectral", segment_name, band, "roi", "peak_power", channel=roi_name)] = r_peak_power
                        record[NamingSchema.build("spectral", segment_name, band, "roi", "peak_ratio", channel=roi_name)] = r_peak_ratio
                        record[NamingSchema.build("spectral", segment_name, band, "roi", "peak_residual", channel=roi_name)] = r_peak_residual
                        record[NamingSchema.build("spectral", segment_name, band, "roi", "center_freq", channel=roi_name)] = r_center
                        record[NamingSchema.build("spectral", segment_name, band, "roi", "bandwidth", channel=roi_name)] = r_bandwidth
                        record[NamingSchema.build("spectral", segment_name, band, "roi", "entropy", channel=roi_name)] = r_entropy

            global_psd = np.nanmean(channel_psd, axis=0)
            edge_fmax = float(freqs_use[-1]) if freqs_use.size else (float(sfreq) / 2.0 - 0.5)
            edge_95 = compute_spectral_edge(global_psd, freqs_use, 1.0, edge_fmax, 0.95)
            record[NamingSchema.build("spectral", segment_name, "broadband", "global", "edge_freq_95")] = edge_95

            if "roi" in spatial_modes and roi_map:
                for roi_name, roi_indices in roi_map.items():
                    if not roi_indices:
                        continue
                    roi_psd = np.nanmean(channel_psd[roi_indices], axis=0)
                    roi_edge = compute_spectral_edge(roi_psd, freqs_use, 1.0, edge_fmax, 0.95)
                    record[NamingSchema.build("spectral", segment_name, "broadband", "roi", "edge_freq_95", channel=roi_name)] = roi_edge
    
    if not records:
        return pd.DataFrame(), []
    
    df = pd.DataFrame(records)
    cols = list(df.columns)
    
    logger.info(f"Extracted {len(cols)} spectral features for {n_epochs} epochs")
    
    return df, cols


###################################################################
# PUBLIC API
###################################################################


__all__ = [
    # TFR-based power extraction
    "extract_power_features",
    # Precomputed-based power extraction
    "extract_power_from_precomputed",
    # Spectral descriptor extraction
    "extract_spectral_features",
    "compute_peak_frequency",
    "compute_spectral_center",
    "compute_spectral_bandwidth",
    "compute_spectral_edge",
    "compute_spectral_entropy",
    "remove_aperiodic_component",
]
