from __future__ import annotations

from typing import Optional, List, Tuple, Any
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.tfr import (
    time_mask,
    freq_mask,
    extract_tfr_object,
    extract_band_power,
    process_temporal_bin,
)
from eeg_pipeline.utils.config.loader import (
    get_config_value,
    get_frequency_bands,
    parse_temporal_bin_config,
)


###################################################################
# Power Feature Extraction
###################################################################


def _prepare_tfr_power_inputs(tfr: Any, config: Any, logger: Any) -> Optional[Tuple[Any, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    tfr_obj = extract_tfr_object(tfr)
    if tfr_obj is None:
        return None

    tfr_data = tfr_obj.data
    if tfr_data.ndim != 4:
        raise RuntimeError("TFR data must have 4D shape (epochs, channels, freqs, times)")

    freqs = tfr_obj.freqs
    times = tfr_obj.times
    channel_names = tfr_obj.info["ch_names"]
    
    return tfr_obj, tfr_data, freqs, times, channel_names


def extract_baseline_power_features(
    tfr: Any, bands: List[str], baseline_window: Tuple[float, float], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    if not bands:
        return pd.DataFrame(), []
    
    result = _prepare_tfr_power_inputs(tfr, config, logger)
    if result is None:
        return pd.DataFrame(), []
    
    tfr_obj, tfr_data, freqs, times, channel_names = result
    
    baseline_start, baseline_end = baseline_window
    baseline_mask = time_mask(times, baseline_start, baseline_end)
    if not np.any(baseline_mask):
        logger.warning(f"No time points in baseline window ({baseline_start}, {baseline_end})")
        return pd.DataFrame(), []
    
    frequency_bands = get_frequency_bands(config)
    feature_arrays = []
    column_names = []
    
    for band in bands:
        if band not in frequency_bands:
            continue
        
        fmin, fmax = frequency_bands[band]
        band_power = extract_band_power(tfr_data, freqs, fmin, fmax, baseline_mask)
        if band_power is None:
            continue
        
        feature_arrays.append(band_power)
        column_names.extend([f"baseline_{band}_{ch}" for ch in channel_names])

    if not feature_arrays:
        return pd.DataFrame(), []

    return pd.DataFrame(np.concatenate(feature_arrays, axis=1), columns=column_names), column_names


def extract_band_power_features(
    tfr: Any, bands: List[str], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    if not bands:
        return pd.DataFrame(), []
    
    result = _prepare_tfr_power_inputs(tfr, config, logger)
    if result is None:
        return pd.DataFrame(), []
    
    tfr_obj, tfr_data, freqs, times, channel_names = result
    
    frequency_bands = get_frequency_bands(config)
    temporal_bins = get_config_value(config, "feature_engineering.features.temporal_bins", [])
    feature_arrays = []
    column_names = []
    
    for band in bands:
        if band not in frequency_bands:
            logger.warning(f"Band '{band}' not defined in config; skipping")
            continue
        
        fmin, fmax = frequency_bands[band]
        freq_mask_indices = freq_mask(freqs, fmin, fmax)
        if not np.any(freq_mask_indices):
            logger.warning(f"TFR freqs contain no points in band '{band}' ({fmin}-{fmax} Hz); skipping")
            continue
        
        for bin_config in temporal_bins:
            bin_params = parse_temporal_bin_config(bin_config)
            if bin_params is None:
                logger.warning(f"Invalid temporal bin configuration: {bin_config}; skipping")
                continue
            
            time_start, time_end, time_label = bin_params
            result = process_temporal_bin(
                tfr_data, freqs, times, channel_names,
                band, fmin, fmax, time_start, time_end, time_label, logger
            )
            if result is not None:
                band_power, cols = result
                feature_arrays.append(band_power)
                column_names.extend(cols)

    if not feature_arrays:
        return pd.DataFrame(), []

    return pd.DataFrame(np.concatenate(feature_arrays, axis=1), columns=column_names), column_names

