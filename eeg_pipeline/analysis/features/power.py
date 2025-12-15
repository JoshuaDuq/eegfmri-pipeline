from __future__ import annotations

from typing import Optional, List, Tuple, Any
import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.utils.analysis.tfr import extract_tfr_object
from eeg_pipeline.utils.analysis.windowing import make_mask_for_times

from eeg_pipeline.analysis.features.precomputed.asymmetry import extract_asymmetry_from_precomputed
from eeg_pipeline.analysis.features.precomputed.spectral import (
    extract_segment_power_from_precomputed,
    extract_spectral_extras_from_precomputed,
)

def _prepare_tfr(tfr: Any, config: Any, logger: Any):
    tfr_obj = extract_tfr_object(tfr)
    if tfr_obj is None:
        return None, None, None, None, None
    return tfr_obj, tfr_obj.data, tfr_obj.freqs, tfr_obj.times, tfr_obj.info["ch_names"]

def extract_power_features(
    ctx: Any, # FeatureContext
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract power features for all defined segments (baseline, ramp, plateau, etc.).
    
    Computes:
    - Raw power for baseline (if available) - internalized for normalization
    - Log-ratio power for active segments (ramp, plateau, bins) relative to baseline
    - Global mean power per band
    """
    if not bands:
        return pd.DataFrame(), []

    tfr = ctx.results.get("tfr")
    if tfr is None:
        return pd.DataFrame(), []

    tfr_obj, tfr_data, freqs, times, channel_names = _prepare_tfr(tfr, ctx.config, ctx.logger)
    if tfr_data is None:
        return pd.DataFrame(), []

    tfr_comment = getattr(tfr_obj, "comment", None)
    tfr_already_baselined = isinstance(tfr_comment, str) and ("BASELINED:" in tfr_comment)

    from eeg_pipeline.utils.config.loader import get_frequency_bands
    freq_bands = get_frequency_bands(ctx.config)
    
    # Pre-calculate band masks and indices to save time
    band_indices = {}
    for band in bands:
        if band in freq_bands:
            fmin, fmax = freq_bands[band]
            mask = (freqs >= fmin) & (freqs <= fmax)
            if np.any(mask):
                band_indices[band] = mask

    n_epochs = len(tfr_data)
    
    def _get_tfr_mask(segment_name):
        return make_mask_for_times(ctx.windows, segment_name, times)

    # Calculate Baseline Power per band/channel first (for normalization)
    baseline_powers = {}
    baseline_window = _get_tfr_mask("baseline")
    has_baseline = np.any(baseline_window)
    
    if has_baseline:
        for band, fmask in band_indices.items():
            # Mean over freq, Mean over time
            # tfr_data: (epochs, channels, freqs, times)
            d = tfr_data[:, :, fmask, :][:, :, :, baseline_window]
            # Average over time first (last axis) -> (epochs, channels, freqs)
            # Then average over freq (axis 2) -> (epochs, channels)
            # Note: TFR is usually Power values already.
            p_base = np.nanmean(np.nanmean(d, axis=3), axis=2)
            baseline_powers[band] = p_base
            
    # Iterate over all defined windows in Spec
    segments_to_process = []
    # Core segments including baseline
    for seg in ["baseline", "ramp", "plateau"]:
        if seg in ctx.windows.masks:
             segments_to_process.append(seg)
    # Bins
    for name in ctx.windows.masks:
        if name.startswith("coarse_") or name.startswith("fine_"):
            segments_to_process.append(name)
            
    ctx.logger.info(f"Computing power features for {len(segments_to_process)} segments")
    
    output_data = {}
    
    for seg_name in segments_to_process:
        mask = _get_tfr_mask(seg_name)
        
        if not np.any(mask):
            continue
            
        for band, fmask in band_indices.items():
            try:
                d = tfr_data[:, :, fmask, :][:, :, :, mask]
                
                raw_power = np.nanmean(np.nanmean(d, axis=3), axis=2) # (n_epochs, n_channels)
                eps_psd = float(ctx.config.get("feature_engineering.constants.epsilon_psd", 1e-20))
                raw_power = np.maximum(raw_power, eps_psd)

                if tfr_already_baselined:
                    # Baseline normalization has already been applied at the TFR level.
                    # Avoid computing a second log-ratio against a separately-estimated baseline.
                    val_matrix = np.nanmean(np.nanmean(d, axis=3), axis=2)
                    stat_name = "baselined"
                elif band in baseline_powers:
                    base = baseline_powers[band]
                    denom = base.copy()
                    denom[denom == 0] = np.nan
                    # Log-ratio
                    val_matrix = np.log10(raw_power / denom)
                    stat_name = "logratio"
                else:
                    # Fallback to raw log10 if no baseline
                    val_matrix = np.log10(raw_power)
                    stat_name = "log10raw"

                # Per-channel
                for ch_idx, ch in enumerate(channel_names):
                    col = NamingSchema.build("power", seg_name, band, "ch", stat_name, channel=ch)
                    output_data[col] = val_matrix[:, ch_idx]
                    
                # Global Mean
                global_mean = np.nanmean(val_matrix, axis=1) # (n_epochs,)
                col_global = NamingSchema.build("power", seg_name, band, "global", stat_name + "_mean")
                output_data[col_global] = global_mean
            except Exception as e:
                ctx.logger.error(f"Error computing power features for {seg_name} {band}: {e}")

    if not output_data:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(output_data)
    return df, list(df.columns)
