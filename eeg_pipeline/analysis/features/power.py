from __future__ import annotations

from typing import List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import EPSILON_PSD
from eeg_pipeline.utils.analysis.tfr import extract_tfr_object
from eeg_pipeline.utils.analysis.windowing import make_mask_for_times

def _prepare_tfr(tfr: Any, config: Any, logger: Any):
    tfr_obj = extract_tfr_object(tfr)
    if tfr_obj is None:
        return None, None, None, None, None
    return tfr_obj, tfr_obj.data, tfr_obj.freqs, tfr_obj.times, tfr_obj.info["ch_names"]

def _build_baseline_arrays(
    baseline_df: Optional[pd.DataFrame],
    channel_names: List[str],
) -> dict:
    if baseline_df is None or baseline_df.empty:
        return {}

    baseline_map: dict = {}
    for col in baseline_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "power" or parsed.get("segment") != "baseline":
            continue
        if parsed.get("scope") != "ch":
            continue
        band = parsed.get("band")
        ch = parsed.get("identifier")
        if not band or not ch:
            continue
        baseline_map.setdefault(band, {})[ch] = baseline_df[col].to_numpy(dtype=float)

    n_epochs = len(baseline_df)
    baseline_arrays = {}
    for band, ch_map in baseline_map.items():
        band_matrix = np.full((n_epochs, len(channel_names)), np.nan)
        for idx, ch in enumerate(channel_names):
            values = ch_map.get(ch)
            if values is not None and len(values) == n_epochs:
                band_matrix[:, idx] = values
        baseline_arrays[band] = band_matrix
    return baseline_arrays

def extract_power_features(
    ctx: Any, # FeatureContext
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract power features for all defined segments (baseline, ramp, active, etc.).
    
    Computes:
    - Raw power for baseline (if available) - internalized for normalization
    - Log-ratio power for active segments (ramp, active, bins) relative to baseline
    - Global mean power per band
    - ROI mean power per band (if spatial_modes includes 'roi')
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
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(ctx.config)
    
    # Get spatial modes from context
    spatial_modes = getattr(ctx, 'spatial_modes', ['roi', 'global'])
    
    # Build ROI map if needed
    roi_map = {}
    if 'roi' in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(ctx.config)
        if roi_defs:
            roi_map = build_roi_map(channel_names, roi_defs)
    
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
    
    baseline_df = getattr(ctx, "baseline_df", None)
    baseline_arrays = _build_baseline_arrays(baseline_df, channel_names)
    require_baseline = bool(ctx.config.get("feature_engineering.power.require_baseline", True))
    if require_baseline and not tfr_already_baselined:
        if baseline_df is None or baseline_df.empty:
            raise ValueError("Power features require baseline_df for log-ratio normalization.")
        if len(baseline_df) != n_epochs:
            raise ValueError(
                "Baseline feature length mismatch for power normalization: "
                f"{len(baseline_df)} vs {n_epochs}"
            )
            
    # Determine segments to process: ONLY the one requested in context
    # unless no name is provided, in which case we fall back to 'full'
    ctx_name = getattr(ctx, "name", None)
    if ctx_name:
        segments_to_process = [ctx_name]
    else:
        segments_to_process = ["full"]

    ctx.logger.info(f"Computing power features for segment: {segments_to_process[0]}")
    
    output_data = {}
    
    for seg_name in segments_to_process:
        mask = _get_tfr_mask(seg_name)
        
        if not np.any(mask):
            continue
            
        for band, fmask in band_indices.items():
            try:
                d = tfr_data[:, :, fmask, :][:, :, :, mask]
                # Average over time, then compute a frequency-weighted mean.
                # TFR frequencies are often log-spaced; weighting by Δf avoids biasing
                # bands based on bin density.
                p_ft = np.nanmean(d, axis=3)  # (n_epochs, n_channels, n_freqs_in_band)
                band_freqs = np.asarray(freqs[fmask], dtype=float)
                if band_freqs.size >= 2 and np.all(np.isfinite(band_freqs)):
                    w = np.gradient(band_freqs).astype(float)
                    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
                else:
                    w = np.ones((band_freqs.size,), dtype=float)

                w3 = w[None, None, :]
                finite = np.isfinite(p_ft) & np.isfinite(w3)
                num = np.nansum(np.where(finite, p_ft * w3, 0.0), axis=2)
                den = np.nansum(np.where(finite, w3, 0.0), axis=2)
                segment_power = np.where(den > 0, num / den, np.nan)  # (n_epochs, n_channels)

                if tfr_already_baselined:
                    # Baselined at TFR level; preserve sign and scale.
                    val_matrix = segment_power
                    stat_name = "baselined"
                else:
                    eps_psd = float(ctx.config.get("feature_engineering.constants.epsilon_psd", EPSILON_PSD))
                    raw_power = np.maximum(segment_power, eps_psd)
                    base = baseline_arrays.get(band)
                    if base is None or not np.isfinite(base).any():
                        if require_baseline:
                            raise ValueError(
                                f"Missing baseline power for band '{band}'; "
                                "set feature_engineering.power.require_baseline=false to allow raw log power."
                            )
                        val_matrix = np.log10(raw_power)
                        stat_name = "log10raw"
                    else:
                        denom = base.copy()
                        denom[denom <= 0] = np.nan
                        val_matrix = np.log10(raw_power / denom)
                        stat_name = "logratio"

                # Per-channel (only if 'channels' in spatial_modes)
                if 'channels' in spatial_modes:
                    for ch_idx, ch in enumerate(channel_names):
                        col = NamingSchema.build("power", seg_name, band, "ch", stat_name, channel=ch)
                        output_data[col] = val_matrix[:, ch_idx]
                    
                # Global Mean (only if 'global' in spatial_modes)
                if 'global' in spatial_modes:
                    global_mean = np.nanmean(val_matrix, axis=1) # (n_epochs,)
                    col_global = NamingSchema.build("power", seg_name, band, "global", stat_name + "_mean")
                    output_data[col_global] = global_mean
                
                # ROI Mean (only if 'roi' in spatial_modes)
                if 'roi' in spatial_modes and roi_map:
                    for roi_name, ch_indices in roi_map.items():
                        if len(ch_indices) > 0:
                            roi_mean = np.nanmean(val_matrix[:, ch_indices], axis=1)
                            col_roi = NamingSchema.build("power", seg_name, band, "roi", stat_name + "_mean", channel=roi_name)
                            output_data[col_roi] = roi_mean
                            
            except Exception as e:
                ctx.logger.error(f"Error computing power features for {seg_name} {band}: {e}")

    if not output_data:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(output_data)
    return df, list(df.columns)
