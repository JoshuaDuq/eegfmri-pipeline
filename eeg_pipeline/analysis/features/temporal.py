"""
Temporal Feature Extraction
============================

Extract features across time windows (sliding bins).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.tfr import process_temporal_bin, extract_tfr_object
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.domain.features.naming import NamingSchema

def extract_temporal_features(
    ctx: Any, # FeatureContext
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract power features across multiple time windows (bins).
    """
    if not bands:
        return pd.DataFrame(), []

    tfr = ctx.results.get("tfr")
    if tfr is None:
        return pd.DataFrame(), []

    tfr_obj = extract_tfr_object(tfr)
    if tfr_obj is None:
        return pd.DataFrame(), []

    tfr_data = tfr_obj.data
    freqs = tfr_obj.freqs
    times = tfr_obj.times
    channel_names = tfr_obj.info["ch_names"]
    
    freq_bands = get_frequency_bands(ctx.config)
    
    # Get windows from context
    if ctx.windows is None:
        ctx.logger.warning("No window specification found; cannot extract temporal features.")
        return pd.DataFrame(), []
        
    # Use the ranges dict from TimeWindows (not metadata)
    segments: List[str] = []
    if getattr(ctx.windows, "name", None):
        if ctx.windows.name in ctx.windows.ranges:
            segments = [ctx.windows.name]
    else:
        # Collect all available time windows that are not 'baseline'
        segments = [name for name in ctx.windows.ranges.keys() if name != "baseline"]
    
    # Also include built-in active_range if no custom segments found
    if not segments and ctx.windows.active_range is not None:
        ar = ctx.windows.active_range
        if np.isfinite(ar[0]) and np.isfinite(ar[1]):
            segments = ["active"]
    
    spatial_modes = getattr(ctx, "spatial_modes", ["roi", "global"])
    if "channels" not in spatial_modes and "roi" not in spatial_modes and "global" not in spatial_modes:
        return pd.DataFrame(), []

    roi_map = {}
    if "roi" in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(ctx.config)
        if roi_defs:
            roi_map = build_roi_map(channel_names, roi_defs)

    output_data = {}
    all_cols = []
    
    for seg_name in segments:
        # Get the range for this segment
        if seg_name == "active":
            seg_range = ctx.windows.active_range
        elif seg_name == "baseline":
            seg_range = ctx.windows.baseline_range
        else:
            seg_range = ctx.windows.ranges.get(seg_name)
        
        if seg_range is None:
            continue
        if not (np.isfinite(seg_range[0]) and np.isfinite(seg_range[1])):
            continue
            
        tstart, tend = float(seg_range[0]), float(seg_range[1])
        
        for band in bands:
            if band not in freq_bands:
                continue
            fmin, fmax = freq_bands[band]
            
            res = process_temporal_bin(
                tfr_data, freqs, times, channel_names,
                band, fmin, fmax, tstart, tend, seg_name,
                logger=ctx.logger
            )
            
            if res:
                data, cols = res

                # Per-channel
                if "channels" in spatial_modes:
                    for i, col in enumerate(cols):
                        output_data[col] = data[:, i]
                    all_cols.extend(cols)

                # ROI mean
                if "roi" in spatial_modes and roi_map:
                    for roi_name, ch_indices in roi_map.items():
                        if not ch_indices:
                            continue
                        roi_vals = np.nanmean(data[:, ch_indices], axis=1)
                        col = NamingSchema.build("power", seg_name, band, "roi", "mean", channel=roi_name)
                        output_data[col] = roi_vals
                        all_cols.append(col)

                # Global mean
                if "global" in spatial_modes:
                    global_vals = np.nanmean(data, axis=1)
                    col = NamingSchema.build("power", seg_name, band, "global", "mean")
                    output_data[col] = global_vals
                    all_cols.append(col)
                
    if not output_data:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(output_data)
    return df, all_cols
