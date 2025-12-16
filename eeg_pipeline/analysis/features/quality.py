"""
Feature Quality Assessment (Signal QC)
=======================================

Quality metrics for evaluating signal integrity per trial:
- Variance
- SNR estimates (Low/High frequency ratio)
- Muscle artifact ratio (High Gamma / Total)
- Peak-to-Peak Amplitude
- Finite Fraction (Missing data)

Computed on 'baseline' and 'plateau' windows.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import get_feature_constant

# --- Helpers ---

def _compute_signal_metrics(data, sfreq):
    # data: (n_ch, n_times)
    res = {}
    
    # Global metrics (mean across channels) or per-channel?
    # Usually QC is per-channel, then aggregated.
    
    # 1. Variance
    var = np.var(data, axis=1) # (n_ch,)
    res["variance"] = var
    
    # 2. PtP
    ptp = np.ptp(data, axis=1)
    res["ptp"] = ptp
    
    # 3. Finite Fraction
    finite = np.mean(np.isfinite(data), axis=1)
    res["finite"] = finite
    
    # Spectral metrics (Simple FFT based for speed)
    # We can use Welch on short segments? Or just filter?
    # Filter is slow per trial.
    # Welch is better.
    
    n_times = data.shape[1]
    if n_times < 10: 
        # Too short
        res["snr"] = np.full(data.shape[0], np.nan)
        res["muscle"] = np.full(data.shape[0], np.nan)
        return res
        
    try:
        # Use simple periodogram or welch
        psds, freqs = mne.time_frequency.psd_array_welch(
            data, sfreq, fmin=1, fmax=100, n_fft=min(n_times, 256), verbose=False
        )
        # psds: (n_ch, n_freqs)
        
        # SNR: Power(1-30) / Power(50+)
        mask_sig = (freqs >= 1) & (freqs <= 30)
        mask_noise = (freqs >= 50)
        
        pow_sig = np.sum(psds[:, mask_sig], axis=1)
        pow_noise = np.sum(psds[:, mask_noise], axis=1)
        
        snr_db = 10 * np.log10(pow_sig / (pow_noise + 1e-12))
        res["snr"] = snr_db
        
        # Muscle: Power(30-100) / Total
        mask_mus = (freqs >= 30) & (freqs <= 100)
        pow_mus = np.sum(psds[:, mask_mus], axis=1)
        total = np.sum(psds, axis=1)
        
        mus_ratio = pow_mus / (total + 1e-12)
        res["muscle"] = mus_ratio
        
    except Exception:
        res["snr"] = np.full(data.shape[0], np.nan)
        res["muscle"] = np.full(data.shape[0], np.nan)
        
    return res

# --- Main API ---

def extract_quality_features(
    ctx: Any # FeatureContext
) -> Tuple[pd.DataFrame, List[str]]:
    
    epochs = ctx.epochs
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0: return pd.DataFrame(), []
    
    full_data = epochs.get_data(picks=picks) # (n_epochs, n_ch, n_times)
    sfreq = epochs.info["sfreq"]
    
    results = {}
    n_epochs = len(full_data)
    
    # Segments
    segments = ["baseline", "plateau"]
    # Check if window exists
    
    for seg in segments:
        mask = ctx.windows.get_mask(seg)
        if not np.any(mask): continue
        
        data_seg = full_data[..., mask] # (n_epochs, n_ch, n_times)
        
        # Iterate epochs
        for e in range(n_epochs):
            ep_data = data_seg[e]
            metrics = _compute_signal_metrics(ep_data, sfreq)
            
            # metrics keys: variance, ptp, finite, snr, muscle (arrays of shape n_ch)
            
            for k, vals in metrics.items():
                # Per channel
                for i, ch in enumerate(ch_names):
                    col = NamingSchema.build("quality", seg, "broadband", "ch", k, channel=ch)
                    if col not in results: results[col] = [np.nan]*n_epochs
                    results[col][e] = vals[i]
                
                # Global
                global_val = np.nanmean(vals)
                col_g = NamingSchema.build("quality", seg, "broadband", "global", k)
                if col_g not in results: results[col_g] = [np.nan]*n_epochs
                results[col_g][e] = global_val

    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)


def generate_quality_report(
    df: pd.DataFrame,
    subject_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate quality report from precomputed features DataFrame.
    
    Args:
        df: DataFrame with feature columns
        subject_col: Optional column name for subject identifier
        
    Returns:
        Dictionary with quality metrics summary
    """
    if df.empty:
        return {"status": "empty", "n_rows": 0, "n_features": 0}
    
    feature_cols = [c for c in df.columns if c not in [subject_col, "epoch", "trial", "condition"]]
    
    report = {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "missing_fraction": {},
        "constant_features": [],
        "high_variance_features": [],
        "summary_stats": {},
    }
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        n_missing = series.isna().sum()
        n_total = len(series)
        
        report["missing_fraction"][col] = float(n_missing / n_total) if n_total > 0 else 1.0
        
        valid = series.dropna()
        if len(valid) > 0:
            std = float(valid.std())
            if std == 0:
                report["constant_features"].append(col)
            elif std > valid.mean() * 10 if valid.mean() != 0 else std > 100:
                report["high_variance_features"].append(col)
    
    report["n_constant"] = len(report["constant_features"])
    report["n_high_variance"] = len(report["high_variance_features"])
    report["mean_missing_fraction"] = float(np.mean(list(report["missing_fraction"].values()))) if report["missing_fraction"] else 0.0
    
    return report


def compute_trial_quality_metrics(
    epochs: mne.Epochs,
    config: Any = None,
    logger: Any = None,
) -> pd.DataFrame:
    """Compute trial-level quality metrics from epochs.
    
    This is a standalone wrapper for use in pipelines that don't have
    a full FeatureContext. For full feature extraction, use extract_quality_features.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs
    config : Any
        Configuration object
    logger : Any
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Quality metrics per trial
    """
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        return pd.DataFrame()
    
    full_data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]
    n_epochs = len(full_data)
    
    results = {}
    
    for e in range(n_epochs):
        ep_data = full_data[e]
        metrics = _compute_signal_metrics(ep_data, sfreq)
        
        for k, vals in metrics.items():
            global_val = np.nanmean(vals)
            col = f"quality_global_{k}"
            if col not in results:
                results[col] = [np.nan] * n_epochs
            results[col][e] = global_val
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)
