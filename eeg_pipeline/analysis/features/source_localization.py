"""
Source Localization Feature Extraction
======================================

Extracts ROI-specific features using source localization methods:
- LCMV beamformer
- eLORETA inverse solution

Provides source-space power, connectivity, and time-course features
for anatomically-defined ROIs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    import mne


###################################################################
# Forward Model Setup
###################################################################


def _setup_forward_model(
    info: Any,
    subjects_dir: Optional[str] = None,
    subject: str = "fsaverage",
    spacing: str = "oct6",
    conductivity: Tuple[float, ...] = (0.3, 0.006, 0.3),
    mindist: float = 5.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Any, Any]:
    """
    Set up forward model for source localization.
    
    Uses fsaverage template if no subject-specific MRI is available.
    
    Returns
    -------
    fwd : mne.Forward
        Forward solution
    src : mne.SourceSpaces
        Source space
    bem : str or mne.bem.ConductorModel
        BEM solution
    """
    import mne
    from mne.datasets import fetch_fsaverage
    
    if logger:
        logger.info(f"Setting up forward model using {subject} template")
    
    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = str(fs_dir).replace("/fsaverage", "")
    
    src = mne.setup_source_space(
        subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    
    bem_path = mne.datasets.fetch_fsaverage(verbose=False)
    bem = f"{bem_path}/bem/fsaverage-5120-5120-5120-bem-sol.fif"
    
    fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",
        src=src,
        bem=bem,
        eeg=True,
        mindist=mindist,
        verbose=False,
    )
    
    if logger:
        logger.info(f"Forward model: {fwd['nsource']} sources")
    
    return fwd, src, bem


def _setup_volume_source_space(
    info: Any,
    roi_coords: Optional[Dict[str, np.ndarray]] = None,
    pos: float = 7.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Any]:
    """
    Set up volume source space for ROI-based analysis.
    
    Parameters
    ----------
    info : mne.Info
        Measurement info
    roi_coords : dict, optional
        Dictionary mapping ROI names to MNI coordinates (N x 3 array)
    pos : float
        Grid spacing in mm (used if roi_coords is None)
        
    Returns
    -------
    fwd : mne.Forward
        Forward solution
    src : mne.SourceSpaces
        Volume source space
    """
    import mne
    from mne.datasets import fetch_fsaverage
    
    if logger:
        logger.info("Setting up volume source space")
    
    fs_dir = fetch_fsaverage(verbose=False)
    
    if roi_coords is not None:
        all_coords = []
        roi_indices = {}
        idx = 0
        for roi_name, coords in roi_coords.items():
            coords = np.atleast_2d(coords)
            roi_indices[roi_name] = list(range(idx, idx + len(coords)))
            all_coords.append(coords)
            idx += len(coords)
        
        all_coords = np.vstack(all_coords) / 1000.0
        
        pos_dict = {
            "rr": all_coords,
            "nn": np.tile([0.0, 0.0, 1.0], (len(all_coords), 1)),
        }
        src = mne.setup_volume_source_space(
            "fsaverage",
            pos=pos_dict,
            verbose=False,
        )
    else:
        src = mne.setup_volume_source_space(
            "fsaverage",
            pos=pos,
            subjects_dir=str(fs_dir).replace("/fsaverage", ""),
            verbose=False,
        )
        roi_indices = None
    
    bem = f"{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif"
    
    fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",
        src=src,
        bem=bem,
        eeg=True,
        verbose=False,
    )
    
    if logger:
        logger.info(f"Volume source space: {fwd['nsource']} sources")
    
    return fwd, src, roi_indices


###################################################################
# LCMV Beamformer
###################################################################


def _compute_lcmv_source_estimates(
    epochs: "mne.Epochs",
    fwd: Any,
    data_cov: Optional[Any] = None,
    noise_cov: Optional[Any] = None,
    reg: float = 0.05,
    pick_ori: str = "max-power",
    weight_norm: str = "unit-noise-gain",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], Any]:
    """
    Compute LCMV beamformer source estimates for epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
    fwd : mne.Forward
        Forward solution
    data_cov : mne.Covariance, optional
        Data covariance (computed from epochs if None)
    noise_cov : mne.Covariance, optional
        Noise covariance for whitening
    reg : float
        Regularization parameter
    pick_ori : str
        Orientation picking strategy
    weight_norm : str
        Weight normalization method
        
    Returns
    -------
    stcs : list of mne.SourceEstimate
        Source estimates for each epoch
    filters : mne.beamformer.Beamformer
        LCMV spatial filters
    """
    import mne
    from mne.beamformer import make_lcmv, apply_lcmv_epochs
    
    if logger:
        logger.info("Computing LCMV beamformer source estimates")
    
    if data_cov is None:
        data_cov = mne.compute_covariance(
            epochs,
            method="empirical",
            keep_sample_mean=False,
            verbose=False,
        )
    
    filters = make_lcmv(
        epochs.info,
        fwd,
        data_cov,
        reg=reg,
        noise_cov=noise_cov,
        pick_ori=pick_ori,
        weight_norm=weight_norm,
        rank="info",
        verbose=False,
    )
    
    stcs = apply_lcmv_epochs(epochs, filters, verbose=False)
    
    if logger:
        logger.info(f"LCMV: {len(stcs)} epochs, {stcs[0].data.shape[0]} sources")
    
    return stcs, filters


###################################################################
# eLORETA Inverse Solution
###################################################################


def _compute_eloreta_source_estimates(
    epochs: "mne.Epochs",
    fwd: Any,
    noise_cov: Optional[Any] = None,
    loose: float = 0.2,
    depth: float = 0.8,
    snr: float = 3.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], Any]:
    """
    Compute eLORETA inverse solution source estimates.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
    fwd : mne.Forward
        Forward solution
    noise_cov : mne.Covariance, optional
        Noise covariance (identity if None)
    loose : float
        Loose orientation constraint (0-1)
    depth : float
        Depth weighting (0-1)
    snr : float
        Assumed SNR for regularization
        
    Returns
    -------
    stcs : list of mne.SourceEstimate
        Source estimates for each epoch
    inv : mne.minimum_norm.InverseOperator
        Inverse operator
    """
    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
    
    if logger:
        logger.info("Computing eLORETA source estimates")
    
    if noise_cov is None:
        noise_cov = mne.make_ad_hoc_cov(epochs.info, verbose=False)
    
    inv = make_inverse_operator(
        epochs.info,
        fwd,
        noise_cov,
        loose=loose,
        depth=depth,
        verbose=False,
    )
    
    lambda2 = 1.0 / snr ** 2
    
    stcs = apply_inverse_epochs(
        epochs,
        inv,
        lambda2=lambda2,
        method="eLORETA",
        pick_ori="normal",
        verbose=False,
    )
    
    if logger:
        logger.info(f"eLORETA: {len(stcs)} epochs, {stcs[0].data.shape[0]} sources")
    
    return stcs, inv


###################################################################
# ROI Feature Extraction
###################################################################


def _extract_roi_timecourses(
    stcs: List[Any],
    labels: List[Any],
    mode: str = "mean_flip",
) -> np.ndarray:
    """
    Extract ROI time courses from source estimates.
    
    Parameters
    ----------
    stcs : list of mne.SourceEstimate
        Source estimates
    labels : list of mne.Label
        ROI labels
    mode : str
        Extraction mode: 'mean', 'mean_flip', 'pca_flip', 'max'
        
    Returns
    -------
    roi_data : ndarray, shape (n_epochs, n_rois, n_times)
        ROI time courses
    """
    import mne
    
    n_epochs = len(stcs)
    n_rois = len(labels)
    n_times = stcs[0].data.shape[1]
    
    roi_data = np.zeros((n_epochs, n_rois, n_times))
    
    for epoch_idx, stc in enumerate(stcs):
        for roi_idx, label in enumerate(labels):
            try:
                tc = stc.extract_label_time_course(
                    label,
                    stcs[0].subject,
                    mode=mode,
                )
                roi_data[epoch_idx, roi_idx, :] = tc.squeeze()
            except Exception:
                roi_data[epoch_idx, roi_idx, :] = np.nan
    
    return roi_data


def _compute_roi_power(
    roi_data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Compute band power for ROI time courses.
    
    Parameters
    ----------
    roi_data : ndarray, shape (n_epochs, n_rois, n_times)
        ROI time courses
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency band limits
        
    Returns
    -------
    power : ndarray, shape (n_epochs, n_rois)
        Band power per epoch and ROI
    """
    from scipy.signal import welch
    
    n_epochs, n_rois, n_times = roi_data.shape
    power = np.zeros((n_epochs, n_rois))
    
    nperseg = min(n_times, int(sfreq * 2))
    
    for epoch_idx in range(n_epochs):
        for roi_idx in range(n_rois):
            tc = roi_data[epoch_idx, roi_idx, :]
            if np.any(np.isnan(tc)):
                power[epoch_idx, roi_idx] = np.nan
                continue
            
            freqs, psd = welch(tc, fs=sfreq, nperseg=nperseg)
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            power[epoch_idx, roi_idx] = np.mean(psd[freq_mask])
    
    return power


def _compute_roi_envelope(
    roi_data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Compute band-limited amplitude envelope for ROI time courses.
    
    Parameters
    ----------
    roi_data : ndarray, shape (n_epochs, n_rois, n_times)
        ROI time courses
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency band limits
        
    Returns
    -------
    envelope : ndarray, shape (n_epochs, n_rois, n_times)
        Amplitude envelope
    """
    from scipy.signal import butter, filtfilt, hilbert
    
    n_epochs, n_rois, n_times = roi_data.shape
    envelope = np.zeros_like(roi_data)
    
    nyq = sfreq / 2.0
    low = fmin / nyq
    high = min(fmax / nyq, 0.99)
    
    if low >= high or low <= 0:
        return np.abs(roi_data)
    
    try:
        b, a = butter(4, [low, high], btype="band")
    except ValueError:
        return np.abs(roi_data)
    
    for epoch_idx in range(n_epochs):
        for roi_idx in range(n_rois):
            tc = roi_data[epoch_idx, roi_idx, :]
            if np.any(np.isnan(tc)) or np.std(tc) < 1e-12:
                envelope[epoch_idx, roi_idx, :] = np.nan
                continue
            
            try:
                filtered = filtfilt(b, a, tc)
                analytic = hilbert(filtered)
                envelope[epoch_idx, roi_idx, :] = np.abs(analytic)
            except Exception:
                envelope[epoch_idx, roi_idx, :] = np.nan
    
    return envelope


###################################################################
# Main Extraction Functions
###################################################################


def extract_source_localization_features(
    ctx: Any,
    bands: List[str],
    method: str = "lcmv",
    roi_labels: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract source-localized ROI features from epochs.
    
    Parameters
    ----------
    ctx : FeatureContext
        Feature extraction context with epochs and config
    bands : list of str
        Frequency bands to analyze
    method : str
        Source localization method: 'lcmv' or 'eloreta'
    roi_labels : list of str, optional
        ROI label names to extract (uses aparc if None)
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    features_df : pd.DataFrame
        Source-localized features per epoch
    feature_cols : list of str
        Feature column names
    """
    import mne
    
    epochs = ctx.epochs
    config = ctx.config
    logger = getattr(ctx, "logger", None) or logging.getLogger(__name__)
    
    n_epochs = len(epochs)
    if n_epochs < 2:
        logger.warning("Source localization requires at least 2 epochs")
        return pd.DataFrame(), []
    
    sfreq = epochs.info["sfreq"]
    freq_bands = ctx.freq_bands if hasattr(ctx, "freq_bands") else {}
    
    if logger:
        logger.info(f"Extracting source-localized features using {method.upper()}")
    
    fwd, src, _ = _setup_forward_model(epochs.info, logger=logger)
    
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc="aparc",
        subjects_dir=None,
        verbose=False,
    )
    
    labels = [l for l in labels if "unknown" not in l.name.lower()]
    
    if roi_labels is not None:
        labels = [l for l in labels if any(r in l.name for r in roi_labels)]
    
    if not labels:
        logger.warning("No ROI labels found for source localization")
        return pd.DataFrame(), []
    
    label_names = [l.name for l in labels]
    
    if logger:
        logger.info(f"Using {len(labels)} ROI labels")
    
    if method.lower() == "lcmv":
        stcs, _ = _compute_lcmv_source_estimates(epochs, fwd, logger=logger)
    elif method.lower() in ("eloreta", "eloreta"):
        stcs, _ = _compute_eloreta_source_estimates(epochs, fwd, logger=logger)
    else:
        raise ValueError(f"Unknown source localization method: {method}")
    
    roi_data = _extract_roi_timecourses(stcs, labels, mode="mean_flip")
    
    records = [{} for _ in range(n_epochs)]
    feature_cols = []
    
    for band in bands:
        if band not in freq_bands:
            continue
        
        fmin, fmax = freq_bands[band]
        
        power = _compute_roi_power(roi_data, sfreq, fmin, fmax)
        
        for roi_idx, roi_name in enumerate(label_names):
            safe_name = roi_name.replace("-", "_").replace(" ", "_")
            col_name = f"src_{method}_{band}_{safe_name}_power"
            feature_cols.append(col_name)
            
            for epoch_idx in range(n_epochs):
                records[epoch_idx][col_name] = power[epoch_idx, roi_idx]
        
        global_power = np.nanmean(power, axis=1)
        col_name = f"src_{method}_{band}_global_power"
        feature_cols.append(col_name)
        for epoch_idx in range(n_epochs):
            records[epoch_idx][col_name] = global_power[epoch_idx]
        
        envelope = _compute_roi_envelope(roi_data, sfreq, fmin, fmax)
        mean_env = np.nanmean(envelope, axis=2)
        
        for roi_idx, roi_name in enumerate(label_names):
            safe_name = roi_name.replace("-", "_").replace(" ", "_")
            col_name = f"src_{method}_{band}_{safe_name}_envelope"
            feature_cols.append(col_name)
            
            for epoch_idx in range(n_epochs):
                records[epoch_idx][col_name] = mean_env[epoch_idx, roi_idx]
    
    features_df = pd.DataFrame(records)
    feature_cols = list(features_df.columns)
    
    if logger:
        logger.info(f"Source localization: {len(feature_cols)} features extracted")
    
    return features_df, feature_cols


def extract_source_connectivity_features(
    ctx: Any,
    bands: List[str],
    method: str = "lcmv",
    connectivity_method: str = "aec",
    roi_labels: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract source-space connectivity features.
    
    Parameters
    ----------
    ctx : FeatureContext
        Feature extraction context
    bands : list of str
        Frequency bands
    method : str
        Source localization method: 'lcmv' or 'eloreta'
    connectivity_method : str
        Connectivity measure: 'aec', 'wpli', 'plv'
    roi_labels : list of str, optional
        ROI labels to use
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    features_df : pd.DataFrame
        Connectivity features per epoch
    feature_cols : list of str
        Feature column names
    """
    import mne
    from mne_connectivity import spectral_connectivity_epochs, envelope_correlation
    
    epochs = ctx.epochs
    logger = getattr(ctx, "logger", None) or logging.getLogger(__name__)
    
    n_epochs = len(epochs)
    if n_epochs < 2:
        logger.warning("Source connectivity requires at least 2 epochs")
        return pd.DataFrame(), []
    
    sfreq = epochs.info["sfreq"]
    freq_bands = ctx.freq_bands if hasattr(ctx, "freq_bands") else {}
    
    if logger:
        logger.info(f"Extracting source-space {connectivity_method.upper()} connectivity")
    
    fwd, src, _ = _setup_forward_model(epochs.info, logger=logger)
    
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc="aparc",
        subjects_dir=None,
        verbose=False,
    )
    labels = [l for l in labels if "unknown" not in l.name.lower()]
    
    if roi_labels is not None:
        labels = [l for l in labels if any(r in l.name for r in roi_labels)]
    
    if len(labels) < 2:
        logger.warning("Need at least 2 ROIs for connectivity")
        return pd.DataFrame(), []
    
    label_names = [l.name for l in labels]
    n_rois = len(labels)
    
    records = [{} for _ in range(n_epochs)]
    feature_cols = []
    
    for band in bands:
        if band not in freq_bands:
            continue
        
        fmin, fmax = freq_bands[band]
        
        epochs_band = epochs.copy().filter(fmin, fmax, n_jobs=n_jobs, verbose=False)
        
        if method.lower() == "lcmv":
            stcs, _ = _compute_lcmv_source_estimates(epochs_band, fwd, logger=None)
        else:
            stcs, _ = _compute_eloreta_source_estimates(epochs_band, fwd, logger=None)
        
        roi_data = _extract_roi_timecourses(stcs, labels, mode="mean_flip")
        
        if connectivity_method.lower() == "aec":
            for epoch_idx in range(n_epochs):
                epoch_data = roi_data[epoch_idx:epoch_idx+1, :, :]
                
                try:
                    con = envelope_correlation(
                        epoch_data,
                        orthogonalize="pairwise",
                        verbose=False,
                    )
                    con_matrix = con.combine().get_data(output="dense")[:, :, 0]
                except Exception:
                    con_matrix = np.full((n_rois, n_rois), np.nan)
                
                triu_idx = np.triu_indices(n_rois, k=1)
                mean_conn = np.nanmean(con_matrix[triu_idx])
                
                col_name = f"src_{method}_{band}_aec_global"
                if col_name not in feature_cols:
                    feature_cols.append(col_name)
                records[epoch_idx][col_name] = mean_conn
                
        elif connectivity_method.lower() in ("wpli", "plv"):
            try:
                con = spectral_connectivity_epochs(
                    stcs,
                    method=connectivity_method.lower(),
                    mode="multitaper",
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    faverage=True,
                    n_jobs=n_jobs,
                    verbose=False,
                )
                
                con_data = con.get_data()
                
                for epoch_idx in range(n_epochs):
                    if con_data.ndim == 3:
                        epoch_con = con_data[epoch_idx, :, 0]
                    else:
                        epoch_con = con_data[:, 0]
                    
                    mean_conn = np.nanmean(epoch_con)
                    
                    col_name = f"src_{method}_{band}_{connectivity_method}_global"
                    if col_name not in feature_cols:
                        feature_cols.append(col_name)
                    records[epoch_idx][col_name] = mean_conn
                    
            except Exception as e:
                logger.warning(f"Source connectivity failed for {band}: {e}")
                col_name = f"src_{method}_{band}_{connectivity_method}_global"
                if col_name not in feature_cols:
                    feature_cols.append(col_name)
                for epoch_idx in range(n_epochs):
                    records[epoch_idx][col_name] = np.nan
    
    features_df = pd.DataFrame(records)
    feature_cols = list(features_df.columns)
    
    if logger:
        logger.info(f"Source connectivity: {len(feature_cols)} features extracted")
    
    return features_df, feature_cols


def extract_source_localization_from_precomputed(
    precomputed: Any,
    method: str = "lcmv",
    bands: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract source-localized features from precomputed data.
    
    This is a wrapper that extracts source features when precomputed
    band-limited data is available.
    
    Parameters
    ----------
    precomputed : PrecomputedData
        Precomputed intermediate data
    method : str
        Source localization method
    bands : list of str, optional
        Bands to process
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    features_df : pd.DataFrame
        Source features
    feature_cols : list of str
        Feature column names
    """
    logger = getattr(precomputed, "logger", None) or logging.getLogger(__name__)
    
    if not hasattr(precomputed, "epochs") or precomputed.epochs is None:
        logger.warning("Source localization requires epochs in precomputed data")
        return pd.DataFrame(), []
    
    class MockContext:
        def __init__(self, precomputed):
            self.epochs = precomputed.epochs
            self.config = getattr(precomputed, "config", {})
            self.logger = logger
            self.freq_bands = {}
            if hasattr(precomputed, "band_data") and precomputed.band_data:
                for band_name, band_info in precomputed.band_data.items():
                    if hasattr(band_info, "fmin") and hasattr(band_info, "fmax"):
                        self.freq_bands[band_name] = (band_info.fmin, band_info.fmax)
    
    ctx = MockContext(precomputed)
    
    if bands is None:
        bands = list(ctx.freq_bands.keys())
    
    return extract_source_localization_features(
        ctx,
        bands=bands,
        method=method,
        n_jobs=n_jobs,
    )
