from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Sequence, Dict, Any
import os
import logging

import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.ndimage import label
from scipy.stats import gaussian_kde
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test

try:
    from .config_loader import load_settings
except ImportError:
    load_settings = None


###################################################################
# Constants Loading
###################################################################

def _load_statistics_constants(config=None):
    if config is None:
        if load_settings is not None:
            config = load_settings()
    
    if config is None:
        raise ValueError("Config is required. Cannot load statistics constants without config.")
    
    constants = config.get("statistics.constants")
    if constants is None:
        raise ValueError("statistics.constants not found in config.")
    
    return {
        "fisher_z_clip_min": constants["fisher_z_clip_min"],
        "fisher_z_clip_max": constants["fisher_z_clip_max"],
        "ci_multiplier_95": constants["ci_multiplier_95"],
        "ci_percentile_low": constants["ci_percentile_low"],
        "ci_percentile_high": constants["ci_percentile_high"],
        "t_critical_alpha_05": constants["t_critical_alpha_05"],
        "min_samples_for_stats": constants["min_samples_for_stats"],
        "min_samples_for_correlation": constants["min_samples_for_correlation"],
        "min_samples_for_bootstrap_warning": constants["min_samples_for_bootstrap_warning"],
        "default_fdr_alpha": constants["default_fdr_alpha"],
        "default_baseline_start": constants["default_baseline_start"],
        "default_baseline_end": constants["default_baseline_end"],
        "k_neighbors_adjacency": constants["k_neighbors_adjacency"],
        "default_min_samples_roi": constants["default_min_samples_roi"],
        "default_min_samples_channel": constants["default_min_samples_channel"],
        "log_base": constants["log_base"],
        "percentage_multiplier": constants["percentage_multiplier"],
        "cluster_structure_2d": np.array(constants["cluster_structure_2d"], dtype=int),
    }


_constants_cache = None


def _get_statistics_constants(config=None):
    global _constants_cache
    if config is None and _constants_cache is not None:
        return _constants_cache
    constants = _load_statistics_constants(config)
    if config is None:
        _constants_cache = constants
    return constants




###################################################################
# FDR (Benjamini–Hochberg)
###################################################################

def fdr_bh(pvals: Iterable[float], alpha: float = 0.05) -> np.ndarray:
    pvals_arr = np.asarray(list(pvals), dtype=float)
    qvals = np.full_like(pvals_arr, np.nan, dtype=float)

    valid_mask = np.isfinite(pvals_arr)
    if not np.any(valid_mask):
        return qvals

    pv = pvals_arr[valid_mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size

    denom = np.arange(1, n + 1, dtype=float)
    adjusted = ranked * n / denom
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted

    qvals[valid_mask] = restored
    return qvals


def fdr_bh_reject(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), np.nan
    
    valid_mask = np.isfinite(p)
    if not np.any(valid_mask):
        return np.zeros_like(p, dtype=bool), np.nan
    
    p_valid = p[valid_mask]
    order = np.argsort(p_valid)
    ranked = np.arange(1, len(p_valid) + 1)
    thresh = (ranked / len(p_valid)) * alpha
    passed = p_valid[order] <= thresh
    
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool), np.nan
    
    k_max = np.max(np.where(passed)[0])
    crit = float(p_valid[order][k_max])
    
    reject = np.zeros_like(p, dtype=bool)
    reject[valid_mask] = p_valid <= crit
    
    return reject, crit


###################################################################
# Data Validation Utilities
###################################################################

def validate_pain_binary_values(
    values: pd.Series,
    column_name: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, int]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    numeric_vals = pd.to_numeric(values, errors="coerce")
    n_total = len(values)
    n_nan = int(numeric_vals.isna().sum())
    n_invalid = int(((numeric_vals != 0) & (numeric_vals != 1) & numeric_vals.notna()).sum())
    
    if n_nan > 0 or n_invalid > 0:
        error_msg = (
            f"Invalid pain binary values in '{column_name}': "
            f"{n_nan} NaN/missing, {n_invalid} non-binary (not 0 or 1) out of {n_total} total. "
            f"Pain binary columns must contain only 0 (non-pain) and 1 (pain) values."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    validated = numeric_vals.fillna(0).astype(int).values
    n_coerced = n_nan + n_invalid
    
    return validated, n_coerced


def validate_temperature_values(
    values: pd.Series,
    column_name: str,
    min_temp: float = 35.0,
    max_temp: float = 55.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, int]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    numeric_vals = pd.to_numeric(values, errors="coerce")
    n_total = len(values)
    n_nan = int(numeric_vals.isna().sum())
    n_out_of_range = int(((numeric_vals < min_temp) | (numeric_vals > max_temp) | numeric_vals.isna()).sum())
    
    if n_out_of_range > 0:
        pct_invalid = 100.0 * n_out_of_range / n_total if n_total > 0 else 0.0
        error_msg = (
            f"Invalid temperature values in '{column_name}': "
            f"{n_nan} NaN/missing, {n_out_of_range} out of range [{min_temp}, {max_temp}]°C "
            f"({pct_invalid:.1f}% invalid) out of {n_total} total."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    validated = numeric_vals.values
    n_dropped = n_out_of_range
    
    return validated, n_dropped


def validate_baseline_window_pre_stimulus(
    baseline: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    b_start, b_end = baseline
    constants = _get_statistics_constants(config)
    b_start = float(b_start) if b_start is not None else constants["default_baseline_start"]
    b_end = float(b_end) if b_end is not None else constants["default_baseline_end"]
    
    if b_end > 0.0:
        error_msg = (
            f"Baseline window must end at or before 0 s (pre-stimulus), got [{b_start}, {b_end}]. "
            f"Post-stimulus baseline windows are invalid and would corrupt all TFR/ERP analyses."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if b_start >= b_end:
        error_msg = (
            f"Baseline window start ({b_start}) must be < end ({b_end}). "
            f"Invalid baseline window configuration."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return (b_start, b_end)


###################################################################
# EEG Adjacency and Cluster Utilities
###################################################################

def _build_distance_based_adjacency(
    info_eeg: mne.Info,
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> Tuple[Any, List[str]]:
    from scipy.spatial import distance_matrix
    from scipy import sparse
    
    channel_positions = np.array([ch['loc'][:3] for ch in info_eeg['chs']])
    if np.all(np.isnan(channel_positions)) or np.allclose(channel_positions, 0):
        logger.warning("Invalid channel positions, returning None adjacency")
        return None, []
    
    position_distances = distance_matrix(channel_positions, channel_positions)
    constants = _get_statistics_constants(config)
    n_neighbors = min(constants["k_neighbors_adjacency"], len(channel_positions) - 1)
    adjacency_matrix = np.zeros((len(channel_positions), len(channel_positions)), dtype=bool)
    
    for channel_idx in range(len(channel_positions)):
        nearest_channel_indices = np.argsort(position_distances[channel_idx])[1:n_neighbors + 1]
        adjacency_matrix[channel_idx, nearest_channel_indices] = True
        adjacency_matrix[nearest_channel_indices, channel_idx] = True
    
    adjacency_sparse = sparse.csr_matrix(adjacency_matrix)
    channel_names = [ch['ch_name'] for ch in info_eeg['chs']]
    
    return adjacency_sparse, channel_names


def get_eeg_adjacency(
    info: mne.Info,
    restrict_picks: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[Any], Optional[np.ndarray], Optional[mne.Info]]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    eeg_picks_all = mne.pick_types(info, eeg=True, exclude=[])
    if len(eeg_picks_all) == 0:
        return None, None, None
    
    if restrict_picks is not None:
        restrict_picks_arr = np.asarray(restrict_picks, dtype=int)
        restrict_set = set(restrict_picks_arr)
        eeg_picks = np.array([p for p in eeg_picks_all if p in restrict_set], dtype=int)
    else:
        eeg_picks = np.asarray(eeg_picks_all, dtype=int)
    
    if eeg_picks.size == 0:
        return None, None, None
    
    info_eeg = mne.pick_info(info, sel=eeg_picks.tolist())
    
    try:
        adjacency, channel_names = mne.channels.find_ch_adjacency(info_eeg, ch_type="eeg")
    except (RuntimeError, ValueError) as e:
        logger.warning(
            f"Delaunay adjacency failed ({e.__class__.__name__}), "
            f"using distance-based fallback"
        )
        adjacency, channel_names = _build_distance_based_adjacency(info_eeg, logger)
        if adjacency is None:
            return None, eeg_picks, info_eeg
    
    return adjacency, eeg_picks, info_eeg


def build_full_mask_from_eeg(
    sig_mask_eeg: np.ndarray,
    n_ch_total: int,
    eeg_picks: np.ndarray,
) -> np.ndarray:
    full_mask = np.zeros(n_ch_total, dtype=bool)
    full_mask[eeg_picks] = sig_mask_eeg.astype(bool)
    return full_mask


def _extract_cluster_indices(cluster: Any, expected_length: int) -> np.ndarray:
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool and cluster.shape[0] == expected_length:
        return np.where(cluster)[0]
    return np.asarray(cluster)


def _compute_cluster_masses(
    clusters: List[Any],
    t_statistic: np.ndarray,
) -> List[float]:
    masses: List[float] = []
    for cluster in clusters:
        cluster_indices = _extract_cluster_indices(cluster, t_statistic.shape[0])
        if cluster_indices.size == 0:
            masses.append(0.0)
        else:
            cluster_mass = float(np.nansum(np.abs(t_statistic[cluster_indices])))
            masses.append(cluster_mass)
    return masses


def cluster_mask_from_clusters(
    clusters: List[Any],
    p_values: np.ndarray,
    n_features: int,
    alpha: float,
) -> np.ndarray:
    significant_mask = np.zeros(n_features, dtype=bool)
    
    for cluster, p_value in zip(clusters, p_values):
        if float(p_value) > float(alpha):
            continue
        
        cluster_indices = _extract_cluster_indices(cluster, n_features)
        if isinstance(cluster, np.ndarray) and cluster.dtype == bool and cluster.shape[0] == n_features:
            significant_mask |= cluster
        else:
            significant_mask[cluster_indices] = True
    
    return significant_mask


def _resolve_cluster_n_jobs(config=None) -> int:
    raw = os.getenv("EEG_CLUSTER_N_JOBS")
    if raw and raw.strip().lower() not in {"auto", ""}:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    
    if config is None and load_settings is not None:
        try:
            config = load_settings()
        except Exception:
            pass
    
    default_n_jobs = -1
    if config is not None:
        default_n_jobs = int(config.get("statistics.cluster_n_jobs", -1))
    
    return default_n_jobs


def cluster_test_two_sample_arrays(
    group_a_data: np.ndarray,
    group_b_data: np.ndarray,
    info: mne.Info,
    alpha: float = 0.05,
    paired: bool = False,
    n_permutations: int = 1024,
    restrict_picks: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    config=None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    adjacency, eeg_picks, info_eeg = get_eeg_adjacency(info, restrict_picks=restrict_picks)
    if eeg_picks is None or info_eeg is None:
        return None, None, None, None
    
    if n_jobs is None:
        n_jobs = _resolve_cluster_n_jobs(config=config)
    
    group_a_eeg = np.asarray(group_a_data)[:, eeg_picks]
    group_b_eeg = np.asarray(group_b_data)[:, eeg_picks]
    
    if paired and group_a_eeg.shape[0] == group_b_eeg.shape[0]:
        difference_data = group_a_eeg - group_b_eeg
        t_statistic, clusters, p_values, _ = permutation_cluster_1samp_test(
            difference_data,
            n_permutations=int(n_permutations),
            adjacency=adjacency,
            tail=0,
            out_type="mask",
            n_jobs=n_jobs,
        )
    else:
        t_statistic, clusters, p_values, _ = permutation_cluster_test(
            [group_a_eeg, group_b_eeg],
            n_permutations=int(n_permutations),
            adjacency=adjacency,
            tail=0,
            out_type="mask",
            n_jobs=n_jobs,
        )
    
    significant_eeg_mask = cluster_mask_from_clusters(
        clusters, p_values, n_features=group_a_eeg.shape[1], alpha=alpha
    )
    significant_full_mask = build_full_mask_from_eeg(
        significant_eeg_mask, n_ch_total=len(info["ch_names"]), eeg_picks=eeg_picks
    )
    
    if len(clusters) == 0:
        return significant_full_mask, None, None, None
    
    cluster_masses = _compute_cluster_masses(clusters, t_statistic)
    if len(cluster_masses) == 0:
        return significant_full_mask, None, None, None
    
    largest_cluster_idx = int(np.nanargmax(np.asarray(cluster_masses)))
    largest_cluster = clusters[largest_cluster_idx]
    largest_cluster_indices = _extract_cluster_indices(largest_cluster, t_statistic.shape[0])
    
    p_value = float(p_values[largest_cluster_idx]) if np.isfinite(p_values[largest_cluster_idx]) else None
    cluster_size = int(largest_cluster_indices.size)
    cluster_mass = float(np.nansum(np.abs(t_statistic[largest_cluster_indices]))) if largest_cluster_indices.size > 0 else 0.0
    
    return significant_full_mask, p_value, cluster_size, cluster_mass


def cluster_test_epochs(
    tfr_epochs: "mne.time_frequency.EpochsTFR",
    group_a_mask: np.ndarray,
    group_b_mask: np.ndarray,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    paired: bool = False,
    alpha: float = 0.05,
    n_permutations: int = 1024,
    restrict_picks: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    config=None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    info = tfr_epochs.info
    
    eeg_picks = mne.pick_types(info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        if logger:
            logger.warning("cluster_test_epochs: No EEG channels found in tfr_epochs.info")
        return None, None, None, None
    
    frequencies = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    frequency_mask = (frequencies >= float(fmin)) & (frequencies <= float(fmax))
    time_mask = (times >= float(tmin)) & (times < float(tmax))
    
    if frequency_mask.sum() == 0 or time_mask.sum() == 0:
        return None, None, None, None
    
    time_freq_data = np.asarray(tfr_epochs.data)[:, :, frequency_mask, :][:, :, :, time_mask]
    channel_power = time_freq_data.mean(axis=(2, 3))
    
    if channel_power.shape[1] != len(info["ch_names"]):
        if logger:
            logger.error(
                f"cluster_test_epochs: Channel dimension mismatch: "
                f"channel_power.shape[1]={channel_power.shape[1]} != len(info['ch_names'])={len(info['ch_names'])}"
            )
        return None, None, None, None
    
    group_a_data = np.asarray(channel_power)[np.asarray(group_a_mask, dtype=bool), :]
    group_b_data = np.asarray(channel_power)[np.asarray(group_b_mask, dtype=bool), :]
    
    if group_a_data.shape[0] < 2 or group_b_data.shape[0] < 2:
        return None, None, None, None
    
    return cluster_test_two_sample_arrays(
        group_a_data, group_b_data, info,
        alpha=alpha, paired=paired, n_permutations=n_permutations,
        restrict_picks=restrict_picks, n_jobs=n_jobs, config=config,
    )


def fdr_bh_mask(p_vals: np.ndarray, alpha: float = 0.05) -> Optional[np.ndarray]:
    p_vals = np.asarray(p_vals, dtype=float)
    if p_vals.ndim != 1 or p_vals.size == 0:
        return None
    finite = np.isfinite(p_vals)
    if not np.any(finite):
        return np.zeros_like(p_vals, dtype=bool)
    rej, _ = fdr_bh_reject(p_vals[finite], alpha=float(alpha))
    mask = np.zeros_like(p_vals, dtype=bool)
    mask[finite] = rej.astype(bool)
    return mask


def fdr_bh_values(p_vals: np.ndarray, alpha: float = 0.05) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p_vals = np.asarray(p_vals, dtype=float)
    if p_vals.ndim != 1 or p_vals.size == 0:
        return None, None
    finite = np.isfinite(p_vals)
    if not np.any(finite):
        return np.zeros_like(p_vals, dtype=bool), np.full_like(p_vals, np.nan, dtype=float)
    rej, _ = fdr_bh_reject(p_vals[finite], alpha=float(alpha))
    q_vals = fdr_bh(p_vals[finite], alpha=float(alpha))
    reject_mask = np.zeros_like(p_vals, dtype=bool)
    q = np.full_like(p_vals, np.nan, dtype=float)
    reject_mask[finite] = rej.astype(bool)
    q[finite] = q_vals.astype(float)
    return reject_mask, q


###################################################################
# PyRiemann and Epoch Utilities
###################################################################

def check_pyriemann() -> bool:
    try:
        import pyriemann
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        return True
    except ImportError:
        return False


def align_epochs_to_pivot_chs(
    epochs: "mne.Epochs",
    pivot_chs: List[str],
    logger: Optional[logging.Logger] = None,
    max_missing_fraction: float = 0.5,
) -> "mne.Epochs":
    current_chs = set(epochs.info["ch_names"])
    missing = set(pivot_chs) - current_chs
    
    if missing:
        n_missing = len(missing)
        n_total = len(pivot_chs)
        missing_fraction = n_missing / n_total if n_total > 0 else 0.0
        
        if missing_fraction > max_missing_fraction:
            raise ValueError(
                f"Too many missing channels: {n_missing}/{n_total} ({missing_fraction:.1%}) "
                f"exceeds maximum allowed fraction {max_missing_fraction:.1%}. "
                f"Missing channels: {sorted(missing)}"
            )
        
        if logger:
            logger.warning(
                f"Interpolating {n_missing} missing channels ({missing_fraction:.1%}): {sorted(missing)}"
            )
        
        epochs.add_channels([mne.io.RawArray(
            np.zeros((len(missing), epochs.times.size)),
            mne.create_info(list(missing), epochs.info["sfreq"], "eeg")
        )])
        epochs.info["bads"].extend(list(missing))
        epochs.interpolate_bads(reset_bads=True)
    
    return epochs.pick_channels(pivot_chs, ordered=True)


###################################################################
# Correlation Utilities and Fisher Aggregation
###################################################################

def get_correlation_method(use_spearman: bool) -> str:
    return "spearman" if use_spearman else "pearson"


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    use_spearman: bool = True,
) -> Tuple[float, float]:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.nan, np.nan
    
    if use_spearman:
        correlation, p_value = stats.spearmanr(x, y, nan_policy="omit")
    else:
        correlation, p_value = stats.pearsonr(x, y)
    
    return _safe_float(correlation), _safe_float(p_value)


def compute_bootstrap_ci(
    x: pd.Series,
    y: pd.Series,
    n_bootstrap: int,
    use_spearman: bool,
    rng: np.random.Generator,
    min_samples: int = 5,
    logger=None,
    config=None,
) -> Tuple[float, float]:
    if n_bootstrap <= 0:
        return np.nan, np.nan
    
    valid_mask = x.notna() & y.notna()
    n_valid = int(valid_mask.sum())
    if n_valid < min_samples:
        return np.nan, np.nan
    
    valid_indices = np.where(valid_mask.to_numpy())[0]
    if len(valid_indices) == 0:
        return np.nan, np.nan
    
    use_fisher_z = False
    if config is not None:
        use_fisher_z = bool(config.get("behavior_analysis.statistics.bootstrap_use_fisher_z", False))
    
    constants = _get_statistics_constants(config)
    min_samples_warning = constants["min_samples_for_bootstrap_warning"]
    if n_valid < min_samples_warning and logger is not None:
        if use_fisher_z:
            logger.info(f"Using Fisher z-transformation for bootstrap CI with small sample size (n={n_valid})")
        else:
            logger.warning(
                f"Bootstrap CI computed with small sample size (n={n_valid}). "
                f"Bootstrap confidence intervals may be biased for n < {min_samples_warning}. "
                f"Consider enabling bootstrap_use_fisher_z in config for better accuracy."
            )
    
    bootstrap_correlations = []
    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(valid_indices, size=len(valid_indices), replace=True)
        x_sample = x.iloc[sampled_indices]
        y_sample = y.iloc[sampled_indices]
        correlation, _ = compute_correlation(x_sample, y_sample, use_spearman)
        
        if not np.isfinite(correlation):
            continue
        
        if use_fisher_z:
            constants = _get_statistics_constants(config)
            correlation = np.clip(correlation, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
            correlation = np.arctanh(correlation)
        
        bootstrap_correlations.append(correlation)
    
    if not bootstrap_correlations:
        return np.nan, np.nan
    
    constants = _get_statistics_constants(config)
    ci_percentiles = (constants["ci_percentile_low"], constants["ci_percentile_high"])
    ci_low, ci_high = np.percentile(bootstrap_correlations, ci_percentiles)
    
    if use_fisher_z:
        ci_low = np.tanh(ci_low)
        ci_high = np.tanh(ci_high)
    
    return _safe_float(ci_low), _safe_float(ci_high)


def get_fdr_alpha_from_config(config) -> float:
    direct = config.get("behavior_analysis.statistics.fdr_alpha")
    if direct is not None:
        return float(direct)
    
    behavior_config = config.get("analysis", {}).get("behavior_analysis", {})
    statistics_config = behavior_config.get("statistics", {})
    constants = _get_statistics_constants(config)
    nested_alpha = statistics_config.get("fdr_alpha", constants["default_fdr_alpha"])
    
    return float(nested_alpha)


def select_p_values_for_fdr(results_df: pd.DataFrame, use_permutation_p: bool) -> np.ndarray:
    if not use_permutation_p:
        return results_df["p"].to_numpy()
    
    if "p_perm" not in results_df.columns:
        return results_df["p"].to_numpy()
    
    p_perm_values = results_df["p_perm"].to_numpy()
    has_valid_permutation = np.isfinite(p_perm_values).any()
    if has_valid_permutation:
        return p_perm_values
    
    return results_df["p"].to_numpy()


def filter_significant_predictors(
    df: pd.DataFrame,
    use_fdr: bool,
    alpha: float,
) -> pd.DataFrame:
    if not use_fdr:
        return df[df["p"] <= alpha].copy()
    
    if "fdr_reject" in df.columns:
        return df[df["fdr_reject"]].copy()
    
    if "fdr_crit_p" in df.columns and "p" in df.columns:
        return df[df["p"] <= df["fdr_crit_p"]].copy()
    
    return df[df["p"] <= alpha].copy()


def compute_fisher_transformed_mean(edge_df: pd.DataFrame, config: Optional[Any] = None) -> pd.Series:
    edge_array = edge_df.to_numpy(dtype=float)
    constants = _get_statistics_constants(config)
    edge_array = np.clip(edge_array, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
    z_scores = np.arctanh(edge_array)
    z_mean = np.nanmean(z_scores, axis=1)
    return pd.Series(np.tanh(z_mean), index=edge_df.index)


def compute_correlation_for_metric_state(
    metric_vals: pd.Series,
    ratings: pd.Series,
    method: str,
    config: Optional[Any] = None,
) -> Optional[Tuple[float, float]]:
    valid_mask = metric_vals.notna() & ratings.notna()
    constants = _get_statistics_constants(config)
    if valid_mask.sum() < constants["min_samples_for_correlation"]:
        return None
    
    x = metric_vals[valid_mask].to_numpy()
    y = ratings[valid_mask].to_numpy()
    if np.std(x) <= 0 or np.std(y) <= 0:
        return None
    
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def compute_duration_p_value(nonpain_data: np.ndarray, pain_data: np.ndarray) -> float:
    if nonpain_data.size == 0 or pain_data.size == 0:
        return np.nan
    
    try:
        _, p_val = stats.mannwhitneyu(nonpain_data, pain_data, alternative="two-sided")
        return float(p_val) if np.isfinite(p_val) else np.nan
    except ValueError:
        return np.nan


def compute_correlation_pvalue(r_values: List[float], config: Optional[Any] = None) -> float:
    if not r_values:
        return np.nan
    
    valid_values = np.array([r for r in r_values if np.isfinite(r)])
    constants = _get_statistics_constants(config)
    if valid_values.size < constants["min_samples_for_stats"]:
        return np.nan
    
    valid_values = np.clip(valid_values, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
    z_scores = np.arctanh(valid_values)
    ttest_result = stats.ttest_1samp(z_scores, popmean=0.0)
    return _get_ttest_pvalue(ttest_result)


def compute_channel_confidence_interval(z_scores: np.ndarray, config: Optional[Any] = None) -> Tuple[float, float]:
    constants = _get_statistics_constants(config)
    if z_scores.size == 0 or len(z_scores) < constants["min_samples_for_stats"]:
        return np.nan, np.nan
    
    std_dev = _safe_float(np.std(z_scores, ddof=1))
    se = std_dev / np.sqrt(len(z_scores)) if std_dev > 0 else np.nan
    
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan
    
    t_crit = _safe_float(stats.t.ppf(constants["t_critical_alpha_05"], df=len(z_scores) - 1))
    mean_z = np.mean(z_scores)
    ci_low = _safe_float(np.tanh(mean_z - t_crit * se))
    ci_high = _safe_float(np.tanh(mean_z + t_crit * se))
    
    return ci_low, ci_high


def compute_group_channel_statistics(correlations_by_channel: Dict[str, List[float]], config: Optional[Any] = None) -> pd.DataFrame:
    if not correlations_by_channel:
        return pd.DataFrame()
    
    constants = _get_statistics_constants(config)
    output_rows = []
    for channel, r_values in sorted(correlations_by_channel.items()):
        values = np.array(r_values, dtype=float)
        values = values[np.isfinite(values)]
        values = np.clip(values, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
        
        if values.size == 0:
            continue
        
        z_scores = np.arctanh(values)
        r_group = _safe_float(np.tanh(np.mean(z_scores)))
        
        if len(z_scores) >= constants["min_samples_for_stats"]:
            ttest_result = stats.ttest_1samp(z_scores, popmean=0.0)
            p_value = _get_ttest_pvalue(ttest_result)
        else:
            p_value = np.nan
        
        ci_low, ci_high = compute_channel_confidence_interval(z_scores)
        
        output_rows.append({
            "channel": channel,
            "r_group": r_group,
            "p_group": p_value,
            "r_ci_low": ci_low,
            "r_ci_high": ci_high,
            "n_subjects": int(len(z_scores)),
        })
    
    return pd.DataFrame(output_rows)


def normalize_series(
    x_series: pd.Series,
    y_series: pd.Series,
    strategy: str,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    if x_series.empty or y_series.empty:
        return None, None
    
    if strategy == "within_subject_centered":
        return x_series - x_series.mean(), y_series - y_series.mean()
    
    if strategy == "within_subject_zscored":
        std_x = x_series.std(ddof=1)
        std_y = y_series.std(ddof=1)
        if std_x <= 0 or std_y <= 0:
            return None, None
        return (
            (x_series - x_series.mean()) / std_x,
            (y_series - y_series.mean()) / std_y
        )
    
    if strategy == "fisher_by_subject":
        return x_series - x_series.mean(), y_series - y_series.mean()
    
    return x_series, y_series


def pool_data_by_strategy(x_lists: List[np.ndarray], y_lists: List[np.ndarray], 
                          strategy: str) -> Tuple[pd.Series, pd.Series]:
    if not x_lists or not y_lists:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    pooled_x = []
    pooled_y = []
    
    for x_values, y_values in zip(x_lists, y_lists):
        x_series = pd.Series(x_values)
        y_series = pd.Series(y_values)
        
        n = min(len(x_series), len(y_series))
        x_series = x_series.iloc[:n]
        y_series = y_series.iloc[:n]
        
        valid_mask = x_series.notna() & y_series.notna()
        x_series = x_series[valid_mask]
        y_series = y_series[valid_mask]
        
        x_normalized, y_normalized = normalize_series(x_series, y_series, strategy)
        if x_normalized is None:
            continue
        
        pooled_x.append(x_normalized)
        pooled_y.append(y_normalized)
    
    if pooled_x:
        return pd.concat(pooled_x, ignore_index=True), pd.concat(pooled_y, ignore_index=True)
    
    return pd.Series(dtype=float), pd.Series(dtype=float)


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, p.size + 1, dtype=float)
    p_sorted = p[order]
    q_raw = p_sorted * p.size / ranks
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def fisher_aggregate(rs: List[float], config: Optional[Any] = None) -> Tuple[float, float, float, int]:
    vals = np.array([r for r in rs if np.isfinite(r)])
    constants = _get_statistics_constants(config)
    vals = np.clip(vals, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
    n = vals.size
    if n < constants["min_samples_for_stats"]:
        return np.nan, np.nan, np.nan, n
    z = np.arctanh(vals)
    mean_z = _safe_float(np.mean(z))
    sd_z = _safe_float(np.std(z, ddof=1))
    se = sd_z / np.sqrt(n) if sd_z > 0 else np.nan
    if np.isnan(se) or se == 0:
        return _safe_float(np.tanh(mean_z)), np.nan, np.nan, n
    tcrit = _safe_float(stats.t.ppf(constants["t_critical_alpha_05"], df=n - 1))
    ci_low_z = mean_z - tcrit * se
    ci_high_z = mean_z + tcrit * se
    return _safe_float(np.tanh(mean_z)), _safe_float(np.tanh(ci_low_z)), _safe_float(np.tanh(ci_high_z)), n


def _build_design_matrix(
    df: pd.DataFrame,
    covariate_columns: List[str],
    method: str,
) -> Optional[np.ndarray]:
    intercept_column = np.ones(len(df))
    
    if method == "spearman":
        ranked_covariates = [
            stats.rankdata(df[col].to_numpy()) for col in covariate_columns
        ]
        if ranked_covariates:
            design_matrix = np.column_stack([intercept_column] + ranked_covariates)
        else:
            design_matrix = intercept_column.reshape(-1, 1)
    else:
        covariate_data = df[covariate_columns].to_numpy() if covariate_columns else np.empty((len(df), 0))
        design_matrix = np.column_stack([intercept_column, covariate_data])
    
    if design_matrix.shape[1] > len(df):
        return None
    
    if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
        return None
    
    return design_matrix


def _compute_partial_residuals(
    df: pd.DataFrame,
    method: str,
    config: Optional[Any] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    constants = _get_statistics_constants(config)
    if len(df) < constants["min_samples_for_correlation"] or df["y"].nunique() <= 1:
        return None
    
    covariate_columns = [col for col in df.columns if col not in ("x", "y")]
    design_matrix = _build_design_matrix(df, covariate_columns, method)
    if design_matrix is None:
        return None
    
    if method == "spearman":
        x_data = stats.rankdata(df["x"].to_numpy())
        y_data = stats.rankdata(df["y"].to_numpy())
    else:
        x_data = df["x"].to_numpy()
        y_data = df["y"].to_numpy()
    
    x_coefficients = np.linalg.lstsq(design_matrix, x_data, rcond=None)[0]
    y_coefficients = np.linalg.lstsq(design_matrix, y_data, rcond=None)[0]
    
    x_residuals = x_data - design_matrix @ x_coefficients
    y_residuals = y_data - design_matrix @ y_coefficients
    
    return x_residuals, y_residuals, int(len(df))


def partial_corr_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str, config: Optional[Any] = None) -> Tuple[float, float, int]:
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    result = _compute_partial_residuals(df, method, config=config)
    if result is None:
        return np.nan, np.nan, 0
    
    x_residuals, y_residuals, sample_size = result
    if np.std(x_residuals) <= 0 or np.std(y_residuals) <= 0:
        return np.nan, np.nan, sample_size
    
    correlation, p_value = stats.pearsonr(x_residuals, y_residuals)
    return _safe_float(correlation), _safe_float(p_value), sample_size


def partial_residuals_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str, config: Optional[Any] = None) -> Tuple[pd.Series, pd.Series, int]:
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    result = _compute_partial_residuals(df, method, config=config)
    if result is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    x_residuals, y_residuals, sample_size = result
    return pd.Series(x_residuals, index=df.index), pd.Series(y_residuals, index=df.index), sample_size


def compute_partial_corr(x: pd.Series, y: pd.Series, Z: Optional[pd.DataFrame], method: str, *, logger=None, context: str = "", config: Optional[Any] = None) -> Tuple[float, float, int]:
    if Z is None or Z.empty:
        return np.nan, np.nan, 0
    
    if len(Z.columns) == 0:
        if logger:
            warning_msg = f"{context}: Z has no columns; skipping partial correlation" if context else "Z has no columns; skipping partial correlation"
            logger.warning(warning_msg)
        return np.nan, np.nan, 0
    
    _ensure_aligned_lengths_for_partial(x, y, Z, context=context, strict=True, logger=logger)
    data = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
    cleaned_data = data.dropna()
    
    if logger and len(data) > len(cleaned_data):
        dropped_count = len(data) - len(cleaned_data)
        prefix = f"{context}: " if context else ""
        logger.warning(f"{prefix}partial correlation dropped {dropped_count} rows due to missing data (kept {len(cleaned_data)}/{len(data)})")
    
    constants = _get_statistics_constants(config)
    if len(cleaned_data) < constants["min_samples_for_correlation"] or cleaned_data["y"].nunique() <= 1:
        return np.nan, np.nan, 0
    
    return partial_corr_xy_given_Z(cleaned_data["x"], cleaned_data["y"], cleaned_data[Z.columns], method, config=config)


def compute_partial_residuals(x: pd.Series, y: pd.Series, Z: Optional[pd.DataFrame], method: str, *, logger=None, context: str = "", config: Optional[Any] = None) -> Tuple[pd.Series, pd.Series, int]:
    if Z is None or Z.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    data = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
    cleaned_data = data.dropna()
    
    if logger and len(data) > len(cleaned_data):
        dropped_count = len(data) - len(cleaned_data)
        prefix = f"{context}: " if context else ""
        logger.warning(f"{prefix}partial residuals dropped {dropped_count} rows due to missing data (kept {len(cleaned_data)}/{len(data)})")
    
    constants = _get_statistics_constants(config)
    if len(cleaned_data) < constants["min_samples_for_correlation"] or cleaned_data["y"].nunique() <= 1:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    return partial_residuals_xy_given_Z(cleaned_data["x"], cleaned_data["y"], cleaned_data[Z.columns], method, config=config)


def fisher_ci(r: float, n: int, config: Optional[Any] = None) -> Tuple[float, float]:
    if not np.isfinite(r) or n < 4:
        return np.nan, np.nan
    constants = _get_statistics_constants(config)
    r = _safe_float(np.clip(r, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"]))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_lo, z_hi = z - constants["ci_multiplier_95"] * se, z + constants["ci_multiplier_95"] * se
    return _safe_float(np.tanh(z_lo)), _safe_float(np.tanh(z_hi))


def joint_valid_mask(*arrays: Sequence, require_all: bool = True) -> np.ndarray:
    filtered = [arr for arr in arrays if arr is not None]
    if not filtered:
        raise ValueError("joint_valid_mask requires at least one non-None array")

    masks: List[np.ndarray] = []
    length: Optional[int] = None
    for arr in filtered:
        if isinstance(arr, pd.Series):
            values = arr.to_numpy()
        elif isinstance(arr, pd.DataFrame):
            values = arr.to_numpy()
        else:
            values = np.asarray(arr)

        if values.ndim > 2:
            raise ValueError("joint_valid_mask only supports 1D/2D inputs")

        if length is None:
            length = values.shape[0]
        elif values.shape[0] != length:
            raise ValueError(
                f"joint_valid_mask length mismatch: expected {length}, got {values.shape[0]}"
            )

        finite = np.isfinite(values)
        if values.ndim == 2:
            finite = np.all(finite, axis=1)
        masks.append(finite.astype(bool))

    if length is None:
        raise ValueError("joint_valid_mask could not determine sequence length")

    if not require_all:
        return masks[0]

    mask = np.ones(length, dtype=bool)
    for finite in masks:
        mask &= finite
    return mask


def prepare_aligned_data(
    x: "pd.Series | np.ndarray",
    y: "pd.Series | np.ndarray",
    Z: Optional[pd.DataFrame] = None,
    *,
    min_samples: int = 5,
    logger=None,
    context: str = "",
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.DataFrame], int, int]:
    def _to_series(arr, name: str) -> pd.Series:
        if isinstance(arr, pd.Series):
            sr = arr.copy()
            sr.name = name
            return sr
        return pd.Series(arr, name=name)

    x_series = _to_series(x, "__x__")
    y_series = _to_series(y, "__y__")

    frames: List[pd.Series | pd.DataFrame] = [x_series, y_series]
    if Z is not None and not Z.empty:
        frames.append(Z)

    data = pd.concat(frames, axis=1)
    n_total = len(data)
    data_clean = data.dropna()
    n_kept = len(data_clean)

    if logger and n_total != n_kept:
        logger.warning(
            f"{context + ': ' if context else ''}dropped {n_total - n_kept} of {n_total} rows due to NaNs"
        )

    if n_kept < max(min_samples, 1):
        if logger:
            logger.warning(
                f"{context + ': ' if context else ''}insufficient samples after cleaning (kept {n_kept}/{n_total}, need >= {min_samples})"
            )
        return None, None, None, n_total, n_kept

    x_clean = data_clean.pop("__x__").rename(getattr(x, "name", "x"))
    y_clean = data_clean.pop("__y__").rename(getattr(y, "name", "y"))
    Z_clean = data_clean if (Z is not None and not Z.empty) else None

    return x_clean, y_clean, Z_clean, n_total, n_kept


def perm_pval_simple(x: pd.Series, y: pd.Series, method: str, n_perm: int, rng: np.random.Generator, config: Optional[Any] = None) -> float:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    constants = _get_statistics_constants(config)
    if len(df) < constants["min_samples_for_correlation"]:
        return np.nan
    
    if method == "pearson":
        corr_func = stats.pearsonr
    elif method == "spearman":
        corr_func = stats.spearmanr
    else:
        raise ValueError(f"Unsupported method: {method}. Must be 'pearson' or 'spearman'")
    
    observed_correlation, _ = corr_func(df["x"], df["y"])
    exceedance_count = 1
    y_values = df["y"].to_numpy()
    
    for _ in range(n_perm):
        y_permuted = y_values[rng.permutation(len(y_values))]
        permuted_correlation, _ = corr_func(df["x"], y_permuted)
        if np.abs(permuted_correlation) >= np.abs(observed_correlation):
            exceedance_count += 1
    
    return exceedance_count / (n_perm + 1)


def perm_pval_partial_freedman_lane(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str, n_perm: int, rng: np.random.Generator, config: Optional[Any] = None) -> float:
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    constants = _get_statistics_constants(config)
    if len(df) < constants["min_samples_for_correlation"]:
        return np.nan
    
    intercept_column = np.ones(len(df))
    
    if method == "spearman":
        x_ranked = stats.rankdata(df["x"].to_numpy())
        y_ranked = stats.rankdata(df["y"].to_numpy())
        Z_ranked = np.column_stack([stats.rankdata(df[col].to_numpy()) for col in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
        design_matrix = np.column_stack([intercept_column, Z_ranked])
        x_coefficients = np.linalg.lstsq(design_matrix, x_ranked, rcond=None)[0]
        y_coefficients = np.linalg.lstsq(design_matrix, y_ranked, rcond=None)[0]
        x_residuals = x_ranked - design_matrix @ x_coefficients
        y_residuals = y_ranked - design_matrix @ y_coefficients
    else:
        Z_data = df[Z.columns].to_numpy()
        design_matrix = np.column_stack([intercept_column, Z_data])
        x_coefficients = np.linalg.lstsq(design_matrix, df["x"].to_numpy(), rcond=None)[0]
        y_coefficients = np.linalg.lstsq(design_matrix, df["y"].to_numpy(), rcond=None)[0]
        x_residuals = df["x"].to_numpy() - design_matrix @ x_coefficients
        y_residuals = df["y"].to_numpy() - design_matrix @ y_coefficients
    
    observed_correlation, _ = stats.pearsonr(x_residuals, y_residuals)
    exceedance_count = 1
    
    for _ in range(n_perm):
        y_residuals_permuted = y_residuals[rng.permutation(len(y_residuals))]
        permuted_correlation, _ = stats.pearsonr(x_residuals, y_residuals_permuted)
        if np.abs(permuted_correlation) >= np.abs(observed_correlation):
            exceedance_count += 1
    
    return exceedance_count / (n_perm + 1)


def bootstrap_corr_ci(
    x: pd.Series,
    y: pd.Series,
    method: str,
    n_boot: int = 1000,
    rng: Optional[np.random.Generator] = None,
    *,
    min_samples: int = 5,
    ci_percentiles: Optional[Tuple[float, float]] = None,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float]:
    constants = _get_statistics_constants(config)
    if ci_percentiles is None:
        ci_percentiles = (constants["ci_percentile_low"], constants["ci_percentile_high"])
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if n_boot <= 0 or len(df) < max(3, int(min_samples)):
        return np.nan, np.nan
    
    n_valid = len(df)
    if n_valid < constants["min_samples_for_bootstrap_warning"] and logger is not None:
        logger.warning(
            f"Bootstrap CI computed with small sample size (n={n_valid}). "
            f"Bootstrap confidence intervals may be biased for n < {constants['min_samples_for_bootstrap_warning']}. "
            f"Consider using bias-corrected bootstrap methods or Fisher z-transformation."
        )
    
    rng = rng or np.random.default_rng(42)
    x_vals = df["x"].to_numpy()
    y_vals = df["y"].to_numpy()
    boots: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(df), size=len(df))
        if method == "pearson":
            r, _ = stats.pearsonr(x_vals[idx], y_vals[idx])
        else:
            r, _ = stats.spearmanr(x_vals[idx], y_vals[idx], nan_policy="omit")
        if np.isfinite(r):
            boots.append(float(r))
    if not boots:
        return np.nan, np.nan
    lo, hi = np.percentile(boots, list(ci_percentiles))
    return float(lo), float(hi)


def _normalize_pair(x_data: np.ndarray, y_data: np.ndarray, strategy: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x_normalized = x_data.copy()
    y_normalized = y_data.copy()
    
    if strategy == "within_subject_centered":
        x_normalized -= np.nanmean(x_normalized)
        y_normalized -= np.nanmean(y_normalized)
        return x_normalized, y_normalized
    
    if strategy == "within_subject_zscored":
        x_std = np.nanstd(x_normalized, ddof=1)
        y_std = np.nanstd(y_normalized, ddof=1)
        if x_std <= 0 or y_std <= 0:
            return None
        x_normalized = (x_normalized - np.nanmean(x_normalized)) / x_std
        y_normalized = (y_normalized - np.nanmean(y_normalized)) / y_std
        return x_normalized, y_normalized
    
    return x_normalized, y_normalized


def _compute_bootstrap_ci_for_pairs(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
    corr_func,
    n_boot: int,
    rng: np.random.Generator,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    constants = _get_statistics_constants(config)
    if n_boot <= 0 or len(pairs) < constants["min_samples_for_stats"]:
        return np.nan, np.nan
    
    bootstrap_correlations: List[float] = []
    pair_indices = np.arange(len(pairs))
    
    for _ in range(int(n_boot)):
        sampled_indices = rng.choice(pair_indices, size=len(pairs), replace=True)
        x_samples: List[np.ndarray] = []
        y_samples: List[np.ndarray] = []
        
        for idx in sampled_indices:
            x_data, y_data = pairs[idx]
            normalized = _normalize_pair(x_data, y_data, strategy)
            if normalized is None:
                continue
            x_samples.append(normalized[0])
            y_samples.append(normalized[1])
        
        if not x_samples or not y_samples:
            continue
        
        x_concatenated = np.concatenate(x_samples)
        y_concatenated = np.concatenate(y_samples)
        correlation, _ = corr_func(x_concatenated, y_concatenated)
        
        if np.isfinite(correlation):
            bootstrap_correlations.append(correlation)
    
    if not bootstrap_correlations:
        return np.nan, np.nan
    
    return (
        _safe_float(np.percentile(bootstrap_correlations, constants["ci_percentile_low"])),
        _safe_float(np.percentile(bootstrap_correlations, constants["ci_percentile_high"])),
    )


def _build_valid_pairs(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    config: Optional[Any] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    constants = _get_statistics_constants(config)
    for x_array, y_array in zip(x_lists, y_lists):
        x_arr = np.asarray(x_array)
        y_arr = np.asarray(y_array)
        
        if len(x_arr) != len(y_arr):
            raise ValueError(
                f"Group correlation requested with mismatched trial counts "
                f"(len(x)={len(x_arr)}, len(y)={len(y_arr)})."
            )
        
        valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if int(valid_mask.sum()) >= constants["min_samples_for_correlation"]:
            pairs.append((x_arr[valid_mask], y_arr[valid_mask]))
    
    return pairs


def _compute_correlation_by_method(
    x_array: np.ndarray,
    y_array: np.ndarray,
    method: str,
) -> Tuple[float, float]:
    if method.lower() == "pearson":
        correlation, p_value = stats.pearsonr(x_array, y_array)
    else:
        correlation, p_value = stats.spearmanr(x_array, y_array, nan_policy="omit")
    return _safe_float(correlation), _safe_float(p_value)


def _compute_pooled_strategy_stats(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
    method: str,
    n_cluster_boot: int,
    rng: np.random.Generator,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, int, Tuple[float, float], float]:
    valid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    subject_correlations: List[float] = []
    constants = _get_statistics_constants(config)
    
    for x_array, y_array in pairs:
        normalized = _normalize_pair(x_array, y_array, strategy)
        if normalized is None:
            continue
        
        x_normalized, y_normalized = normalized
        valid_pairs.append((x_normalized, y_normalized))
        
        correlation, _ = _compute_correlation_by_method(x_normalized, y_normalized, method)
        if np.isfinite(correlation):
            clipped_correlation = np.clip(correlation, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
            subject_correlations.append(_safe_float(clipped_correlation))
    
    if not valid_pairs:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan
    
    x_pooled = np.concatenate([x for x, _ in valid_pairs])
    y_pooled = np.concatenate([y for _, y in valid_pairs])
    
    r_observed, p_observed = _compute_correlation_by_method(x_pooled, y_pooled, method)
    n_trials = len(x_pooled)
    n_subjects = len(valid_pairs)
    
    p_group = np.nan
    if len(subject_correlations) >= constants["min_samples_for_stats"]:
        z_scores = np.arctanh(np.array(subject_correlations))
        ttest_result = stats.ttest_1samp(z_scores, popmean=0.0, nan_policy="omit")
        p_group = _get_ttest_pvalue(ttest_result)
    
    ci = (np.nan, np.nan)
    if n_cluster_boot and n_subjects >= constants["min_samples_for_stats"]:
        def correlation_wrapper(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[float, float]:
            return _compute_correlation_by_method(x_arr, y_arr, method)
        
        ci = _compute_bootstrap_ci_for_pairs(
            valid_pairs, strategy, correlation_wrapper, n_cluster_boot, rng, config=config
        )
    
    return (
        _safe_float(r_observed),
        _safe_float(p_group),
        n_trials,
        n_subjects,
        ci,
        _safe_float(p_observed),
    )


def _compute_fisher_strategy_stats(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    method: str,
    n_cluster_boot: int,
    rng: np.random.Generator,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, int, Tuple[float, float], float]:
    subject_correlations: List[float] = []
    constants = _get_statistics_constants(config)
    
    for x_array, y_array in pairs:
        correlation, _ = _compute_correlation_by_method(x_array, y_array, method)
        if np.isfinite(correlation):
            clipped_correlation = np.clip(correlation, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
            subject_correlations.append(_safe_float(clipped_correlation))
    
    if not subject_correlations:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan
    
    z_scores = np.arctanh(np.array(subject_correlations))
    r_group = _safe_float(np.tanh(np.nanmean(z_scores)))
    
    p_group = np.nan
    if len(z_scores) >= constants["min_samples_for_stats"]:
        ttest_result = stats.ttest_1samp(z_scores, popmean=0.0, nan_policy="omit")
        p_group = _get_ttest_pvalue(ttest_result)
    
    n_trials = int(sum(len(x) for x, _ in pairs))
    
    ci = (np.nan, np.nan)
    if n_cluster_boot and len(subject_correlations) >= constants["min_samples_for_stats"]:
        bootstrap_values = []
        indices = np.arange(len(subject_correlations))
        for _ in range(int(n_cluster_boot)):
            sampled_indices = rng.choice(indices, size=len(subject_correlations), replace=True)
            z_boot = np.mean(z_scores[sampled_indices])
            bootstrap_values.append(_safe_float(np.tanh(z_boot)))
        
        if bootstrap_values:
            ci = (
                _safe_float(np.percentile(bootstrap_values, constants["ci_percentile_low"])),
                _safe_float(np.percentile(bootstrap_values, constants["ci_percentile_high"])),
            )
    
    return r_group, _safe_float(p_group), n_trials, len(subject_correlations), ci, np.nan


def compute_group_corr_stats(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    method: str,
    *,
    strategy: str,
    n_cluster_boot: int = 0,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, int, Tuple[float, float], float]:
    pairs = _build_valid_pairs(x_lists, y_lists, config=config)
    if not pairs:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan
    
    rng = rng or np.random.default_rng(42)
    
    if strategy in {"pooled_trials", "within_subject_centered", "within_subject_zscored"}:
        return _compute_pooled_strategy_stats(pairs, strategy, method, n_cluster_boot, rng, config=config)
    
    return _compute_fisher_strategy_stats(pairs, method, n_cluster_boot, rng, config=config)


###################################################################
# Permutation and Partial Correlation Utilities
###################################################################

def compute_perm_and_partial_perm(
    x: pd.Series,
    y: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    method: str,
    n_perm: int,
    rng: np.random.Generator,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    p_perm = p_partial_perm = np.nan
    if n_perm <= 0:
        return p_perm, p_partial_perm
    
    p_perm = perm_pval_simple(x, y, method, n_perm, rng, config=config)
    
    if covariates_df is None or covariates_df.empty:
        return p_perm, p_partial_perm
    
    p_partial_perm = perm_pval_partial_freedman_lane(x, y, covariates_df, method, n_perm, rng, config=config)
    return p_perm, p_partial_perm


def compute_partial_correlation_with_covariates(
    roi_values: pd.Series,
    target_values: pd.Series,
    covariates_df: pd.DataFrame,
    method: str,
    context: str,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    if min_samples is None:
        constants = _get_statistics_constants(config)
        min_samples = constants["min_samples_for_correlation"]
    x_aligned, y_aligned, covariates_aligned, _, _ = prepare_aligned_data(
        roi_values,
        target_values,
        covariates_df,
        min_samples=min_samples,
        logger=logger,
        context=context,
    )
    
    if x_aligned is None or covariates_aligned is None:
        return np.nan, np.nan, 0
    
    return partial_corr_xy_given_Z(x_aligned, y_aligned, covariates_aligned, method)


def compute_partial_correlations(
    roi_values: pd.Series,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temperature_series: Optional[pd.Series],
    method: str,
    context: str,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, float, float, int]:
    partial_corr = partial_p = np.nan
    partial_n = 0
    if covariates_df is not None and not covariates_df.empty:
        partial_corr, partial_p, partial_n = compute_partial_correlation_with_covariates(
            roi_values, target_values, covariates_df, method, f"{context} partial", logger, min_samples
        )

    partial_corr_temp = partial_p_temp = np.nan
    partial_n_temp = 0
    if temperature_series is not None and not temperature_series.empty:
        temp_covariates_df = pd.DataFrame({"temp": temperature_series})
        partial_corr_temp, partial_p_temp, partial_n_temp = (
            compute_partial_correlation_with_covariates(
                roi_values,
                target_values,
                temp_covariates_df,
                method,
                f"{context} rating|temp",
                logger,
                min_samples,
            )
        )
    
    return (
        partial_corr,
        partial_p,
        partial_n,
        partial_corr_temp,
        partial_p_temp,
        partial_n_temp,
    )


def compute_permutation_pvalue_partial(
    x_aligned: pd.Series,
    y_aligned: pd.Series,
    covariates_df: pd.DataFrame,
    method: str,
    n_perm: int,
    rng: np.random.Generator,
    context: str = "",
    logger: Optional[logging.Logger] = None,
) -> float:
    from .io_utils import ensure_aligned_lengths
    ensure_aligned_lengths(
        x_aligned, y_aligned, covariates_df, context=context, strict=True
    )
    return perm_pval_partial_freedman_lane(x_aligned, y_aligned, covariates_df, method, n_perm, rng)


def compute_permutation_pvalues(
    x_aligned: pd.Series,
    y_aligned: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    method: str,
    n_perm: int,
    n_eff: int,
    rng: np.random.Generator,
    band: str = "",
    roi: str = "",
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, float]:
    p_perm = p_partial_perm = p_partial_temp_perm = np.nan
    
    if n_perm <= 0 or n_eff < min_samples:
        return p_perm, p_partial_perm, p_partial_temp_perm

    p_perm = perm_pval_simple(x_aligned, y_aligned, method, n_perm, rng)
    
    if covariates_df is not None and not covariates_df.empty:
        context_partial = f"ROI perm partial (band={band}, roi={roi})"
        p_partial_perm = compute_permutation_pvalue_partial(
            x_aligned, y_aligned, covariates_df, method, n_perm, rng, context_partial, logger
        )
    
    if temp_series is not None and not temp_series.empty:
        temp_covariates_df = pd.DataFrame({"temp": temp_series})
        context_temp = f"ROI perm temp (band={band}, roi={roi})"
        p_partial_temp_perm = compute_permutation_pvalue_partial(
            x_aligned, y_aligned, temp_covariates_df, method, n_perm, rng, context_temp, logger
        )
    
    return p_perm, p_partial_perm, p_partial_temp_perm


def compute_temp_permutation_pvalues(
    roi_values: pd.Series,
    temp_values: pd.Series,
    covariates_without_temp_df: Optional[pd.DataFrame],
    method: str,
    n_perm: int,
    rng: np.random.Generator,
    band: str = "",
    roi: str = "",
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    p_temp_perm = np.nan
    p_temp_partial_perm = np.nan
    
    if n_perm <= 0:
        return p_temp_perm, p_temp_partial_perm
    
    p_temp_perm = perm_pval_simple(roi_values, temp_values, method, n_perm, rng, config=config)
    
    if covariates_without_temp_df is None or covariates_without_temp_df.empty:
        return p_temp_perm, p_temp_partial_perm
    
    from .io_utils import ensure_aligned_lengths
    context_partial = f"ROI temp perm partial (band={band}, roi={roi})"
    ensure_aligned_lengths(
        roi_values,
        temp_values,
        covariates_without_temp_df,
        context=context_partial,
        strict=True,
    )
    p_temp_partial_perm = perm_pval_partial_freedman_lane(
        roi_values, temp_values, covariates_without_temp_df, method, n_perm, rng, config=config
    )
    
    return p_temp_perm, p_temp_partial_perm


def compute_channel_rating_correlations(
    channel_values: pd.Series,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, float, float, float, float, int, float, float]:
    if channel_values.empty or target_values.empty:
        return (
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan
        )
    
    correlation, p_value = compute_correlation(channel_values, target_values, use_spearman)
    ci_low, ci_high = compute_bootstrap_ci(
        channel_values,
        target_values,
        bootstrap,
        use_spearman,
        rng,
        min_samples,
        logger=logger,
        config=config,
    )
    
    r_partial = p_partial = np.nan
    n_partial = 0
    if covariates_df is not None and not covariates_df.empty:
        r_partial, p_partial, n_partial = partial_corr_xy_given_Z(
            channel_values, target_values, covariates_df, method, config=config
        )
    
    p_perm, p_partial_perm = compute_perm_and_partial_perm(
        channel_values, target_values, covariates_df, method, n_perm, rng, config=config
    )
    
    return (
        correlation,
        p_value,
        ci_low,
        ci_high,
        r_partial,
        p_partial,
        n_partial,
        p_perm,
        p_partial_perm,
    )


def compute_partial_correlation_for_roi_pair(
    x_masked: pd.Series,
    y_masked: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    mask: pd.Series,
    method: str,
) -> Tuple[float, float, int]:
    r_partial = p_partial = np.nan
    n_partial = 0
    
    if covariates_df is None or covariates_df.empty:
        return r_partial, p_partial, n_partial
    
    covariates_valid = covariates_df.iloc[mask]
    if covariates_valid.empty:
        return r_partial, p_partial, n_partial
    
    r_partial, p_partial, n_partial = partial_corr_xy_given_Z(
        x_masked, y_masked, covariates_valid, method
    )
    
    return r_partial, p_partial, n_partial


def compute_permutation_pvalues_for_roi_pair(
    x_masked: pd.Series,
    y_masked: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    mask: pd.Series,
    method: str,
    n_perm: int,
    n_eff: int,
    rng: np.random.Generator,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    p_perm = p_partial_perm = np.nan
    
    if n_perm <= 0 or n_eff < min_samples:
        return p_perm, p_partial_perm

    covariates_valid = None
    if covariates_df is not None and not covariates_df.empty:
        covariates_valid = covariates_df.iloc[mask]
        if covariates_valid.empty:
            covariates_valid = None

    p_perm, p_partial_perm = compute_perm_and_partial_perm(
        x_masked, y_masked, covariates_valid, method, int(n_perm), rng, config=config
    )
    return p_perm, p_partial_perm


def compute_temp_correlations_for_roi(
    roi_values: pd.Series,
    temp_values: pd.Series,
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    band: str = "",
    roi: str = "",
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    from .io_utils import ensure_aligned_lengths, build_partial_covars_string
    context_temp = f"ROI temperature stats (band={band}, roi={roi})"
    ensure_aligned_lengths(
        roi_values, temp_values, context=context_temp, strict=True
    )
    
    if min_samples is None:
        constants = _get_statistics_constants(config)
        min_samples = constants["min_samples_for_correlation"]
    valid_mask = roi_values.notna() & temp_values.notna()
    n_valid = int(valid_mask.sum())
    if n_valid < min_samples:
        return None

    correlation, p_value = compute_correlation(
        roi_values[valid_mask], temp_values[valid_mask], use_spearman
    )
    ci_low, ci_high = compute_bootstrap_ci(
        roi_values,
        temp_values,
        bootstrap,
        use_spearman,
        rng,
        min_samples,
        logger=logger,
        config=config,
    )

    p_perm, p_partial_perm = compute_temp_permutation_pvalues(
        roi_values,
        temp_values,
        covariates_without_temp_df,
        method,
        n_perm,
        rng,
        band=band,
        roi=roi,
        logger=logger,
        config=config,
    )

    partial_covars_str = build_partial_covars_string(covariates_without_temp_df)

    return {
        "roi": roi,
        "band": band,
        "band_range": "",
        "r": correlation,
        "p": p_value,
        "n": n_valid,
        "method": method,
        "r_ci_low": ci_low,
        "r_ci_high": ci_high,
        "r_partial": np.nan,
        "p_partial": np.nan,
        "n_partial": 0,
        "partial_covars": partial_covars_str,
        "p_perm": _safe_float(p_perm),
        "p_partial_perm": _safe_float(p_partial_perm),
        "n_perm": n_perm,
    }


def compute_correlation_for_time_freq_bin(
    power: np.ndarray,
    y_array: np.ndarray,
    times: np.ndarray,
    f_idx: int,
    t_start: float,
    t_end: float,
    min_valid_points: int,
    use_spearman: bool,
) -> Tuple[Optional[float], Optional[float], int]:
    time_mask = (times >= t_start) & (times < t_end)
    if not np.any(time_mask):
        return None, None, 0
    
    mean_power = power[:, f_idx, time_mask].mean(axis=1)
    valid_mask = np.isfinite(mean_power) & np.isfinite(y_array)
    n_observations = int(valid_mask.sum())
    
    if n_observations < min_valid_points:
        return None, None, n_observations
    
    power_valid = mean_power[valid_mask]
    y_valid = y_array[valid_mask]
    correlation, p_value = compute_correlation(power_valid, y_valid, use_spearman)
    return correlation, p_value, n_observations


def compute_correlation_from_vectors(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    min_valid_points: int,
    use_spearman: bool,
) -> Tuple[float, float]:
    mask = np.isfinite(x_vec) & np.isfinite(y_vec)
    if mask.sum() < min_valid_points:
        return np.nan, np.nan
    return compute_correlation(x_vec[mask], y_vec[mask], use_spearman)


###################################################################
# Cluster Correction Utilities
###################################################################

def compute_cluster_masses_2d(
    correlation_matrix: np.ndarray,
    pvalue_matrix: np.ndarray,
    cluster_alpha: float,
    cluster_structure: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    if cluster_structure is None:
        constants = _get_statistics_constants(config)
        cluster_structure = constants["cluster_structure_2d"]
    
    significant_mask = np.isfinite(correlation_matrix) & np.isfinite(pvalue_matrix) & (pvalue_matrix < cluster_alpha)
    cluster_labels, n_clusters = label(significant_mask, structure=cluster_structure)
    
    cluster_masses: Dict[int, float] = {}
    for cluster_id in range(1, n_clusters + 1):
        cluster_region = (cluster_labels == cluster_id)
        if not cluster_region.any():
            continue
        cluster_mass = float(np.nansum(np.abs(correlation_matrix)[cluster_region]))
        cluster_masses[cluster_id] = cluster_mass
    
    return cluster_labels, cluster_masses


def compute_permutation_max_masses(
    bin_data: np.ndarray,
    informative_bins: List[Tuple[int, int]],
    y_array: np.ndarray,
    correlations_shape: Tuple[int, ...],
    cluster_alpha: float,
    min_valid_points: int,
    use_spearman: bool,
    n_cluster_perm: int,
    cluster_rng: np.random.Generator,
    cluster_structure: Optional[np.ndarray] = None,
) -> List[float]:
    permutation_max_masses: List[float] = []
    if n_cluster_perm <= 0:
        return permutation_max_masses

    permuted_correlations = np.full(correlations_shape, np.nan)
    permuted_pvalues = np.full(correlations_shape, np.nan)
    
    for _ in range(n_cluster_perm):
        permuted_correlations.fill(np.nan)
        permuted_pvalues.fill(np.nan)
        y_permuted = cluster_rng.permutation(y_array)
        
        for freq_idx, time_idx in informative_bins:
            bin_vector = bin_data[freq_idx, time_idx, :]
            correlation, p_value = compute_correlation_from_vectors(
                bin_vector, y_permuted, min_valid_points, use_spearman
            )
            permuted_correlations[freq_idx, time_idx] = correlation
            permuted_pvalues[freq_idx, time_idx] = p_value
        
        _, permuted_masses = compute_cluster_masses_2d(
            permuted_correlations, permuted_pvalues, cluster_alpha, cluster_structure
        )
        max_mass = max(permuted_masses.values()) if permuted_masses else 0.0
        permutation_max_masses.append(max_mass)
    
    return permutation_max_masses


def _create_cluster_record(cluster_id: int, mass: float, size: int, p_value: float) -> Dict[str, Any]:
    return {
        "cluster_id": int(cluster_id),
        "mass": mass,
        "size": size,
        "p_value": _safe_float(p_value),
    }


def _compute_cluster_pvalue(mass: float, perm_max_masses_array: np.ndarray, denominator: float) -> float:
    exceedances = np.sum(perm_max_masses_array >= mass)
    return (exceedances + 1) / denominator


def compute_cluster_pvalues(
    cluster_labels_obs: np.ndarray,
    cluster_masses: Dict[int, float],
    perm_max_masses: List[float],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    cluster_pvals = np.full_like(cluster_labels_obs, np.nan, dtype=float)
    cluster_sig_mask = np.zeros_like(cluster_labels_obs, dtype=bool)
    cluster_records: List[Dict[str, Any]] = []
    
    if not perm_max_masses:
        for cid, mass in cluster_masses.items():
            cluster_region = (cluster_labels_obs == cid)
            size = int(cluster_region.sum())
            cluster_records.append(_create_cluster_record(cid, mass, size, np.nan))
        return cluster_pvals, cluster_sig_mask, cluster_records
    
    denominator = len(perm_max_masses) + 1
    perm_max_masses_array = np.asarray(perm_max_masses)
    
    for cid, mass in cluster_masses.items():
        cluster_region = (cluster_labels_obs == cid)
        size = int(cluster_region.sum())
        p_cluster = _compute_cluster_pvalue(mass, perm_max_masses_array, denominator)
        
        cluster_pvals[cluster_region] = p_cluster
        if p_cluster <= alpha:
            cluster_sig_mask[cluster_region] = True
        
        cluster_records.append(_create_cluster_record(cid, mass, size, p_cluster))
    
    return cluster_pvals, cluster_sig_mask, cluster_records


def compute_cluster_correction_2d(
    correlations: np.ndarray,
    p_values: np.ndarray,
    bin_data: np.ndarray,
    informative_bins: List[Tuple[int, int]],
    y_array: np.ndarray,
    cluster_alpha: float,
    n_cluster_perm: int,
    alpha: float,
    min_valid_points: int,
    use_spearman: bool,
    cluster_rng: np.random.Generator,
    cluster_structure: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[float]]:
    cluster_labels = np.zeros_like(correlations, dtype=int)
    cluster_pvals = np.full_like(correlations, np.nan)
    cluster_sig_mask = np.zeros_like(correlations, dtype=bool)
    cluster_records: List[Dict[str, Any]] = []

    if not informative_bins:
        return cluster_labels, cluster_pvals, cluster_sig_mask, cluster_records, []

    cluster_labels_obs, cluster_masses = compute_cluster_masses_2d(
        correlations, p_values, cluster_alpha, cluster_structure
    )
    if not cluster_masses:
        return cluster_labels_obs, cluster_pvals, cluster_sig_mask, cluster_records, []

    cluster_labels = cluster_labels_obs
    
    perm_max_masses = compute_permutation_max_masses(
        bin_data,
        informative_bins,
        y_array,
        correlations.shape,
        cluster_alpha,
        min_valid_points,
        use_spearman,
        n_cluster_perm,
        cluster_rng,
        cluster_structure,
    )

    cluster_pvals, cluster_sig_mask, cluster_records = compute_cluster_pvalues(
        cluster_labels_obs, cluster_masses, perm_max_masses, alpha
    )

    return cluster_labels, cluster_pvals, cluster_sig_mask, cluster_records, perm_max_masses


def compute_cluster_masses_1d(
    correlation_vector: np.ndarray,
    pvalue_vector: np.ndarray,
    cluster_alpha: float,
    ch_to_eeg_idx: Dict[int, int],
    eeg_picks: np.ndarray,
    adjacency: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, float]]:
    from scipy.sparse.csgraph import connected_components

    significant_mask = np.isfinite(correlation_vector) & np.isfinite(pvalue_vector) & (pvalue_vector < cluster_alpha)
    if not np.any(significant_mask):
        return np.zeros(len(correlation_vector), dtype=int), {}

    eeg_mask = np.zeros(len(eeg_picks), dtype=bool)
    for channel_idx, eeg_idx in ch_to_eeg_idx.items():
        if significant_mask[channel_idx]:
            eeg_mask[eeg_idx] = True

    if not np.any(eeg_mask):
        return np.zeros(len(correlation_vector), dtype=int), {}

    eeg_indices = np.where(eeg_mask)[0]
    if len(eeg_indices) < 2:
        return np.zeros(len(correlation_vector), dtype=int), {}

    adjacency_subset = adjacency[eeg_indices, :][:, eeg_indices]
    n_components, eeg_labels = connected_components(
        csgraph=adjacency_subset, directed=False, return_labels=True
    )

    full_labels = np.zeros(len(correlation_vector), dtype=int)
    eeg_to_channel = {eeg_idx: ch_idx for ch_idx, eeg_idx in ch_to_eeg_idx.items()}
    for local_idx, global_eeg_idx in enumerate(eeg_indices):
        if global_eeg_idx in eeg_to_channel:
            channel_idx = eeg_to_channel[global_eeg_idx]
            full_labels[channel_idx] = eeg_labels[local_idx] + 1

    cluster_masses: Dict[int, float] = {}
    for cluster_id in range(1, n_components + 1):
        cluster_region = (full_labels == cluster_id)
        if not cluster_region.any():
            continue
        cluster_mass = float(np.nansum(np.abs(correlation_vector)[cluster_region]))
        cluster_masses[cluster_id] = cluster_mass

    return full_labels, cluster_masses


def compute_topomap_permutation_masses(
    channel_data: np.ndarray,
    temp_series: pd.Series,
    n_channels: int,
    n_cluster_perm: int,
    cluster_alpha: float,
    min_valid_points: int,
    use_spearman: bool,
    ch_to_eeg_idx: Dict[int, int],
    eeg_picks: np.ndarray,
    adjacency: np.ndarray,
    cluster_rng: np.random.Generator,
) -> List[float]:
    permutation_max_masses: List[float] = []
    if n_cluster_perm <= 0:
        return permutation_max_masses

    temp_array = temp_series.to_numpy(dtype=float)
    for _ in range(n_cluster_perm):
        temp_permuted = cluster_rng.permutation(temp_array)
        permuted_correlations = np.full(n_channels, np.nan)
        permuted_pvalues = np.full(n_channels, np.nan)

        for channel_idx in range(n_channels):
            channel_vector = channel_data[channel_idx, :]
            correlation, p_value = compute_correlation_from_vectors(
                channel_vector, temp_permuted, min_valid_points, use_spearman
            )
            permuted_correlations[channel_idx] = correlation
            permuted_pvalues[channel_idx] = p_value

        _, permuted_masses = compute_cluster_masses_1d(
            permuted_correlations, permuted_pvalues, cluster_alpha, ch_to_eeg_idx, eeg_picks, adjacency
        )
        max_mass = max(permuted_masses.values()) if permuted_masses else 0.0
        permutation_max_masses.append(max_mass)

    return permutation_max_masses


def compute_cluster_pvalues_1d(
    cluster_labels_obs: np.ndarray,
    cluster_masses: Dict[int, float],
    perm_max_masses: List[float],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    return compute_cluster_pvalues(cluster_labels_obs, cluster_masses, perm_max_masses, alpha)


def compute_temp_correlation_for_roi_pair(
    xi: pd.Series,
    temp_series: pd.Series,
    covariates_without_temp_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    measure_band: str,
    roi_i: str,
    roi_j: str,
    n_edges: int,
    rng: np.random.Generator,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    from .io_utils import build_partial_covars_string
    from .tfr_utils import get_summary_type
    
    if len(xi) != len(temp_series):
        if logger:
            logger.warning(
                f"Channel vs temp length mismatch: power={len(xi)}, "
                f"temp={len(temp_series)}. Using overlap."
            )
    
    min_len = min(len(xi), len(temp_series))
    xi_aligned = xi.iloc[:min_len]
    temp_aligned = temp_series.iloc[:min_len]
    if min_samples is None:
        constants = _get_statistics_constants(config)
        min_samples = constants["min_samples_for_correlation"]
    mask_temp = joint_valid_mask(xi_aligned, temp_aligned)
    n_eff_temp = int(mask_temp.sum())
    
    if n_eff_temp < min_samples:
        return None

    method_temp = get_correlation_method(use_spearman)
    correlation_temp, p_value_temp = compute_correlation(
        xi_aligned.iloc[mask_temp], temp_aligned.iloc[mask_temp], use_spearman
    )

    r_partial_temp, p_partial_temp, n_partial_temp = (
        compute_partial_correlation_for_roi_pair(
            xi_aligned.iloc[mask_temp],
            temp_aligned.iloc[mask_temp],
            covariates_without_temp_df,
            mask_temp,
            method_temp,
        )
    )

    ci_temp_low, ci_temp_high = compute_bootstrap_ci(
        xi_aligned,
        temp_aligned,
        bootstrap,
        use_spearman,
        rng,
        min_samples,
        logger=logger,
        config=config,
    )

    p_perm_temp, p_partial_perm_temp = compute_permutation_pvalues_for_roi_pair(
        xi_aligned.iloc[mask_temp],
        temp_aligned.iloc[mask_temp],
        covariates_without_temp_df,
        mask_temp,
        method_temp,
        n_perm,
        n_eff_temp,
        rng,
        min_samples,
    )

    partial_covars_temp_str = build_partial_covars_string(covariates_without_temp_df)

    return {
        "measure_band": measure_band,
        "roi_i": roi_i,
        "roi_j": roi_j,
        "summary_type": get_summary_type(roi_i, roi_j),
        "n_edges": n_edges,
        "r": correlation_temp,
        "p": p_value_temp,
        "n": n_eff_temp,
        "method": method_temp,
        "r_ci_low": ci_temp_low,
        "r_ci_high": ci_temp_high,
        "r_partial": _safe_float(r_partial_temp),
        "p_partial": _safe_float(p_partial_temp),
        "n_partial": n_partial_temp,
        "partial_covars": partial_covars_temp_str,
        "p_perm": _safe_float(p_perm_temp),
        "p_partial_perm": _safe_float(p_partial_perm_temp),
        "n_perm": n_perm,
    }


###################################################################
# Summary Statistics Utilities
###################################################################

def compute_band_summary_statistics(band_data: pd.Series, config: Optional[Any] = None) -> Tuple[float, float, float, int]:
    if band_data.empty:
        return np.nan, np.nan, np.nan, 0
    
    valid_values = band_data[np.isfinite(band_data)].to_numpy(dtype=float)
    if valid_values.size == 0:
        return np.nan, np.nan, np.nan, 0
    
    mean = float(np.mean(valid_values))
    n = len(valid_values)
    
    if n <= 1:
        return mean, np.nan, np.nan, n
    
    std = float(np.std(valid_values, ddof=1))
    se = std / np.sqrt(n)
    constants = _get_statistics_constants(config)
    delta = constants["ci_multiplier_95"] * se if np.isfinite(se) else np.nan
    
    ci_low = mean - delta if np.isfinite(delta) else np.nan
    ci_high = mean + delta if np.isfinite(delta) else np.nan
    
    return mean, ci_low, ci_high, n


def compute_band_summaries(means_df: pd.DataFrame, bands_present: List[str]) -> pd.DataFrame:
    if means_df.empty or not bands_present:
        return pd.DataFrame()
    
    summaries = []
    for band in bands_present:
        band_values = means_df[means_df["band"] == band]["mean_power"]
        mean, ci_low, ci_high, n = compute_band_summary_statistics(band_values)
        summaries.append({
            "band": band,
            "group_mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_subjects": n,
        })
    return pd.DataFrame(summaries)


###################################################################
# Aperiodic Fitting Utilities
###################################################################

def fit_aperiodic(log_freqs: np.ndarray, log_psd: np.ndarray) -> Tuple[float, float]:
    try:
        slope, intercept = np.polyfit(log_freqs, log_psd, 1)
        return float(intercept), float(slope)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), float("nan")


def fit_aperiodic_to_all_epochs(log_freqs: np.ndarray, log_psd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_epochs, n_channels, _ = log_psd.shape
    offsets = np.full((n_epochs, n_channels), np.nan)
    slopes = np.full((n_epochs, n_channels), np.nan)
    
    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            intercept, slope = fit_aperiodic(log_freqs, log_psd[epoch_idx, channel_idx, :])
            offsets[epoch_idx, channel_idx] = intercept
            slopes[epoch_idx, channel_idx] = slope
    
    return offsets, slopes


def compute_residuals(log_freqs: np.ndarray, log_psd: np.ndarray, offsets: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    n_epochs, n_channels, n_freqs = log_psd.shape
    residuals = np.empty_like(log_psd)
    
    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            fitted = offsets[epoch_idx, channel_idx] + slopes[epoch_idx, channel_idx] * log_freqs
            residuals[epoch_idx, channel_idx, :] = log_psd[epoch_idx, channel_idx, :] - fitted
    
    return residuals


###################################################################
# Linear Regression Residuals
###################################################################

def compute_linear_residuals(x_data: pd.Series, y_data: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_series = pd.to_numeric(x_data, errors="coerce")
    y_series = pd.to_numeric(y_data, errors="coerce")
    mask = x_series.notna() & y_series.notna()
    x_clean = x_series[mask].to_numpy(dtype=float)
    y_clean = y_series[mask].to_numpy(dtype=float)
    slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
    fitted = intercept + slope * x_clean
    residuals = y_clean - fitted
    return fitted, residuals, x_clean


###################################################################
# Regression and Binned Statistics
###################################################################

def extract_finite_mask(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask], mask


def fit_linear_regression(x: np.ndarray, y: np.ndarray, x_range: np.ndarray, min_samples: Optional[int] = None, config: Optional[Any] = None) -> np.ndarray:
    if min_samples is None:
        constants = _get_statistics_constants(config)
        min_samples = constants["min_samples_for_stats"]
    if len(x) < min_samples:
        return np.full_like(x_range, np.nan)
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    return polynomial(x_range)


def compute_binned_statistics(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int) -> Tuple[List[float], List[float], List[float]]:
    bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_stds = []
    
    for i in range(n_bins):
        is_last_bin = i == n_bins - 1
        mask_bin = (y_pred >= bins[i]) & (y_pred <= bins[i+1] if is_last_bin else y_pred < bins[i+1])
        
        if mask_bin.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(y_true[mask_bin]))
            bin_stds.append(np.std(y_true[mask_bin]) / np.sqrt(mask_bin.sum()))
    
    return bin_centers, bin_means, bin_stds


def compute_error_bars_from_ci_dicts(values: List[float], ci_dicts: List[Optional[Dict[str, List[float]]]]) -> Tuple[List[float], List[float]]:
    errors_lower = []
    errors_upper = []
    for val, ci_dict in zip(values, ci_dicts):
        if ci_dict is None:
            errors_lower.append(0)
            errors_upper.append(0)
            continue
        
        ci = ci_dict.get('ci95', [np.nan, np.nan])
        lower = val - ci[0] if np.isfinite(ci[0]) else 0
        upper = ci[1] - val if np.isfinite(ci[1]) else 0
        errors_lower.append(lower)
        errors_upper.append(upper)
    return errors_lower, errors_upper


def compute_consensus_labels(labels_all_trials: List[np.ndarray], n_timepoints: int) -> np.ndarray:
    n_trials = len(labels_all_trials)
    labels_consensus = np.zeros(n_timepoints, dtype=int)
    
    for time_idx in range(n_timepoints):
        state_counts = np.bincount([
            labels_all_trials[trial][time_idx]
            for trial in range(n_trials)
        ])
        labels_consensus[time_idx] = np.argmax(state_counts)
    
    return labels_consensus


def compute_inter_band_coupling_matrix(tfr_avg, band_names: List[str], features_freq_bands: Dict[str, Tuple[float, float]], extract_band_channel_means_func) -> np.ndarray:
    n_bands = len(band_names)
    coupling_matrix = np.zeros((n_bands, n_bands))
    
    for i, band1 in enumerate(band_names):
        fmin1, fmax1 = features_freq_bands[band1]
        freq_mask1 = (tfr_avg.freqs >= fmin1) & (tfr_avg.freqs <= fmax1)
        if not freq_mask1.any():
            continue
        
        coupling_matrix[i, i] = 1.0
        band1_channels = extract_band_channel_means_func(tfr_avg, freq_mask1)
        
        for j in range(i + 1, n_bands):
            band2 = band_names[j]
            fmin2, fmax2 = features_freq_bands[band2]
            freq_mask2 = (tfr_avg.freqs >= fmin2) & (tfr_avg.freqs <= fmax2)
            if not freq_mask2.any():
                continue
            
            band2_channels = extract_band_channel_means_func(tfr_avg, freq_mask2)
            correlation = compute_band_spatial_correlation(band1_channels, band2_channels)
            coupling_matrix[i, j] = correlation
            coupling_matrix[j, i] = correlation
    
    return coupling_matrix


def compute_group_channel_power_statistics(subj_pow: Dict[str, pd.DataFrame], bands: List[str], all_channels: List[str]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    heatmap_rows = []
    statistics_rows = []
    for band in bands:
        band_str = str(band)
        subject_means_per_channel = []
        for _, df in subj_pow.items():
            values = []
            for channel in all_channels:
                col = f"pow_{band_str}_{channel}"
                if col in df.columns:
                    mean_val = float(pd.to_numeric(df[col], errors="coerce").mean())
                    values.append(mean_val)
                else:
                    values.append(np.nan)
            subject_means_per_channel.append(values)
        array = np.asarray(subject_means_per_channel, dtype=float)
        mean_across_subjects = np.nanmean(array, axis=0)
        heatmap_rows.append(mean_across_subjects)
        n_effective = np.sum(np.isfinite(array), axis=0)
        std_across_subjects = np.nanstd(array, axis=0, ddof=1)
        for j, channel in enumerate(all_channels):
            statistics_rows.append({
                "band": band_str,
                "channel": channel,
                "mean": float(mean_across_subjects[j]) if np.isfinite(mean_across_subjects[j]) else np.nan,
                "std": float(std_across_subjects[j]) if np.isfinite(std_across_subjects[j]) else np.nan,
                "n_subjects": int(n_effective[j])
            })
    return heatmap_rows, statistics_rows


def compute_group_band_statistics(df: pd.DataFrame, bands: List[str], ci_multiplier: Optional[float] = None, config: Optional[Any] = None) -> Tuple[List[str], List[float], List[float], List[float], List[int]]:
    if ci_multiplier is None:
        constants = _get_statistics_constants(config)
        ci_multiplier = constants["ci_multiplier_95"]
    bands_present = [b for b in bands if b in set(df["band"])]
    means = []
    ci_lower = []
    ci_upper = []
    n_subjects = []
    
    for band in bands_present:
        values = df[df["band"] == band]["mean_power"].to_numpy(dtype=float)
        mean_val, ci_low, ci_high, n = compute_band_statistics_array(values, ci_multiplier=ci_multiplier, config=config)
        means.append(mean_val)
        ci_lower.append(ci_low)
        ci_upper.append(ci_high)
        n_subjects.append(n)
    
    return bands_present, means, ci_lower, ci_upper, n_subjects


def compute_error_bars_from_arrays(means: List[float], ci_lower: List[float], ci_upper: List[float]) -> np.ndarray:
    yerr_lower = [
        mu - lo if np.isfinite(mu) and np.isfinite(lo) else 0
        for lo, mu in zip(ci_lower, means)
    ]
    yerr_upper = [
        hi - mu if np.isfinite(mu) and np.isfinite(hi) else 0
        for hi, mu in zip(ci_upper, means)
    ]
    return np.array([yerr_lower, yerr_upper])


def compute_band_pair_correlation(vec_i: Dict[str, float], vec_j: Dict[str, float]) -> float:
    if vec_i is None or vec_j is None:
        return np.nan
    
    common_channels = sorted(set(vec_i.keys()) & set(vec_j.keys()))
    if len(common_channels) < 2:
        return np.nan
    
    values_i = np.array([vec_i[ch] for ch in common_channels], dtype=float)
    values_j = np.array([vec_j[ch] for ch in common_channels], dtype=float)
    
    if np.std(values_i) < 1e-12 or np.std(values_j) < 1e-12:
        return np.nan
    
    return float(np.corrcoef(values_i, values_j)[0, 1])


def compute_subject_band_correlation_matrix(band_vectors: Dict[str, Dict[str, float]], band_names: List[str]) -> np.ndarray:
    n_bands = len(band_names)
    correlation_matrix = np.eye(n_bands, dtype=float)
    
    for i, band_i in enumerate(band_names):
        vec_i = band_vectors.get(band_i)
        for j in range(i + 1, n_bands):
            band_j = band_names[j]
            vec_j = band_vectors.get(band_j)
            correlation = compute_band_pair_correlation(vec_i, vec_j)
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
    
    return correlation_matrix


def compute_group_band_correlation_matrix(per_subject_correlations: List[np.ndarray], n_bands: int) -> np.ndarray:
    group_correlation = np.eye(n_bands, dtype=float)
    correlation_array = np.stack(per_subject_correlations, axis=0)
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            r_values = correlation_array[:, i, j]
            r_values = r_values[np.isfinite(r_values)]
            
            if r_values.size == 0:
                group_correlation[i, j] = np.nan
                group_correlation[j, i] = np.nan
            else:
                r_mean = fisher_z_transform_mean(r_values)
                group_correlation[i, j] = r_mean
                group_correlation[j, i] = r_mean
    
    return group_correlation


def compute_correlation_ci_fisher(z_mean: float, se: float, ci_multiplier: Optional[float] = None, config: Optional[Any] = None) -> Tuple[float, float]:
    if not np.isfinite(se):
        return np.nan, np.nan
    if ci_multiplier is None:
        constants = _get_statistics_constants(config)
        ci_multiplier = constants["ci_multiplier_95"]
    z_lower = z_mean - ci_multiplier * se
    z_upper = z_mean + ci_multiplier * se
    ci_lower = float(np.tanh(z_lower))
    ci_upper = float(np.tanh(z_upper))
    return ci_lower, ci_upper


def compute_inter_band_correlation_statistics(
    per_subject_correlations: List[np.ndarray], band_names: List[str], ci_multiplier: Optional[float] = None, config: Optional[Any] = None
) -> List[Dict[str, Any]]:
    if ci_multiplier is None:
        constants = _get_statistics_constants(config)
        ci_multiplier = constants["ci_multiplier_95"]
    rows = []
    n_bands = len(band_names)
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            r_values = np.array([cm[i, j] for cm in per_subject_correlations], dtype=float)
            r_values = r_values[np.isfinite(r_values)]
            if r_values.size == 0:
                continue
            
            constants = _get_statistics_constants(config)
            z_scores = np.arctanh(np.clip(r_values, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"]))
            z_mean = float(np.mean(z_scores))
            n = len(z_scores)
            se = float(np.std(z_scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            ci_lower, ci_upper = compute_correlation_ci_fisher(z_mean, se, ci_multiplier)
            
            rows.append({
                "band_i": band_names[i],
                "band_j": band_names[j],
                "r_group": float(np.tanh(z_mean)),
                "r_ci_low": ci_lower,
                "r_ci_high": ci_upper,
                "n_subjects": int(n)
            })
    
    return rows


###################################################################
# ROI Statistics Utilities
###################################################################


def compute_roi_percentage_change(roi_data: np.ndarray, is_percent_format: bool, config: Optional[Any] = None) -> float:
    roi_mean = float(np.nanmean(roi_data))
    
    if is_percent_format:
        return roi_mean
    
    constants = _get_statistics_constants(config)
    return (constants["log_base"] ** roi_mean - 1.0) * constants["percentage_multiplier"]


def compute_roi_pvalue(
    mask_vec: np.ndarray,
    ch_names: List[str],
    p_ch: Optional[np.ndarray],
    sig_mask: Optional[np.ndarray],
    is_cluster: bool,
    cluster_p_min: Optional[float],
    data_group_a: Optional[np.ndarray] = None,
    data_group_b: Optional[np.ndarray] = None,
    paired: bool = False,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Optional[float]:
    from scipy.stats import ttest_rel, ttest_ind
    
    if data_group_a is None or data_group_b is None:
        return None
    
    try:
        roi_data_a = data_group_a[:, mask_vec]
        roi_data_b = data_group_b[:, mask_vec]
        
        roi_vals_a = np.nanmean(roi_data_a, axis=1)
        roi_vals_b = np.nanmean(roi_data_b, axis=1)
        
        valid_a = np.isfinite(roi_vals_a)
        valid_b = np.isfinite(roi_vals_b)
        
        if np.sum(valid_a) < min_samples or np.sum(valid_b) < min_samples:
            return None
        
        if paired and len(roi_vals_a) == len(roi_vals_b):
            valid_both = valid_a & valid_b
            res = ttest_rel(roi_vals_a[valid_both], roi_vals_b[valid_both], nan_policy="omit")
        else:
            res = ttest_ind(roi_vals_a[valid_a], roi_vals_b[valid_b], nan_policy="omit")
        
        if res.pvalue is not None and np.isfinite(res.pvalue):
            return float(res.pvalue)
    except Exception:
        pass
    
    return None


###################################################################
# Data Transformation Utilities
###################################################################

def center_series(series: pd.Series) -> pd.Series:
    return series - series.mean()


def zscore_series(series: pd.Series) -> pd.Series:
    std_val = series.std(ddof=1)
    if std_val <= 0:
        return pd.Series(dtype=float)
    return (series - series.mean()) / std_val


def apply_pooling_strategy(
    x: pd.Series,
    y: pd.Series,
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series]:
    if pooling_strategy == "within_subject_centered":
        return center_series(x), center_series(y)
    if pooling_strategy == "within_subject_zscored":
        x_z = zscore_series(x)
        y_z = zscore_series(y)
        return x_z, y_z
    return x, y


###################################################################
# Correlation Statistics
###################################################################

def compute_correlation_stats(
    x: pd.Series,
    y: pd.Series,
    method_code: str,
    bootstrap_ci: int,
    rng: Optional[np.random.Generator],
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, Tuple[float, float]]:
    mask = joint_valid_mask(x, y)
    n_eff = int(mask.sum())
    r_val, p_val = np.nan, np.nan
    ci_val = (np.nan, np.nan)
    
    if n_eff >= min_samples:
        x_vals = x.iloc[mask] if isinstance(x, pd.Series) else x[mask]
        y_vals = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
        r_val, p_val = stats.spearmanr(x_vals, y_vals, nan_policy="omit")
        if bootstrap_ci > 0:
            ci_val = bootstrap_corr_ci(x_vals, y_vals, method_code, n_boot=bootstrap_ci, rng=rng)
    
    return r_val, p_val, n_eff, ci_val


def compute_partial_residuals_stats(
    x_res: pd.Series,
    y_res: pd.Series,
    stats_df: Optional[pd.Series],
    n_res: int,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int, Tuple[float, float]]:
    r_resid = np.nan
    p_resid = np.nan
    n_partial = n_res
    
    if stats_df is not None:
        r_resid = _safe_float(stats_df.get("r_partial", r_resid))
        p_resid = _safe_float(stats_df.get("p_partial", p_resid))
        n_partial = int(stats_df.get("n_partial", n_partial))
    
    if not np.isfinite(r_resid) or not np.isfinite(p_resid):
        r_resid, p_resid = stats.spearmanr(x_res, y_res, nan_policy="omit")
    
    ci_resid = (np.nan, np.nan)
    if bootstrap_ci > 0:
        ci_resid = bootstrap_corr_ci(x_res, y_res, method_code, n_boot=bootstrap_ci, rng=rng)
    
    return r_resid, p_resid, n_partial, ci_resid


###################################################################
# Band and Connectivity Correlations
###################################################################

def compute_band_correlations(
    pow_df: pd.DataFrame,
    y: pd.Series,
    band: str,
    power_prefix: str = "pow_",
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    band_cols = [col for col in pow_df.columns if col.startswith(f'{power_prefix}{band}_')]
    if not band_cols:
        return [], np.array([]), np.array([])
    
    channel_names = [col.replace(f'{power_prefix}{band}_', '') for col in band_cols]
    correlations = []
    p_values = []
    
    for col in band_cols:
        valid_data = pd.concat([pow_df[col], y], axis=1).dropna()
        if len(valid_data) >= min_samples:
            correlation, p_value = stats.spearmanr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
        else:
            correlation, p_value = np.nan, 1.0
        correlations.append(correlation)
        p_values.append(p_value)
    
    return channel_names, np.array(correlations), np.array(p_values)


def compute_connectivity_correlations(
    conn_df: pd.DataFrame,
    y: pd.Series,
    measure_cols: List[str],
    measure: str,
    band: str,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
    min_correlation: float = 0.3,
    max_pvalue: float = 0.05,
) -> Tuple[List[float], List[str]]:
    correlations = []
    connections = []
    
    prefix = f'{measure}_{band}_'
    for col in measure_cols:
        valid_mask = ~(conn_df[col].isna() | y.isna())
        if valid_mask.sum() < min_samples:
            continue
        
        x_vals = conn_df[col][valid_mask].to_numpy()
        y_vals = y[valid_mask].to_numpy()
        if np.std(x_vals) <= 0 or np.std(y_vals) <= 0:
            continue
            
        correlation, p_value = stats.spearmanr(x_vals, y_vals)
        if abs(correlation) > min_correlation and p_value < max_pvalue:
            correlations.append(correlation)
            connection_pair = col.replace(prefix, '').replace('conn_', '')
            connections.append(connection_pair)
    
    return correlations, connections


###################################################################
# Visualization Utilities
###################################################################

def compute_kde_scale(
    data: pd.Series,
    hist_bins: int = 15,
    kde_points: int = 100,
) -> float:
    hist_counts, _ = np.histogram(data, bins=hist_bins)
    kde = gaussian_kde(data)
    data_range = np.linspace(data.min(), data.max(), kde_points)
    kde_vals = kde(data_range)
    if kde_vals.max() > 0:
        return hist_counts.max() / kde_vals.max()
    return 1.0


def compute_correlation_vmax(bands_with_data: List[Dict]) -> float:
    all_significant_correlations = []
    for band_data in bands_with_data:
        significant_correlations = band_data['correlations'][band_data['significant_mask']]
        all_significant_correlations.extend(significant_correlations[np.isfinite(significant_correlations)])
    
    if all_significant_correlations:
        return max(abs(np.min(all_significant_correlations)), abs(np.max(all_significant_correlations)))
    
    all_correlations = []
    for band_data in bands_with_data:
        all_correlations.extend(band_data['correlations'][np.isfinite(band_data['correlations'])])
    return max(abs(np.min(all_correlations)), abs(np.max(all_correlations))) if all_correlations else 0.5


###################################################################
# Data Preparation Utilities
###################################################################

def prepare_data_for_plotting(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[pd.Series, pd.Series, int]:
    mask = x_data.notna() & y_data.notna()
    n_eff = int(mask.sum())
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    return x_clean, y_clean, n_eff


def prepare_data_without_validation(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[pd.Series, pd.Series, int]:
    n_eff = len(x_data)
    return x_data, y_data, n_eff


def _prepare_single_subject_data(
    x_array: np.ndarray,
    y_array: np.ndarray,
    pooling_strategy: str,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    x_series = pd.Series(np.asarray(x_array))
    y_series = pd.Series(np.asarray(y_array))
    
    min_length = min(len(x_series), len(y_series))
    x_series = x_series.iloc[:min_length]
    y_series = y_series.iloc[:min_length]
    
    valid_mask = x_series.notna() & y_series.notna()
    x_series = x_series[valid_mask]
    y_series = y_series[valid_mask]
    
    if x_series.empty or y_series.empty:
        return None
    
    x_normalized, y_normalized = apply_pooling_strategy(x_series, y_series, pooling_strategy)
    if x_normalized.empty or y_normalized.empty:
        return None
    
    return x_normalized.reset_index(drop=True), y_normalized.reset_index(drop=True)


def prepare_group_data(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    subj_order: List[str],
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    x_series_list: List[pd.Series] = []
    y_series_list: List[pd.Series] = []
    subject_ids: List[str] = []

    for idx, (x_array, y_array) in enumerate(zip(x_lists, y_lists)):
        prepared = _prepare_single_subject_data(x_array, y_array, pooling_strategy)
        if prepared is None:
            continue
        
        x_normalized, y_normalized = prepared
        subject_id = subj_order[idx] if idx < len(subj_order) else str(idx)
        
        subject_ids.extend([subject_id] * len(x_normalized))
        x_series_list.append(x_normalized)
        y_series_list.append(y_normalized)

    if not x_series_list:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    x_combined = pd.concat(x_series_list, ignore_index=True)
    y_combined = pd.concat(y_series_list, ignore_index=True)
    
    return x_combined, y_combined, subject_ids


###################################################################
# Masked Statistics
###################################################################

def compute_statistics_for_mask(data_values: pd.Series, mask: np.ndarray) -> Tuple[float, float]:
    masked_data = data_values[mask].to_numpy()
    masked_data = masked_data[np.isfinite(masked_data)]
    
    if len(masked_data) == 0:
        return 0.0, 0.0
    
    mean_val = np.mean(masked_data)
    n = len(masked_data)
    sem_val = np.std(masked_data) / np.sqrt(n) if n > 1 else 0.0
    
    return mean_val, sem_val


def compute_coverage_statistics(
    coverage_values: pd.Series,
    nonpain_mask: np.ndarray,
    pain_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    mean_nonpain, sem_nonpain = compute_statistics_for_mask(coverage_values, nonpain_mask)
    mean_pain, sem_pain = compute_statistics_for_mask(coverage_values, pain_mask)
    return mean_nonpain, mean_pain, sem_nonpain, sem_pain


###################################################################
# Band Correlations
###################################################################

def compute_band_spatial_correlation(band1_channels: np.ndarray, band2_channels: np.ndarray) -> float:
    if len(band1_channels) <= 1 or len(band2_channels) <= 1:
        return np.nan
    return np.corrcoef(band1_channels, band2_channels)[0, 1]


###################################################################
# Fisher Z-Transform
###################################################################

def fisher_z_transform_mean(r_values: np.ndarray, config: Optional[Any] = None) -> float:
    constants = _get_statistics_constants(config)
    r_clipped = np.clip(r_values, constants["fisher_z_clip_min"], constants["fisher_z_clip_max"])
    z_scores = np.arctanh(r_clipped)
    z_mean = float(np.mean(z_scores))
    return float(np.tanh(z_mean))


def compute_band_statistics_array(values: np.ndarray, ci_multiplier: Optional[float] = None, config: Optional[Any] = None) -> Tuple[float, float, float, int]:
    if ci_multiplier is None:
        constants = _get_statistics_constants(config)
        ci_multiplier = constants["ci_multiplier_95"]
    values_clean = values[np.isfinite(values)]
    if values_clean.size == 0:
        return np.nan, np.nan, np.nan, 0
    
    mean_val = float(np.mean(values_clean))
    n = len(values_clean)
    se = float(np.std(values_clean, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    delta = ci_multiplier * se if np.isfinite(se) else np.nan
    
    ci_lower = mean_val - delta if np.isfinite(delta) else np.nan
    ci_upper = mean_val + delta if np.isfinite(delta) else np.nan
    
    return mean_val, ci_lower, ci_upper, n


###################################################################
# Data Extraction Utilities
###################################################################

def extract_roi_statistics(
    df: Optional[pd.DataFrame],
    roi_name: str,
    band_name: str,
) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "roi" not in df.columns or "band" not in df.columns:
        return None
    
    band_match = df["band"].astype(str).str.lower() == band_name.lower()
    roi_match = df["roi"].astype(str).str.lower() == roi_name.lower()
    mask = band_match & roi_match
    
    if mask.any():
        return df.loc[mask].iloc[0]
    return None


def extract_overall_statistics(
    df: Optional[pd.DataFrame],
    band_name: str,
    overall_keys: List[str] = None,
) -> Optional[pd.Series]:
    if overall_keys is None:
        overall_keys = ["overall", "all", "global"]
    
    for key in overall_keys:
        row = extract_roi_statistics(df, key, band_name)
        if row is not None:
            return row
    return None


def update_stats_from_dataframe(
    stats_df: Optional[pd.Series],
    r_val: float,
    p_val: float,
    n_eff: int,
    ci_val: Tuple[float, float],
) -> Tuple[float, float, int, Tuple[float, float]]:
    if stats_df is None:
        return r_val, p_val, n_eff, ci_val
    
    n_eff = int(stats_df.get("n", n_eff))
    r_val = _safe_float(stats_df.get("r", r_val))
    p_val = _safe_float(stats_df.get("p", p_val))
    ci_val = (
        _safe_float(stats_df.get("r_ci_low", ci_val[0])),
        _safe_float(stats_df.get("r_ci_high", ci_val[1])),
    )
    return r_val, p_val, n_eff, ci_val


###################################################################
# Internal helpers
###################################################################

def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return np.nan


def _get_ttest_pvalue(ttest_result: Any) -> float:
    if ttest_result is None:
        return np.nan
    if not hasattr(ttest_result, "pvalue"):
        return np.nan
    return _safe_float(ttest_result.pvalue)


def _ensure_aligned_lengths_for_partial(x, y, Z, *, context: str = "", strict: bool = True, logger=None) -> None:
    if Z is None:
        return
    non_null = [obj for obj in (x, y, Z) if obj is not None]
    if len(non_null) < 2:
        return
    lengths = {len(obj) for obj in non_null}
    if len(lengths) > 1:
        msg = f"{context}: Length mismatch detected: {[len(obj) for obj in non_null]}"
        if strict:
            raise ValueError(msg)
        if logger:
            logger.warning(msg)


###################################################################
# Statistical Formatting Utilities
###################################################################

def format_p_value(p):
    try:
        p_arr = np.asarray(p, dtype=float)
    except Exception:
        return "p=nan"
    if p_arr.size == 0:
        return "p=nan"
    p_val = float(np.nanmin(p_arr)) if p_arr.size > 1 else float(p_arr)
    if not np.isfinite(p_val):
        return "p=nan"
    return "p<.001" if p_val < 1e-3 else f"p={p_val:.3f}"


def format_cluster_ann(p, k=None, mass=None, config=None):
    parts = [format_p_value(p)]
    if config is None:
        try:
            from .config_loader import load_settings
            config = load_settings()
        except Exception:
            config = None
    report_size = config.get("statistics.cluster_report_size", False) if config else False
    report_mass = config.get("statistics.cluster_report_mass", False) if config else False
    if report_size and isinstance(k, (int, np.integer)) and k and k > 0:
        parts.append(f"k={int(k)}")
    if report_mass and mass is not None and np.isfinite(mass):
        parts.append(f"mass={float(mass):.1f}")
    return "; ".join(parts)


def format_correlation_text(r_val: float, p_val: Optional[float] = None) -> str:
    if not np.isfinite(r_val):
        return "r=nan"
    r_str = f"r={r_val:.3f}"
    if p_val is not None and np.isfinite(p_val):
        p_str = "p<.001" if p_val < 1e-3 else f"p={p_val:.3f}"
        return f"{r_str}\n{p_str}"
    return r_str


def format_fdr_ann(q_min: Optional[float], k_rej: Optional[int], alpha: float = 0.05) -> str:
    parts = []
    if q_min is not None and np.isfinite(q_min):
        parts.append(f"FDR q={q_min:.3f}" if q_min >= 1e-3 else "FDR q<.001")
    if k_rej is not None and isinstance(k_rej, (int, np.integer)) and int(k_rej) > 0:
        parts.append(f"k={int(k_rej)}")
    return "; ".join(parts) if parts else ""


def format_correlation_stats_text(
    r_val: float,
    p_val: float,
    n_val: int,
    ci_val: Optional[Tuple[float, float]],
    stats_tag: Optional[str],
) -> str:
    label = "Spearman \u03c1"
    ci_str = ""
    if ci_val is not None and np.all(np.isfinite(ci_val)):
        ci_str = f"\nCI [{ci_val[0]:.2f}, {ci_val[1]:.2f}]"
    tag_str = f" {stats_tag}" if stats_tag else ""
    return f"{label}{tag_str} = {r_val:.3f}\np = {p_val:.3f}\nn = {n_val}{ci_str}"


def apply_fdr_correction_and_save(
    results_df: pd.DataFrame,
    output_path: Path,
    config,
    logger: logging.Logger,
    use_permutation_p: bool = True,
) -> None:
    if results_df.empty:
        return
    
    if "p" not in results_df.columns:
        logger.warning(
            f"No p-values column found in results; skipping FDR correction for {output_path}"
        )
        return
    
    alpha = get_fdr_alpha_from_config(config)
    p_vector = select_p_values_for_fdr(results_df, use_permutation_p)
    
    logger.info(
        f"Applying per-analysis-type FDR correction (alpha={alpha}) to {len(results_df)} tests. "
        f"Note: This controls FDR within this analysis type only. "
        f"For global FDR across all analysis types, use apply_global_fdr() separately."
    )
    
    rejections, critical_p = fdr_bh_reject(p_vector, alpha=alpha)
    results_df["fdr_reject"] = rejections
    results_df["fdr_crit_p"] = critical_p
    
    results_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(results_df)} results to {output_path}")


###################################################################
# Trial Counting Utilities
###################################################################

def count_trials_by_condition(
    epochs: "mne.Epochs",
    condition_column: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """Count trials by binary condition (e.g., pain vs non-pain).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object with metadata
    condition_column : Optional[str]
        Column name in metadata containing binary condition (0/1)
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    Tuple[int, int]
        (n_condition_1, n_condition_0) counts
    """
    if condition_column is None or epochs.metadata is None or condition_column not in epochs.metadata.columns:
        return 0, 0
    
    from eeg_pipeline.utils.stats_utils import validate_pain_binary_values
    condition_values, _ = validate_pain_binary_values(
        epochs.metadata[condition_column], condition_column, logger=logger
    )
    n_condition_1 = int((condition_values == 1).sum())
    n_condition_0 = int((condition_values == 0).sum())
    return n_condition_1, n_condition_0


__all__ = [
    # FDR
    "fdr_bh",
    "fdr_bh_reject",
    "fdr_bh_mask",
    "fdr_bh_values",
    # Data validation utilities
    "validate_pain_binary_values",
    "validate_temperature_values",
    "validate_baseline_window_pre_stimulus",
    # Trial counting utilities
    "count_trials_by_condition",
    # EEG cluster utilities
    "get_eeg_adjacency",
    "build_full_mask_from_eeg",
    "cluster_mask_from_clusters",
    "cluster_test_two_sample_arrays",
    "cluster_test_epochs",
    # PyRiemann and epoch utilities
    "check_pyriemann",
    "align_epochs_to_pivot_chs",
    # Correlation utilities
    "bh_adjust",
    "fisher_aggregate",
    "partial_corr_xy_given_Z",
    "partial_residuals_xy_given_Z",
    "compute_partial_corr",
    "compute_partial_residuals",
    "fisher_ci",
    "joint_valid_mask",
    "perm_pval_simple",
    "perm_pval_partial_freedman_lane",
    "bootstrap_corr_ci",
    "compute_group_corr_stats",
    "get_correlation_method",
    "compute_correlation",
    "compute_bootstrap_ci",
    "get_fdr_alpha_from_config",
    "select_p_values_for_fdr",
    "filter_significant_predictors",
    "compute_fisher_transformed_mean",
    "compute_correlation_pvalue",
    "compute_correlation_for_metric_state",
    "compute_duration_p_value",
    "compute_channel_confidence_interval",
    "compute_group_channel_statistics",
    "normalize_series",
    "pool_data_by_strategy",
    # Linear regression residuals
    "compute_linear_residuals",
    # Regression and binned statistics
    "extract_finite_mask",
    "fit_linear_regression",
    "compute_binned_statistics",
    "compute_error_bars_from_ci_dicts",
    "compute_consensus_labels",
    "compute_inter_band_coupling_matrix",
    "compute_group_channel_power_statistics",
    "compute_group_band_statistics",
    "compute_error_bars_from_arrays",
    "compute_band_pair_correlation",
    "compute_subject_band_correlation_matrix",
    "compute_group_band_correlation_matrix",
    "compute_correlation_ci_fisher",
    "compute_inter_band_correlation_statistics",
    # ROI statistics utilities
    "compute_roi_percentage_change",
    "compute_roi_pvalue",
    # Data transformation utilities
    "center_series",
    "zscore_series",
    "apply_pooling_strategy",
    # Correlation statistics
    "compute_correlation_stats",
    "compute_partial_residuals_stats",
    # Band and connectivity correlations
    "compute_band_correlations",
    "compute_connectivity_correlations",
    # Visualization utilities
    "compute_kde_scale",
    "compute_correlation_vmax",
    # Data preparation utilities
    "prepare_data_for_plotting",
    "prepare_data_without_validation",
    "prepare_group_data",
    # Data extraction utilities
    "extract_roi_statistics",
    "extract_overall_statistics",
    "update_stats_from_dataframe",
    # Masked statistics
    "compute_statistics_for_mask",
    "compute_coverage_statistics",
    # Band correlations
    "compute_band_spatial_correlation",
    # Fisher z-transform
    "fisher_z_transform_mean",
    # Band statistics (array version)
    "compute_band_statistics_array",
    # Permutation and partial correlation utilities
    "compute_perm_and_partial_perm",
    "compute_partial_correlation_with_covariates",
    "compute_partial_correlations",
    # Statistical formatting utilities
    "format_p_value",
    "format_correlation_text",
    "format_cluster_ann",
    "format_fdr_ann",
    "format_correlation_stats_text",
    "apply_fdr_correction_and_save",
    "compute_permutation_pvalue_partial",
    "compute_permutation_pvalues",
    "compute_temp_permutation_pvalues",
    "compute_channel_rating_correlations",
    "compute_partial_correlation_for_roi_pair",
    "compute_permutation_pvalues_for_roi_pair",
    "compute_temp_correlations_for_roi",
    "compute_temp_correlation_for_roi_pair",
    "compute_correlation_for_time_freq_bin",
    "compute_correlation_from_vectors",
    # Cluster correction utilities
    "compute_cluster_masses_2d",
    "compute_permutation_max_masses",
    "compute_cluster_pvalues",
    "compute_cluster_correction_2d",
    "compute_cluster_masses_1d",
    "compute_topomap_permutation_masses",
    "compute_cluster_pvalues_1d",
    # Constants
    "DEFAULT_MIN_SAMPLES_ROI",
    "DEFAULT_MIN_SAMPLES_CHANNEL",
    "CLUSTER_STRUCTURE_2D",
    # Summary statistics utilities
    "compute_band_summary_statistics",
    "compute_band_summaries",
    # Aperiodic fitting utilities
    "fit_aperiodic",
    "fit_aperiodic_to_all_epochs",
    "compute_residuals",
    # Internal helpers
    "_safe_float",
    "_get_ttest_pvalue",
    "joint_valid_mask",
    "prepare_aligned_data",
    # Data extraction utilities
    "extract_pain_masks",
    "extract_duration_data",
    # Statistical utilities
    "should_apply_fisher_transform",
    "get_cluster_correction_config",
    "get_pvalue_series",
    "extract_pvalue_from_dataframe",
    "compute_fdr_rejections_for_heatmap",
    "build_correlation_matrices_for_prefix",
]


###################################################################
# Data Extraction Utilities
###################################################################

def extract_pain_masks(pain_vals: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = pain_vals.notna()
    nonpain_mask = valid_mask & (pain_vals == 0)
    pain_mask = valid_mask & (pain_vals == 1)
    return nonpain_mask, pain_mask

def extract_duration_data(durations: pd.Series, mask: np.ndarray) -> np.ndarray:
    data = durations[mask].to_numpy(dtype=float)
    return data[np.isfinite(data)]


###################################################################
# Statistical Utilities
###################################################################

def should_apply_fisher_transform(prefix: str) -> bool:
    measure_name = prefix.split("_", 1)[0].lower()
    fisher_transform_measures = ("aec", "aec_orth", "corr", "pearsonr")
    return measure_name in fisher_transform_measures

def get_cluster_correction_config(
    heatmap_config: Dict[str, Any],
    config,
    alpha: float,
    default_rng_seed: int = 42,
) -> Tuple[float, int, np.random.Generator, int]:
    cluster_cfg = config.get("behavior_analysis.cluster_correction", {})
    cluster_alpha = float(
        heatmap_config.get("cluster_alpha", cluster_cfg.get("alpha", alpha))
    )
    n_cluster_perm = int(
        heatmap_config.get("n_cluster_perm", cluster_cfg.get("n_permutations"))
    )
    if n_cluster_perm is None:
        raise ValueError("n_cluster_perm or behavior_analysis.cluster_correction.n_permutations must be specified in config")
    cluster_rng_seed = int(
        heatmap_config.get(
            "cluster_rng_seed", cluster_cfg.get("rng_seed", default_rng_seed)
        )
    )
    cluster_rng = np.random.default_rng(cluster_rng_seed)
    
    return cluster_alpha, n_cluster_perm, cluster_rng, cluster_rng_seed

def get_pvalue_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Extract permutation and raw p-value series from dataframe.
    
    Returns
    -------
    p_permutation_series : pd.Series
        Series of permutation p-values (p_perm, p_partial_perm, etc.), aligned by row index
    p_raw_series : pd.Series
        Series of raw p-values (p, p_value, p_partial, etc.), aligned by row index
    """
    p_permutation_series = pd.Series(index=df.index, dtype=float)
    p_raw_series = pd.Series(index=df.index, dtype=float)
    
    permutation_priority = ["p_perm", "p_partial_perm", "p_partial_temp_perm"]
    raw_priority = ["p", "p_value", "p_partial", "p_partial_temp"]
    
    for col in permutation_priority:
        if col in df.columns:
            p_vals = pd.to_numeric(df[col], errors="coerce")
            p_permutation_series = p_permutation_series.fillna(p_vals)
    
    for col in raw_priority:
        if col in df.columns:
            p_vals = pd.to_numeric(df[col], errors="coerce")
            p_raw_series = p_raw_series.fillna(p_vals)
    
    return p_permutation_series, p_raw_series

def extract_pvalue_from_dataframe(df: pd.DataFrame, row_idx: int) -> Tuple[float, str]:
    p_cols = [c for c in df.columns if c.startswith("p") and not c.startswith("pow")]
    if not p_cols:
        return np.nan, ""
    
    for col in p_cols:
        p_val = pd.to_numeric(df.iloc[row_idx][col], errors="coerce")
        if pd.notna(p_val):
            return float(p_val), col
    
    return np.nan, ""

def compute_fdr_rejections_for_heatmap(
    p_value_matrix: np.ndarray,
    n_nodes: int,
    config,
) -> Tuple[Dict[Tuple[int, int], bool], float]:
    from .io_utils import fdr_bh_reject
    from .stats_utils import get_fdr_alpha_from_config
    
    upper_triangle_indices = np.triu_indices(n_nodes, k=1)
    p_upper_triangle = p_value_matrix[upper_triangle_indices]
    valid_mask = np.isfinite(p_upper_triangle)
    p_valid = p_upper_triangle[valid_mask]
    
    fdr_alpha = get_fdr_alpha_from_config(config)
    rejections, critical_p = fdr_bh_reject(p_valid, alpha=fdr_alpha)
    
    valid_pairs = [
        (upper_triangle_indices[0][k], upper_triangle_indices[1][k])
        for k in np.where(valid_mask)[0]
    ]
    rejection_map = {pair: bool(rejections[k]) for k, pair in enumerate(valid_pairs)}
    critical_value = (
        _safe_float(np.max(p_valid[rejections])) if np.any(rejections) else np.nan
    )
    
    return rejection_map, critical_value

def build_correlation_matrices_for_prefix(
    prefix: str,
    prefix_columns: List[str],
    connectivity_dataframe: pd.DataFrame,
    target_values: pd.Series,
    node_to_index: Dict[str, int],
    use_spearman: bool,
    min_samples: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    from .stats_utils import compute_correlation
    
    n_nodes = len(node_to_index)
    correlation_matrix = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    p_value_matrix = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    
    for column_name in prefix_columns:
        from .tfr_utils import extract_node_pair_from_column
        node_pair = extract_node_pair_from_column(column_name, prefix)
        if node_pair is None:
            continue
        
        node_a, node_b = node_pair
        if node_a not in node_to_index or node_b not in node_to_index:
            continue
        
        index_a = node_to_index[node_a]
        index_b = node_to_index[node_b]
        
        edge_values = pd.to_numeric(connectivity_dataframe[column_name], errors="coerce")
        valid_mask = edge_values.notna() & target_values.notna()
        
        if valid_mask.sum() < min_samples:
            continue
        
        correlation, p_value = compute_correlation(
            edge_values[valid_mask], target_values[valid_mask], use_spearman
        )
        
        correlation_matrix[index_a, index_b] = correlation
        correlation_matrix[index_b, index_a] = correlation
        p_value_matrix[index_a, index_b] = p_value
        p_value_matrix[index_b, index_a] = p_value
    
    return correlation_matrix, p_value_matrix





