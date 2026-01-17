"""
Power topomap correlations with temperature and cluster correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.stats import (
    compute_correlation,
    get_fdr_alpha,
    get_eeg_adjacency,
    compute_cluster_masses_1d,
    compute_topomap_permutation_masses,
    compute_cluster_pvalues_1d,
    bh_adjust as _bh_adjust,
    _safe_float,
)
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.infra.paths import ensure_dir


def _extract_band_channels(
    power_df: pd.DataFrame, band: str
) -> tuple[List[str], List[str]]:
    """
    Extract column names and channel identifiers for a specific frequency band.

    Returns
    -------
    columns : List[str]
        Column names matching the band.
    channel_names : List[str]
        Channel identifiers extracted from column names.
    """
    columns: List[str] = []
    channel_names: List[str] = []
    band_lowercase = str(band).lower()

    for column_name in power_df.columns:
        name = str(column_name)
        parsed = NamingSchema.parse(name)
        is_valid_power = parsed.get("valid") and parsed.get("group") == "power"
        if not is_valid_power:
            continue

        parsed_band = str(parsed.get("band") or "").lower()
        if parsed_band != band_lowercase:
            continue

        identifier = parsed.get("identifier")
        if identifier is None:
            continue

        columns.append(name)
        channel_names.append(str(identifier))

    return columns, channel_names


def _compute_channel_correlations(
    power_df: pd.DataFrame,
    temperature: pd.Series,
    columns: List[str],
    min_samples: int,
    use_spearman: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute correlations for each channel column.

    Returns
    -------
    correlations : np.ndarray
        Correlation coefficients per channel.
    pvalues : np.ndarray
        P-values per channel.
    n_valid_samples : np.ndarray
        Number of valid samples per channel.
    channel_data : np.ndarray
        Channel data matrix (n_channels, n_samples).
    """
    n_channels = len(columns)
    correlations = np.full(n_channels, np.nan)
    pvalues = np.full(n_channels, np.nan)
    n_valid_samples = np.zeros(n_channels, dtype=int)
    channel_data = np.full((n_channels, len(power_df)), np.nan)

    correlation_method = "spearman" if use_spearman else "pearson"

    for channel_idx, column_name in enumerate(columns):
        series = pd.to_numeric(power_df[column_name], errors="coerce")
        channel_data[channel_idx] = series.values
        valid_mask = series.notna() & temperature.notna()
        n_valid_samples[channel_idx] = int(valid_mask.sum())

        if n_valid_samples[channel_idx] >= min_samples:
            correlations[channel_idx], pvalues[channel_idx] = (
                compute_correlation(
                    series[valid_mask],
                    temperature[valid_mask],
                    correlation_method,
                )
            )

    return correlations, pvalues, n_valid_samples, channel_data


def _apply_cluster_correction(
    correlations: np.ndarray,
    pvalues: np.ndarray,
    channel_data: np.ndarray,
    temperature: pd.Series,
    channel_names: List[str],
    cluster_params: Dict[str, Any],
    eeg_info: Dict[str, Any],
    min_samples: int,
    use_spearman: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply cluster-based multiple comparisons correction.

    Returns
    -------
    cluster_labels : np.ndarray
        Cluster assignment per channel.
    cluster_pvalues : np.ndarray
        Cluster-corrected p-values per channel.
    cluster_significant : np.ndarray
        Boolean array indicating significant clusters.
    """
    n_channels = len(channel_names)
    cluster_labels = np.zeros(n_channels, dtype=int)
    cluster_pvalues = np.full(n_channels, np.nan)
    cluster_significant = np.zeros(n_channels, dtype=bool)

    eeg_channel_names = [
        eeg_info["info"]["ch_names"][pick_idx] for pick_idx in eeg_info["picks"]
    ]
    eeg_name_to_index = {
        name: index for index, name in enumerate(eeg_channel_names)
    }
    channel_map = {
        channel_idx: eeg_name_to_index[name]
        for channel_idx, name in enumerate(channel_names)
        if name in eeg_name_to_index
    }

    min_channels_required = cluster_params["min_channels_for_adjacency"]
    if len(channel_map) < min_channels_required:
        return cluster_labels, cluster_pvalues, cluster_significant

    cluster_labels, cluster_masses = compute_cluster_masses_1d(
        correlations,
        pvalues,
        cluster_params["cluster_alpha"],
        channel_map,
        eeg_info["picks"],
        eeg_info["adjacency"],
    )

    if not cluster_masses:
        return cluster_labels, cluster_pvalues, cluster_significant

    permutation_masses = compute_topomap_permutation_masses(
        channel_data,
        temperature,
        n_channels,
        cluster_params["n_cluster_perm"],
        cluster_params["cluster_alpha"],
        min_samples,
        use_spearman,
        channel_map,
        eeg_info["picks"],
        eeg_info["adjacency"],
        cluster_params["cluster_rng"],
    )

    cluster_pvalues, cluster_significant, _ = compute_cluster_pvalues_1d(
        cluster_labels, cluster_masses, permutation_masses, cluster_params["alpha"]
    )

    return cluster_labels, cluster_pvalues, cluster_significant


def _process_band(
    band: str,
    power_df: pd.DataFrame,
    temperature: pd.Series,
    min_samples: int,
    cluster_params: Dict[str, Any],
    eeg_info: Dict[str, Any],
    use_spearman: bool,
) -> List[Dict[str, Any]]:
    """
    Compute per-channel correlations for a single frequency band.
    """
    columns, channel_names = _extract_band_channels(power_df, band)
    if not columns:
        return []

    correlations, pvalues, n_valid_samples, channel_data = (
        _compute_channel_correlations(
            power_df, temperature, columns, min_samples, use_spearman
        )
    )

    has_valid_data = np.isfinite(pvalues) & (n_valid_samples >= min_samples)
    if not has_valid_data.any():
        return []

    corrected_pvalues = np.full_like(pvalues, np.nan)
    corrected_pvalues[has_valid_data] = _bh_adjust(pvalues[has_valid_data])
    alpha_threshold = cluster_params["alpha"]
    is_significant = (corrected_pvalues < alpha_threshold) & has_valid_data

    cluster_labels, cluster_pvalues, cluster_significant = (
        _apply_cluster_correction(
            correlations,
            pvalues,
            channel_data,
            temperature,
            channel_names,
            cluster_params,
            eeg_info,
            min_samples,
            use_spearman,
        )
    )

    n_channels = len(channel_names)
    return [
        {
            "band": band,
            "channel": channel_names[channel_idx],
            "correlation": _safe_float(correlations[channel_idx]),
            "p_value": _safe_float(pvalues[channel_idx]),
            "p_corrected": _safe_float(corrected_pvalues[channel_idx]),
            "significant": bool(is_significant[channel_idx]),
            "cluster_id": int(cluster_labels[channel_idx]),
            "cluster_p": _safe_float(cluster_pvalues[channel_idx]),
            "cluster_significant": bool(cluster_significant[channel_idx]),
            "n_valid": int(n_valid_samples[channel_idx]),
        }
        for channel_idx in range(n_channels)
    ]


def _build_cluster_parameters(
    config: Any, alpha: float, rng: Optional[np.random.Generator]
) -> Dict[str, Any]:
    """
    Build cluster correction parameters from configuration.

    Parameters
    ----------
    config : Any
        Configuration object/dict.
    alpha : float
        FDR alpha threshold.
    rng : np.random.Generator, optional
        Random generator for cluster permutations.

    Returns
    -------
    Dict[str, Any]
        Cluster correction parameters.
    """
    cluster_config = (
        config.get("behavior_analysis.cluster_correction", {})
        if config is not None
        else {}
    )

    default_random_state = (
        config.get("project.random_state", 42) if config is not None else 42
    )
    random_generator = rng or np.random.default_rng(default_random_state)

    default_min_channels = (
        config.get("behavior_analysis.statistics.min_channels_for_adjacency", 2)
        if config is not None
        else 2
    )

    return {
        "alpha": float(alpha),
        "cluster_alpha": float(cluster_config.get("alpha", alpha)),
        "n_cluster_perm": int(cluster_config.get("n_permutations", 100)),
        "cluster_rng": random_generator,
        "min_channels_for_adjacency": default_min_channels,
    }


def _validate_inputs(
    power_df: Optional[pd.DataFrame],
    temperature: Optional[pd.Series],
    epochs_info: Any,
    logger,
) -> bool:
    """
    Validate input data for topomap correlation analysis.

    Returns
    -------
    bool
        True if inputs are valid, False otherwise.
    """
    has_power_data = power_df is not None and not power_df.empty
    has_temperature_data = (
        temperature is not None and not temperature.isna().all()
    )
    if not (has_power_data and has_temperature_data):
        logger.warning(
            "Missing power or temperature data; skipping topomap correlations"
        )
        return False

    if epochs_info is None:
        logger.warning("Missing EEG info; skipping topomap correlations")
        return False

    return True


def run_power_topomap_correlations(
    subject: str,
    task: Optional[str],
    power_df: Optional[pd.DataFrame],
    temperature: Optional[pd.Series],
    epochs_info: Any,
    stats_dir: Path,
    config: Any,
    logger,
    use_spearman: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Optional[pd.DataFrame]:
    """
    Correlate power topomaps with temperature using pre-loaded data.

    Parameters
    ----------
    subject : str
        Subject label.
    task : str
        Task label.
    power_df : pd.DataFrame
        Power features with columns pow_<band>_<channel>.
    temperature : pd.Series
        Temperature/target series aligned to power_df rows.
    epochs_info : mne.Info
        EEG info object used to derive adjacency.
    stats_dir : Path
        Directory for saving outputs.
    config : Any
        Configuration object/dict.
    logger : logging.Logger
        Logger for status messages.
    use_spearman : bool
        Whether to use Spearman correlations (default True).
    rng : np.random.Generator, optional
        Random generator for cluster permutations.

    Returns
    -------
    pd.DataFrame, optional
        Results dataframe with correlations and statistics, or None if
        analysis cannot be performed.
    """
    if not _validate_inputs(power_df, temperature, epochs_info, logger):
        return None

    ensure_dir(stats_dir)

    adjacency, picks, _ = get_eeg_adjacency(epochs_info)
    if adjacency is None:
        logger.warning("No EEG adjacency; skipping topomap correlations")
        return None

    bands = get_frequency_band_names(config)
    alpha = get_fdr_alpha(config)
    cluster_params = _build_cluster_parameters(config, alpha, rng)

    min_samples = (
        int(config.get("behavior_analysis.statistics.min_samples_roi", 5))
        if config is not None
        else 5
    )

    eeg_info = {"info": epochs_info, "adjacency": adjacency, "picks": picks}
    records: List[Dict[str, Any]] = []
    for band in bands:
        band_results = _process_band(
            band,
            power_df,
            temperature,
            min_samples,
            cluster_params,
            eeg_info,
            use_spearman,
        )
        records.extend(band_results)

    if not records:
        logger.warning(f"No topomap correlations for sub-{subject}")
        return None

    results_df = pd.DataFrame(records)
    logger.info(f"Computed {len(results_df)} topomap correlations for sub-{subject}")
    return results_df


__all__ = [
    "run_power_topomap_correlations",
]

