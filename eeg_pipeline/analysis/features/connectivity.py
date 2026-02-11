"""
Connectivity Feature Extraction
================================

Consolidated module for all connectivity features:
- Undirected connectivity: wPLI, PLI, imCoh, PLV, AEC, AEC-orth
- Directed connectivity: PSI, DTF, PDC
- Graph metrics: clustering, efficiency, participation, small-world

This module consolidates connectivity.py and directed_connectivity.py
to eliminate duplicated spatial aggregation and graph metric logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

try:
    from mne_connectivity import envelope_correlation, spectral_connectivity_time
except ImportError:
    envelope_correlation = None
    spectral_connectivity_time = None

try:
    from mne_connectivity import spectral_connectivity_epochs
except ImportError:
    spectral_connectivity_epochs = None

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    KMeans = None
    StandardScaler = None

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import get_frequency_bands, get_nested_value
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.analysis.graph_metrics import (
    symmetrize_adjacency as _symmetrize_and_clip,
    compute_global_efficiency_weighted as _global_efficiency_weighted,
    compute_small_world_sigma,
    threshold_adjacency as _threshold_adjacency,
)

from eeg_pipeline.analysis.features.preparation import (
    _get_spatial_transform_type,
    precompute_data,
)


###################################################################
# Connectivity Configuration
###################################################################


@dataclass
class ConnectivityConfig:
    """Normalized connectivity configuration with validated types.

    Single-pass config validation eliminates redundant conn_cfg.get() calls.
    """
    measures: List[str]
    granularity: str
    condition_column: Optional[str]
    condition_values: List[str]
    phase_estimator: str
    output_level: str
    mode: str
    aec_mode: str
    aec_absolute: bool
    n_freqs_per_band: int
    n_cycles: Optional[float]
    decim: int
    min_segment_samples: int
    min_segment_sec: float
    min_cycles_per_band: float
    min_epochs_per_group: int
    enable_graph_metrics: bool
    enable_aec: bool
    enable_aec_raw: bool
    enable_aec_z: bool
    graph_top_prop: float
    small_world_n_rand: int
    force_within_epoch_for_ml: bool
    warn_if_no_spatial_transform: bool
    sliding_window_len: float
    sliding_window_step: float
    dynamic_enabled: bool
    dynamic_measures: List[str]
    dynamic_autocorr_lag: int
    dynamic_min_windows: int
    dynamic_include_roi_pairs: bool
    dynamic_state_enabled: bool
    dynamic_state_n_states: int
    dynamic_state_min_windows: int
    dynamic_state_random_state: Optional[int]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ConnectivityConfig":
        """Create normalized config from raw dict with validation."""
        conn_cfg = {}
        if isinstance(cfg, dict):
            conn_cfg = get_nested_value(cfg, "feature_engineering.connectivity", {}) or {}
        if not isinstance(conn_cfg, dict) and hasattr(cfg, "get"):
            conn_cfg = cfg.get("feature_engineering.connectivity", {}) or {}
        if not isinstance(conn_cfg, dict):
            conn_cfg = {}

        measures_cfg = conn_cfg.get("measures", ["wpli", "aec"])
        if isinstance(measures_cfg, str):
            measures_cfg = [measures_cfg]
        if not isinstance(measures_cfg, (list, tuple)):
            raise TypeError(
                "feature_engineering.connectivity.measures must be a list/tuple of strings "
                f"(got {type(measures_cfg).__name__})."
            )
        measures = [str(m).strip().lower() for m in measures_cfg]

        granularity = str(conn_cfg.get("granularity", "trial")).strip().lower()
        if granularity not in {"trial", "condition", "subject"}:
            raise ValueError(
                "feature_engineering.connectivity.granularity must be one of "
                "{'trial','condition','subject'} "
                f"(got '{granularity}')."
            )

        condition_column = conn_cfg.get("condition_column", None)
        if condition_column is not None:
            condition_column = str(condition_column).strip()
        if condition_column == "":
            condition_column = None

        condition_values_cfg = conn_cfg.get("condition_values", [])
        if condition_values_cfg is None:
            condition_values_cfg = []
        if isinstance(condition_values_cfg, str):
            condition_values_cfg = [condition_values_cfg]
        if not isinstance(condition_values_cfg, (list, tuple)):
            raise TypeError(
                "feature_engineering.connectivity.condition_values must be a list/tuple of strings "
                f"(got {type(condition_values_cfg).__name__})."
            )
        condition_values = [str(v).strip() for v in condition_values_cfg if str(v).strip() != ""]

        phase_estimator = str(conn_cfg.get("phase_estimator", "within_epoch")).strip().lower()
        if phase_estimator not in {"within_epoch", "across_epochs"}:
            raise ValueError(
                "feature_engineering.connectivity.phase_estimator must be one of "
                "{'within_epoch','across_epochs'} "
                f"(got '{phase_estimator}')."
            )

        output_level = str(conn_cfg.get("output_level", "full")).strip().lower()
        if output_level not in {"full", "global_only"}:
            raise ValueError(
                "feature_engineering.connectivity.output_level must be one of "
                "{'full','global_only'} "
                f"(got '{output_level}')."
            )

        mode = str(conn_cfg.get("mode", "cwt_morlet")).strip().lower()

        aec_mode = str(conn_cfg.get("aec_mode", "orth")).strip().lower()
        aec_absolute = bool(conn_cfg.get("aec_absolute", True))

        n_freqs_per_band = int(conn_cfg.get("n_freqs_per_band", 8))

        n_cycles = conn_cfg.get("n_cycles", None)
        n_cycles = float(n_cycles) if n_cycles is not None else None

        decim = int(conn_cfg.get("decim", 1))
        min_segment_samples = int(conn_cfg.get("min_segment_samples", 50))
        min_segment_sec = float(conn_cfg.get("min_segment_sec", 1.0))
        min_cycles_per_band = float(conn_cfg.get("min_cycles_per_band", 3.0))
        min_epochs_per_group = int(conn_cfg.get("min_epochs_per_group", 5))

        enable_graph_metrics = bool(conn_cfg.get("enable_graph_metrics", False))
        enable_aec = bool(conn_cfg.get("enable_aec", True))

        aec_output_modes = conn_cfg.get("aec_output", ["r"])
        if isinstance(aec_output_modes, str):
            aec_output_modes = [aec_output_modes]
        aec_output_modes = [str(m).strip().lower() for m in aec_output_modes]
        enable_aec_raw = "r" in aec_output_modes
        enable_aec_z = "z" in aec_output_modes

        graph_top_prop = conn_cfg.get("graph_top_prop", 0.1)
        graph_top_prop = float(graph_top_prop)
        if not np.isfinite(graph_top_prop) or graph_top_prop <= 0 or graph_top_prop > 1:
            raise ValueError(
                "feature_engineering.connectivity.graph_top_prop must be finite and in (0, 1] "
                f"(got {graph_top_prop})."
            )

        small_world_n_rand = conn_cfg.get("small_world_n_rand", 100)
        small_world_n_rand = int(small_world_n_rand)

        force_within_epoch_for_ml = bool(conn_cfg.get("force_within_epoch_for_ml", True))
        warn_if_no_spatial_transform = bool(conn_cfg.get("warn_if_no_spatial_transform", True))
        sliding_window_len = float(conn_cfg.get("sliding_window_len", 1.0))
        sliding_window_step = float(conn_cfg.get("sliding_window_step", 0.5))
        if not np.isfinite(sliding_window_len) or sliding_window_len <= 0:
            sliding_window_len = 1.0
        if not np.isfinite(sliding_window_step) or sliding_window_step <= 0:
            sliding_window_step = 0.5
        dynamic_enabled = bool(conn_cfg.get("dynamic_enabled", False))
        dynamic_measures_cfg = conn_cfg.get("dynamic_measures", ["wpli", "aec"])
        if isinstance(dynamic_measures_cfg, str):
            dynamic_measures_cfg = [dynamic_measures_cfg]
        if not isinstance(dynamic_measures_cfg, (list, tuple)):
            dynamic_measures_cfg = ["wpli", "aec"]
        dynamic_measures = [
            str(m).strip().lower()
            for m in dynamic_measures_cfg
            if str(m).strip().lower() in {"wpli", "aec"}
        ]
        if not dynamic_measures:
            dynamic_measures = ["wpli", "aec"]
        dynamic_autocorr_lag = int(conn_cfg.get("dynamic_autocorr_lag", 1))
        dynamic_autocorr_lag = max(1, dynamic_autocorr_lag)
        dynamic_min_windows = int(conn_cfg.get("dynamic_min_windows", 3))
        dynamic_min_windows = max(2, dynamic_min_windows)
        dynamic_include_roi_pairs = bool(conn_cfg.get("dynamic_include_roi_pairs", True))
        dynamic_state_enabled = bool(conn_cfg.get("dynamic_state_enabled", True))
        dynamic_state_n_states = int(conn_cfg.get("dynamic_state_n_states", 3))
        dynamic_state_n_states = max(2, dynamic_state_n_states)
        dynamic_state_min_windows = int(conn_cfg.get("dynamic_state_min_windows", 8))
        dynamic_state_min_windows = max(3, dynamic_state_min_windows)
        dynamic_state_random_state_raw = conn_cfg.get("dynamic_state_random_state", None)
        dynamic_state_random_state = (
            int(dynamic_state_random_state_raw)
            if dynamic_state_random_state_raw is not None
            else None
        )

        return cls(
            measures=measures,
            granularity=granularity,
            condition_column=condition_column,
            condition_values=condition_values,
            phase_estimator=phase_estimator,
            output_level=output_level,
            mode=mode,
            aec_mode=aec_mode,
            aec_absolute=aec_absolute,
            n_freqs_per_band=n_freqs_per_band,
            n_cycles=n_cycles,
            decim=decim,
            min_segment_samples=min_segment_samples,
            min_segment_sec=min_segment_sec,
            min_cycles_per_band=min_cycles_per_band,
            min_epochs_per_group=min_epochs_per_group,
            enable_graph_metrics=enable_graph_metrics,
            enable_aec=enable_aec,
            enable_aec_raw=enable_aec_raw,
            enable_aec_z=enable_aec_z,
            graph_top_prop=graph_top_prop,
            small_world_n_rand=small_world_n_rand,
            force_within_epoch_for_ml=force_within_epoch_for_ml,
            warn_if_no_spatial_transform=warn_if_no_spatial_transform,
            sliding_window_len=sliding_window_len,
            sliding_window_step=sliding_window_step,
            dynamic_enabled=dynamic_enabled,
            dynamic_measures=dynamic_measures,
            dynamic_autocorr_lag=dynamic_autocorr_lag,
            dynamic_min_windows=dynamic_min_windows,
            dynamic_include_roi_pairs=dynamic_include_roi_pairs,
            dynamic_state_enabled=dynamic_state_enabled,
            dynamic_state_n_states=dynamic_state_n_states,
            dynamic_state_min_windows=dynamic_state_min_windows,
            dynamic_state_random_state=dynamic_state_random_state,
        )


def _resolve_connectivity_condition_column(
    *,
    events: pd.DataFrame,
    config: Any,
    condition_column_cfg: Optional[str],
) -> str:
    """Resolve the events column used for condition grouping.

    Fail-fast:
    - If condition_column_cfg is provided, it must exist in events.
    - Otherwise, we attempt the legacy auto-detection.
    """
    if condition_column_cfg is not None:
        if condition_column_cfg not in events.columns:
            raise ValueError(
                "Connectivity granularity='condition' requested but "
                f"feature_engineering.connectivity.condition_column='{condition_column_cfg}' "
                "was not found in aligned_events."
            )
        return condition_column_cfg

    for candidate in ("condition", "trial_type"):
        if candidate in events.columns:
            return candidate

    candidates = []
    if hasattr(config, "get"):
        candidates = config.get("event_columns.pain_binary", []) or []
    elif isinstance(config, dict):
        candidates = get_nested_value(config, "event_columns.pain_binary", []) or []
    if isinstance(candidates, (list, tuple)):
        for c in candidates:
            c = str(c).strip()
            if c and c in events.columns:
                return c

    raise ValueError(
        "Connectivity granularity='condition' requested but no condition column found in aligned_events. "
        "Set feature_engineering.connectivity.condition_column explicitly (recommended), or ensure one of "
        "{'condition','trial_type'} exists, or list a valid column in event_columns.pain_binary."
    )


def _resolve_connectivity_condition_selection(
    *,
    labels: pd.Series,
    selected_values: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Resolve which trials/labels are included in condition-level connectivity.

    Semantics:
    - If selected_values is empty: include all labels.
    - If selected_values is non-empty: include only trials where label is in selected_values.

    Returns:
        (included_mask, included_labels)
    """
    labels_str = labels.astype(str)
    if not selected_values:
        included_mask = np.ones(int(labels_str.shape[0]), dtype=bool)
        included_labels = sorted(labels_str.unique())
        return included_mask, included_labels

    selected_set = {str(v) for v in selected_values if str(v).strip() != ""}
    if not selected_set:
        included_mask = np.ones(int(labels_str.shape[0]), dtype=bool)
        included_labels = sorted(labels_str.unique())
        return included_mask, included_labels

    included_mask = labels_str.isin(selected_set).to_numpy()
    if not np.any(included_mask):
        raise ValueError(
            "Connectivity granularity='condition' requested with "
            "feature_engineering.connectivity.condition_values set, but none of the selected "
            "values were found in aligned_events."
        )
    included_labels = sorted(labels_str[included_mask].unique())
    return included_mask, included_labels


###################################################################
# UNDIRECTED CONNECTIVITY (wPLI, PLI, imCoh, AEC)
###################################################################


def _warn_if_phase_connectivity_without_spatial_transform(
    config: Any,
    measures: List[str],
    logger: Any,
) -> None:
    """
    Warn if phase-based connectivity measures are used without CSD/Laplacian transform.
    
    Phase-based measures (wPLI, PLI, ImCoh, PLV) are sensitive to volume conduction.
    Using CSD or Laplacian transform reduces spurious connectivity from field spread.
    """
    phase_measures = {"wpli", "imcoh", "plv", "pli"}
    uses_phase = bool(set(m.lower() for m in measures) & phase_measures)
    
    if not uses_phase:
        return
    
    conn_cfg = config.get("feature_engineering.connectivity", {}) if hasattr(config, "get") else {}
    if not isinstance(conn_cfg, dict):
        conn_cfg = {}
    warn_enabled = bool(conn_cfg.get("warn_if_no_spatial_transform", True))
    if not warn_enabled:
        return
    
    from eeg_pipeline.analysis.features.preparation import _get_spatial_transform_type

    spatial_transform = _get_spatial_transform_type(config, feature_family="connectivity")
    if spatial_transform in {"csd", "laplacian"}:
        return
    
    if logger is not None:
        logger.warning(
            "Phase-based connectivity measures (%s) are used without CSD/Laplacian transform. "
            "This can lead to spurious connectivity from volume conduction. "
            "Consider setting feature_engineering.spatial_transform='csd' for more valid results. "
            "Set feature_engineering.connectivity.warn_if_no_spatial_transform=false to suppress this warning.",
            ", ".join(sorted(set(m.lower() for m in measures) & phase_measures)),
        )


def _is_connectivity_precomputed_compatible(
    precomputed: Any,
    *,
    expected_transform: str,
    bands: List[str],
) -> Tuple[bool, str]:
    """Validate that a precomputed object is safe to reuse for connectivity."""
    if precomputed is None:
        return False, "missing precomputed object"

    family = str(getattr(precomputed, "feature_family", "") or "").strip().lower()
    valid_families = {
        "",
        "connectivity",
        "directedconnectivity",
        "directed_connectivity",
        "dconn",
    }
    if family not in valid_families:
        return False, f"feature_family='{family}'"

    current_transform = str(getattr(precomputed, "spatial_transform", "none") or "none").strip().lower()
    expected = str(expected_transform or "none").strip().lower()
    if expected not in {"none", "csd", "laplacian"}:
        expected = "none"
    if current_transform != expected:
        return False, f"spatial_transform='{current_transform}' (expected '{expected}')"

    band_data = getattr(precomputed, "band_data", {}) or {}
    missing = [b for b in bands if b not in band_data]
    if missing:
        return False, f"missing required bands={missing}"

    return True, ""


def _validate_segment_duration_for_connectivity(
    segment_duration_sec: float,
    fmin: float,
    min_cycles: float,
    band: str,
    logger: Any,
) -> bool:
    """
    Validate that segment duration is sufficient for reliable connectivity estimation.
    
    For phase-based connectivity, we need enough oscillatory cycles to estimate
    phase relationships reliably. Rule of thumb: min_cycles / fmin seconds.
    """
    min_duration_sec = min_cycles / fmin if fmin > 0 else np.inf
    
    if segment_duration_sec < min_duration_sec:
        if logger is not None:
            logger.warning(
                "Connectivity: segment duration (%.2fs) is shorter than recommended "
                "for band '%s' (need %.2fs for %d cycles at %.1f Hz). "
                "Results may be unreliable. Consider longer segments or setting "
                "feature_engineering.connectivity.min_cycles_per_band lower.",
                segment_duration_sec,
                band,
                min_duration_sec,
                int(min_cycles),
                fmin,
            )
        return False
    return True


def _resolve_phase_measures(conn_cfg: Dict[str, Any]) -> List[str]:
    """Extract phase-based connectivity measures from config dict."""
    supported_measures = {"wpli", "imcoh", "plv", "pli"}
    measures_cfg = conn_cfg.get("measures", [])
    if isinstance(measures_cfg, (list, tuple)) and measures_cfg:
        measures = {str(m).strip().lower() for m in measures_cfg}
        measures = measures & supported_measures
        return [m for m in ("wpli", "imcoh", "plv", "pli") if m in measures]
    return []


def _apply_across_epochs_phase_estimates_inplace(
    df: pd.DataFrame,
    *,
    precomputed: Any,
    segments: List[str],
    bands: List[str],
    epoch_groups: Dict[str, np.ndarray],
    config: Any,
    logger: Any,
) -> None:
    """
    Replace phase-based connectivity columns with across-epochs estimates (broadcast to row groups).
    This is used to avoid the scientifically-invalid "compute per-epoch then average" shortcut for
    wPLI/PLV/PLI/imCoh in group-level (subject/condition) summaries.
    """
    if df is None or df.empty:
        return
    if spectral_connectivity_time is None:
        raise ImportError(
            "Connectivity across-epochs phase estimates require 'mne-connectivity'. "
            "Install it with: pip install mne-connectivity"
        )

    conn_cfg = config.get("feature_engineering.connectivity", {}) if hasattr(config, "get") else {}
    phase_measures = _resolve_phase_measures(conn_cfg)
    if not phase_measures:
        return

    output_level = str(conn_cfg.get("output_level", "full")).strip().lower()
    if output_level not in {"full", "global_only"}:
        output_level = "full"
    enable_graph_metrics = bool(conn_cfg.get("enable_graph_metrics", False))

    n_freqs_per_band = int(conn_cfg.get("n_freqs_per_band", 8))
    conn_mode = str(conn_cfg.get("mode", "cwt_morlet"))
    n_cycles = conn_cfg.get("n_cycles", None)
    n_cycles = float(n_cycles) if n_cycles is not None else None
    decim = int(conn_cfg.get("decim", 1))

    min_cycles_per_band = float(conn_cfg.get("min_cycles_per_band", 3.0))
    min_segment_sec = float(conn_cfg.get("min_segment_sec", 0.0))

    sfreq = float(getattr(precomputed, "sfreq", None))
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Connectivity across-epochs estimates require a valid precomputed.sfreq.")

    ch_names = list(getattr(precomputed, "ch_names", []))
    n_channels = len(ch_names)
    if n_channels < 2:
        return
    pair_i, pair_j = np.triu_indices(n_channels, k=1)
    pair_names = [f"{ch_names[i]}-{ch_names[j]}" for i, j in zip(pair_i, pair_j)]
    indices = (pair_i.astype(int), pair_j.astype(int))

    freq_bands = getattr(precomputed, "frequency_bands", None) or get_frequency_bands(config)
    masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)

    def _run(method: str, seg_data: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float, use_n_cycles: Any):
        return spectral_connectivity_time(
            seg_data,
            freqs=freqs,
            method=method,
            indices=indices,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            average=True,
            faverage=True,
            mode=conn_mode,
            n_cycles=use_n_cycles,
            decim=decim,
            n_jobs=1,
            verbose=False,
        )

    for _label, epoch_idx in epoch_groups.items():
        epoch_idx = np.asarray(epoch_idx, dtype=int)
        if epoch_idx.size == 0:
            continue

        for seg_name in segments:
            if seg_name == "full":
                seg_mask = np.ones_like(precomputed.times, dtype=bool)
            else:
                seg_mask = masks.get(seg_name)
            if seg_mask is None or not np.any(seg_mask):
                continue

            seg_len = int(np.sum(seg_mask))
            seg_sec = float(seg_len) / sfreq if sfreq > 0 else 0.0
            if min_segment_sec > 0 and seg_sec < min_segment_sec:
                continue

            req_cycles = max(float(min_cycles_per_band), 1.0) if np.isfinite(min_cycles_per_band) else 1.0
            min_viable_freq = (req_cycles / seg_sec) if seg_sec > 0 else np.inf

            seg_data = precomputed.data[epoch_idx][:, :, seg_mask]
            if seg_data.ndim != 3 or seg_data.shape[-1] < 2:
                continue

            for band in bands:
                if band not in freq_bands:
                    continue
                if band in precomputed.band_data and getattr(precomputed.band_data[band], "fmin", None) is not None:
                    fmin = float(precomputed.band_data[band].fmin)
                    fmax = float(precomputed.band_data[band].fmax)
                else:
                    fmin, fmax = freq_bands[band]
                try:
                    fmin = float(fmin)
                    fmax = float(fmax)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid frequency band range for '{band}': ({fmin}, {fmax})") from exc
                if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
                    raise ValueError(f"Invalid frequency band range for '{band}': ({fmin}, {fmax})")
                if fmax < min_viable_freq:
                    continue

                freqs = np.linspace(fmin, fmax, max(n_freqs_per_band, 2))
                freqs = freqs[np.asarray(freqs) >= min_viable_freq]
                if freqs.size < 2:
                    continue

                use_n_cycles = n_cycles
                if conn_mode == "cwt_morlet":
                    # Ensure n_cycles / f < duration (MNE constraint), with a safety factor.
                    base = float(use_n_cycles) if use_n_cycles is not None else 7.0
                    max_cycles = 0.9 * seg_sec * freqs
                    use_n_cycles = np.minimum(base, np.maximum(max_cycles, 0.5))

                for method in phase_measures:
                    method_use = method
                    method_label = method
                    con = _run(method_use, seg_data, freqs, fmin, fmax, use_n_cycles)

                    con_data = np.asarray(con.get_data())
                    if con_data.ndim == 3:
                        con_mean = np.nanmean(con_data, axis=0)
                        con_pairs = np.nanmean(con_mean, axis=-1) if con_mean.shape[-1] > 1 else con_mean[:, 0]
                    elif con_data.ndim == 2:
                        con_pairs = np.nanmean(con_data, axis=-1) if con_data.shape[-1] > 1 else con_data[:, 0]
                    elif con_data.ndim == 1:
                        con_pairs = con_data
                    else:
                        continue
                    con_pairs = np.asarray(con_pairs, dtype=float).reshape(-1)
                    if con_pairs.size != len(pair_names):
                        continue

                    if output_level == "full":
                        prefix = f"conn_{seg_name}_{band}_chpair_"
                        suffix = f"_{method_label}"
                        cols = [f"{prefix}{pair_name}{suffix}" for pair_name in pair_names]
                        for c, v in zip(cols, con_pairs):
                            if c in df.columns:
                                df.loc[epoch_idx, c] = float(v)
                    glob_col = f"conn_{seg_name}_{band}_global_{method_label}_mean"
                    if glob_col in df.columns:
                        df.loc[epoch_idx, glob_col] = float(np.nanmean(con_pairs))

                    if enable_graph_metrics:
                        adj = np.zeros((n_channels, n_channels), dtype=float)
                        adj[pair_i, pair_j] = con_pairs
                        adj[pair_j, pair_i] = con_pairs
                        g = _graph_metrics(adj, method_label, band, seg_name, conn_cfg)
                        for k, v in g.items():
                            if k in df.columns:
                                df.loc[epoch_idx, k] = float(v)


def _graph_metrics(
    adj: np.ndarray,
    measure: str,
    band: str,
    segment_name: str,
    conn_cfg: Any,
) -> Dict[str, float]:
    adj = np.asarray(adj, dtype=float)
    adj[~np.isfinite(adj)] = 0.0
    np.fill_diagonal(adj, 0.0)

    if isinstance(conn_cfg, ConnectivityConfig):
        top_prop = conn_cfg.graph_top_prop
        small_world_n_rand = conn_cfg.small_world_n_rand
    elif isinstance(conn_cfg, dict):
        top_prop = conn_cfg.get("graph_top_prop", 0.1)
        small_world_n_rand = conn_cfg.get("small_world_n_rand", 100)
    else:
        top_prop = 0.1
        small_world_n_rand = 100
    try:
        top_prop = float(top_prop)
    except (ValueError, TypeError):
        top_prop = 0.1
    if not np.isfinite(top_prop) or top_prop <= 0 or top_prop > 1:
        top_prop = 0.1

    try:
        small_world_n_rand = int(small_world_n_rand)
    except (ValueError, TypeError):
        small_world_n_rand = 100
    if small_world_n_rand > 0:
        small_world_n_rand = max(5, small_world_n_rand)

    adj_sym = _symmetrize_and_clip(adj)
    adj_abs = np.abs(adj_sym)
    adj_bin = _threshold_adjacency(adj_abs, top_proportion=top_prop)

    geff = _global_efficiency_weighted(adj_abs)

    try:
        clust_vals = nx.clustering(nx.from_numpy_array(adj_bin)).values()
        clust = float(np.mean(list(clust_vals))) if clust_vals else np.nan
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        clust = np.nan

    if small_world_n_rand > 0:
        try:
            smallworld = float(compute_small_world_sigma(adj_bin, n_rand=small_world_n_rand))
        except (ValueError, RuntimeError, ZeroDivisionError):
            smallworld = np.nan
    else:
        smallworld = np.nan

    return {
        NamingSchema.build("conn", segment_name, band, "global", f"{measure}_geff"): geff,
        NamingSchema.build("conn", segment_name, band, "global", f"{measure}_clust"): clust,
        NamingSchema.build("conn", segment_name, band, "global", f"{measure}_smallworld"): smallworld,
    }


def _compute_graph_metrics_for_epochs(
    con_vals: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n_channels: int,
    method_label: str,
    band: str,
    seg_name: str,
    conn_cfg: ConnectivityConfig,
    n_jobs: int,
    logger: Any,
) -> pd.DataFrame:
    """Compute graph metrics for all epochs from connectivity values.

    Common implementation for both phase-based and envelope-based connectivity.
    """
    def _graph_row(ep_idx: int) -> Dict[str, float]:
        adj = np.zeros((n_channels, n_channels), dtype=float)
        adj[pair_i, pair_j] = con_vals[ep_idx]
        adj[pair_j, pair_i] = con_vals[ep_idx]
        return _graph_metrics(adj, method_label, band, seg_name, conn_cfg)

    if logger is not None:
        logger.info(
            "Connectivity graph metrics: seg=%s band=%s method=%s (epochs=%d, channels=%d, small_world_n_rand=%s)",
            seg_name,
            band,
            method_label,
            int(con_vals.shape[0]),
            int(n_channels),
            str(conn_cfg.small_world_n_rand),
        )

    if n_jobs == 1:
        graph_rows = [_graph_row(ep_idx) for ep_idx in range(con_vals.shape[0])]
    else:
        graph_rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_graph_row)(ep_idx) for ep_idx in range(con_vals.shape[0])
        )
    return pd.DataFrame(graph_rows)


def _build_sliding_window_slices(
    n_times: int,
    sfreq: float,
    window_len_sec: float,
    window_step_sec: float,
    min_segment_samples: int,
) -> List[Tuple[int, int]]:
    """Build sliding-window sample slices for a segment."""
    if n_times <= 0 or not np.isfinite(sfreq) or sfreq <= 0:
        return []
    if window_len_sec <= 0 or window_step_sec <= 0:
        return []

    win_len = max(1, int(round(float(window_len_sec) * float(sfreq))))
    win_step = max(1, int(round(float(window_step_sec) * float(sfreq))))
    win_len = max(win_len, int(min_segment_samples))
    if win_len > n_times:
        return []

    windows: List[Tuple[int, int]] = []
    start = 0
    while start + win_len <= n_times:
        end = start + win_len
        windows.append((start, end))
        start += win_step
    return windows


def _dense_from_envelope_output(
    ec_data: np.ndarray,
    n_epochs: int,
    n_channels: int,
) -> Optional[np.ndarray]:
    """Convert envelope_correlation output to dense (epochs, channels, channels)."""
    expected_packed = int(n_channels * (n_channels + 1) // 2)
    dense: Optional[np.ndarray] = None

    if ec_data.ndim >= 1 and ec_data.shape[-1] == 1:
        ec_data = np.squeeze(ec_data, axis=-1)

    if ec_data.ndim == 4 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == n_channels and ec_data.shape[2] == n_channels:
        dense = np.nanmean(ec_data, axis=-1)
    elif ec_data.ndim == 3 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == n_channels and ec_data.shape[2] == n_channels:
        dense = ec_data
    elif ec_data.ndim == 3 and ec_data.shape[0] == n_channels and ec_data.shape[1] == n_channels and ec_data.shape[2] == n_epochs:
        dense = np.moveaxis(ec_data, -1, 0)
    elif ec_data.ndim == 3 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == expected_packed:
        packed = np.nanmean(ec_data, axis=-1)
        tril = np.tril_indices(n_channels, k=0)
        dense = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
        dense[:, tril[0], tril[1]] = packed
        dense[:, tril[1], tril[0]] = packed
    elif ec_data.ndim == 2 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == expected_packed:
        tril = np.tril_indices(n_channels, k=0)
        dense = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
        dense[:, tril[0], tril[1]] = ec_data
        dense[:, tril[1], tril[0]] = ec_data
    elif ec_data.ndim == 1 and ec_data.shape[0] == expected_packed:
        tril = np.tril_indices(n_channels, k=0)
        dense = np.zeros((1, n_channels, n_channels), dtype=float)
        dense[:, tril[0], tril[1]] = ec_data[None, :]
        dense[:, tril[1], tril[0]] = ec_data[None, :]
        dense = np.repeat(dense, n_epochs, axis=0)

    return dense


def _compute_windowed_wpli(
    analytic_seg: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    windows: List[Tuple[int, int]],
) -> np.ndarray:
    """Compute windowed wPLI per epoch and channel pair."""
    n_epochs = int(analytic_seg.shape[0])
    n_pairs = int(len(pair_i))
    n_windows = int(len(windows))
    out = np.full((n_epochs, n_windows, n_pairs), np.nan, dtype=float)

    eps = float(np.finfo(float).eps)
    for w_idx, (start, end) in enumerate(windows):
        ai = analytic_seg[:, pair_i, start:end]
        aj = analytic_seg[:, pair_j, start:end]
        imag_cross = np.imag(ai * np.conj(aj))
        num = np.abs(np.nanmean(imag_cross, axis=-1))
        den = np.nanmean(np.abs(imag_cross), axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, w_idx, :] = np.where(den > eps, num / den, np.nan)
    return out


def _compute_windowed_aec(
    analytic_seg: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    windows: List[Tuple[int, int]],
    *,
    conn_cfg: ConnectivityConfig,
) -> np.ndarray:
    """Compute windowed AEC per epoch and channel pair."""
    n_epochs = int(analytic_seg.shape[0])
    n_channels = int(analytic_seg.shape[1])
    n_pairs = int(len(pair_i))
    n_windows = int(len(windows))
    out = np.full((n_epochs, n_windows, n_pairs), np.nan, dtype=float)

    aec_mode = conn_cfg.aec_mode
    orthogonalize = "pairwise"
    if aec_mode in {"none", "raw", "no"}:
        orthogonalize = False
    elif aec_mode in {"sym", "symmetric"}:
        orthogonalize = "sym"

    for w_idx, (start, end) in enumerate(windows):
        ec = envelope_correlation(
            analytic_seg[:, :, start:end],
            orthogonalize=orthogonalize,
            log=False,
            absolute=bool(conn_cfg.aec_absolute),
        )
        dense = _dense_from_envelope_output(np.asarray(ec.get_data()), n_epochs, n_channels)
        if dense is None or dense.shape[0] != n_epochs:
            continue
        out[:, w_idx, :] = dense[:, pair_i, pair_j]
    return out


def _series_autocorr_lag(
    values: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Compute lagged autocorrelation along the window axis."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim < 2 or lag < 1:
        return np.full(arr.shape[:-1], np.nan, dtype=float)
    n_windows = int(arr.shape[-1])
    if n_windows <= lag:
        return np.full(arr.shape[:-1], np.nan, dtype=float)

    x = arr[..., :-lag]
    y = arr[..., lag:]
    valid = np.isfinite(x) & np.isfinite(y)
    valid_count = np.sum(valid, axis=-1)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)

    x_mean = np.nanmean(x, axis=-1, keepdims=True)
    y_mean = np.nanmean(y, axis=-1, keepdims=True)
    x_center = x - x_mean
    y_center = y - y_mean
    num = np.nansum(x_center * y_center, axis=-1)
    den = np.sqrt(np.nansum(x_center ** 2, axis=-1) * np.nansum(y_center ** 2, axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.where(den > 0, num / den, np.nan)
    ac = np.where(valid_count >= 3, ac, np.nan)
    return ac


def _window_vector_adjacent_stability(
    window_vectors: np.ndarray,
) -> np.ndarray:
    """Temporal stability of connectivity topology (adjacent-window vector similarity)."""
    n_epochs = int(window_vectors.shape[0])
    out = np.full((n_epochs,), np.nan, dtype=float)
    for ep_idx in range(n_epochs):
        series = np.asarray(window_vectors[ep_idx], dtype=float)
        if series.ndim != 2 or series.shape[0] < 2:
            continue
        sims: List[float] = []
        for t_idx in range(series.shape[0] - 1):
            v1 = series[t_idx]
            v2 = series[t_idx + 1]
            mask = np.isfinite(v1) & np.isfinite(v2)
            if int(np.sum(mask)) < 3:
                continue
            r = np.corrcoef(v1[mask], v2[mask])[0, 1]
            if np.isfinite(r):
                sims.append(float(r))
        if sims:
            out[ep_idx] = float(np.mean(sims))
    return out


def _fit_dynamic_state_labels(
    window_vectors: np.ndarray,
    *,
    n_states: int,
    random_state: int,
) -> Optional[np.ndarray]:
    """Fit k-means states over windowed connectivity vectors."""
    if KMeans is None or StandardScaler is None:
        return None
    if window_vectors.ndim != 3:
        return None

    n_epochs, n_windows, n_edges = window_vectors.shape
    flat = window_vectors.reshape(n_epochs * n_windows, n_edges)
    valid_rows = np.sum(np.isfinite(flat), axis=1) >= max(3, int(0.5 * n_edges))
    if int(np.sum(valid_rows)) < max(2 * n_states, n_states + 1):
        return None

    x = flat[valid_rows]
    col_means = np.nanmean(x, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    model = KMeans(n_clusters=n_states, n_init=10, random_state=random_state)
    labels_valid = model.fit_predict(x_scaled)

    labels = np.full((n_epochs * n_windows,), -1, dtype=int)
    labels[valid_rows] = labels_valid
    return labels.reshape(n_epochs, n_windows)


def _state_metrics_per_epoch(
    labels: np.ndarray,
    *,
    n_states: int,
    step_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute switching rate, dwell time, and state entropy per epoch."""
    n_epochs = int(labels.shape[0])
    switch_rate = np.full((n_epochs,), np.nan, dtype=float)
    dwell_sec = np.full((n_epochs,), np.nan, dtype=float)
    entropy = np.full((n_epochs,), np.nan, dtype=float)

    log_norm = np.log(float(n_states)) if n_states > 1 else np.nan

    for ep_idx in range(n_epochs):
        seq = labels[ep_idx]
        seq = seq[seq >= 0]
        if seq.size < 2:
            continue

        switches = np.sum(seq[1:] != seq[:-1])
        switch_rate[ep_idx] = float(switches) / float(max(1, seq.size - 1))

        run_lengths: List[int] = []
        run_start = 0
        for idx in range(1, seq.size + 1):
            if idx == seq.size or seq[idx] != seq[run_start]:
                run_lengths.append(int(idx - run_start))
                run_start = idx
        if run_lengths:
            dwell_sec[ep_idx] = float(np.mean(run_lengths) * float(step_sec))

        counts = np.bincount(seq, minlength=n_states).astype(float)
        p = counts / np.sum(counts)
        p = p[p > 0]
        if p.size > 0 and np.isfinite(log_norm) and log_norm > 0:
            entropy[ep_idx] = float((-np.sum(p * np.log(p))) / log_norm)

    return switch_rate, dwell_sec, entropy


def _build_roi_pair_index_map(
    *,
    config: Any,
    ch_names: List[str],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build mapping from ROI-pair label to channel-pair indices."""
    from eeg_pipeline.utils.analysis.channels import build_roi_map
    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions

    roi_defs = get_roi_definitions(config)
    if not roi_defs:
        return {}
    roi_map = build_roi_map(ch_names, roi_defs)
    roi_items = [(name, np.asarray(idxs, dtype=int)) for name, idxs in roi_map.items() if idxs]
    if not roi_items:
        return {}

    pair_map: Dict[str, np.ndarray] = {}
    for a_idx, (roi_a, ch_a) in enumerate(roi_items):
        set_a = set(ch_a.tolist())
        for b_idx in range(a_idx, len(roi_items)):
            roi_b, ch_b = roi_items[b_idx]
            set_b = set(ch_b.tolist())
            if a_idx == b_idx:
                mask = np.array([(int(i) in set_a and int(j) in set_a) for i, j in zip(pair_i, pair_j)], dtype=bool)
            else:
                mask = np.array(
                    [
                        ((int(i) in set_a and int(j) in set_b) or (int(i) in set_b and int(j) in set_a))
                        for i, j in zip(pair_i, pair_j)
                    ],
                    dtype=bool,
                )
            idxs = np.where(mask)[0]
            if idxs.size > 0:
                pair_map[f"{roi_a}-{roi_b}"] = idxs
    return pair_map


def extract_connectivity_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if not bands:
        return pd.DataFrame(), []

    conn_cfg = ConnectivityConfig.from_dict(ctx.config)
    _warn_if_phase_connectivity_without_spatial_transform(ctx.config, conn_cfg.measures, ctx.logger)
    expected_transform = _get_spatial_transform_type(ctx.config, feature_family="connectivity")

    precomputed = None
    getter = getattr(ctx, "get_precomputed_for_family", None)
    if callable(getter):
        precomputed = getter("connectivity")
    if precomputed is None:
        precomputed = getattr(ctx, "precomputed", None)
    if precomputed is not None:
        compatible, reason = _is_connectivity_precomputed_compatible(
            precomputed,
            expected_transform=expected_transform,
            bands=bands,
        )
        if not compatible:
            if ctx.logger is not None:
                ctx.logger.warning(
                    "Connectivity: existing precomputed intermediates are incompatible (%s); "
                    "recomputing connectivity-specific precomputed data.",
                    reason,
                )
            precomputed = None

    if precomputed is None:
        if getattr(ctx, "epochs", None) is None:
            return pd.DataFrame(), []
        if not getattr(ctx.epochs, "preload", False):
            ctx.logger.info("Preloading epochs data...")
            ctx.epochs.load_data()
        precomputed = precompute_data(
            ctx.epochs,
            bands,
            ctx.config,
            ctx.logger,
            windows_spec=ctx.windows,
            feature_family="connectivity",
            train_mask=getattr(ctx, "train_mask", None),
            analysis_mode=getattr(ctx, "analysis_mode", None),
        )
        setter = getattr(ctx, "set_precomputed_for_family", None)
        if callable(setter):
            setter("connectivity", precomputed)
        else:
            ctx.set_precomputed(precomputed)

    ctx_name = getattr(ctx, "name", None)
    segments: List[str] = []
    if ctx_name:
        segments = [ctx_name]
    elif getattr(ctx, "windows", None) is not None:
        for key in ("active", "plateau"):
            mask = ctx.windows.get_mask(key)
            if mask is not None and np.any(mask):
                segments = [key]
                break
        if not segments:
            mask_names = [k for k in ctx.windows.masks.keys() if k != "baseline"]
            if mask_names:
                segments = [mask_names[0]]
    if not segments:
        segments = ["full"]

    granularity = conn_cfg.granularity
    phase_estimator = conn_cfg.phase_estimator
    phase_estimator_effective = phase_estimator

    # Guardrail: detect CV/machine learning mode and warn/force within_epoch for phase estimator
    # across_epochs is cross-trial by nature and WILL leak test information in CV
    train_mask = getattr(ctx, "train_mask", None)

    if (
        granularity in {"subject", "condition"}
        and phase_estimator_effective == "within_epoch"
        and train_mask is None
    ):
        phase_estimator_effective = "across_epochs"
        if ctx.logger is not None:
            ctx.logger.info(
                "Connectivity: auto-setting phase_estimator='across_epochs' for granularity='%s' "
                "to avoid averaging per-epoch phase-connectivity estimates.",
                granularity,
            )

    if train_mask is not None and phase_estimator_effective == "across_epochs":
        if conn_cfg.force_within_epoch_for_ml:
            raise ValueError(
                "Connectivity: train_mask detected (CV/machine learning mode) with phase_estimator='across_epochs'. "
                "Across-epochs estimates leak test-trial information. "
                "Set feature_engineering.connectivity.phase_estimator='within_epoch' (recommended) "
                "or set feature_engineering.connectivity.force_within_epoch_for_ml=false to explicitly allow leakage."
            )
        else:
            if ctx.logger is not None:
                ctx.logger.warning(
                    "Connectivity: train_mask detected (CV/machine learning mode) with phase_estimator='across_epochs'. "
                    "CAUTION: Across-epochs estimates leak test-trial information and will inflate machine learning accuracy. "
                    "Consider using phase_estimator='within_epoch' for valid cross-validation."
                )

    df, cols = extract_connectivity_from_precomputed(
        precomputed,
        bands=bands,
        segments=segments,
        config=ctx.config,
        logger=ctx.logger,
        phase_estimator_override=phase_estimator_effective,
    )
    if df is None or df.empty:
        return pd.DataFrame(), []

    if granularity in {"subject", "condition"} and phase_estimator_effective == "across_epochs":
        n_epochs = int(df.shape[0])
        groups_map: Dict[str, np.ndarray] = {}
        if granularity == "subject":
            groups_map["__all__"] = np.arange(n_epochs, dtype=int)
        if granularity == "condition":
            events = getattr(ctx, "aligned_events", None)
            if events is not None and not getattr(events, "empty", True) and len(events) == n_epochs:
                cond_col = _resolve_connectivity_condition_column(
                    events=events,
                    config=ctx.config,
                    condition_column_cfg=conn_cfg.condition_column,
                )
                labels = events[cond_col].astype(str)
                included_mask, included_labels = _resolve_connectivity_condition_selection(
                    labels=labels,
                    selected_values=conn_cfg.condition_values,
                )
                min_n = conn_cfg.min_epochs_per_group
                for lab in included_labels:
                    idx = np.where(((labels == lab).to_numpy() & included_mask))[0]
                    if idx.size >= min_n:
                        groups_map[f"cond:{lab}"] = idx

                # If user selected specific condition values, clear phase-based columns for excluded trials.
                # Otherwise, excluded trials would retain the global across-epochs broadcast estimate.
                if not np.all(included_mask):
                    exclude_idx = np.where(~included_mask)[0]
                    conn_cfg_dict = ctx.config.get("feature_engineering.connectivity", {}) if hasattr(ctx.config, "get") else {}
                    phase_methods = _resolve_phase_measures(conn_cfg_dict if isinstance(conn_cfg_dict, dict) else {})
                    if phase_methods:
                        phase_cols = [
                            c
                            for c in df.columns
                            if any((f"_{m}" in c) or (f"{m}_" in c) for m in phase_methods)
                        ]
                        if phase_cols:
                            df.loc[exclude_idx, phase_cols] = np.nan

        if not groups_map:
            groups_map["__all__"] = np.arange(n_epochs, dtype=int)

        _apply_across_epochs_phase_estimates_inplace(
            df,
            precomputed=precomputed,
            segments=segments,
            bands=bands,
            epoch_groups=groups_map,
            config=ctx.config,
            logger=ctx.logger,
        )

    if granularity == "trial":
        df.attrs["feature_granularity"] = "trial"
        df.attrs["phase_estimator"] = phase_estimator_effective
        if phase_estimator_effective == "across_epochs":
            df.attrs["broadcast_warning"] = (
                "phase_estimator='across_epochs' produces one connectivity estimate per group "
                "that is broadcast to all trials. Treat rows as non-i.i.d.; aggregate before "
                "trial-level inference."
            )
            if ctx.logger is not None:
                ctx.logger.warning(
                    "Connectivity: granularity='trial' with phase_estimator='across_epochs' "
                    "broadcasts cross-trial estimates to all rows (non-i.i.d.)."
                )
        return df, cols

    n_epochs = int(df.shape[0])
    numeric = df.apply(pd.to_numeric, errors="coerce")

    if granularity == "subject":
        means = numeric.mean(axis=0)
        out = pd.DataFrame([means.values] * n_epochs, columns=means.index)
        # Mark as broadcast feature to prevent pseudo-replication in downstream stats
        out.attrs["feature_granularity"] = "subject"
        out.attrs["phase_estimator"] = phase_estimator_effective
        out.attrs["broadcast_warning"] = (
            "These features are subject-level means broadcast to all trials. "
            "Do NOT use as i.i.d. trial observations in correlations/regressions. "
            "Use primary_unit='subject' or aggregate to run/condition means first."
        )
        ctx.logger.info(
            "Connectivity features computed at subject-level and broadcast to %d trials. "
            "Mark as non-i.i.d. for downstream analysis.",
            n_epochs,
        )
        return out, list(out.columns)

    # condition-level: broadcast within each condition label
    events = getattr(ctx, "aligned_events", None)
    if events is None or getattr(events, "empty", True) or len(events) != n_epochs:
        raise ValueError(
            "Connectivity granularity='condition' requested but aligned_events is missing, empty, or length-mismatched "
            f"(n_epochs={n_epochs}, aligned_events_len={len(events) if events is not None else 'None'})."
        )

    cond_col = _resolve_connectivity_condition_column(
        events=events,
        config=ctx.config,
        condition_column_cfg=conn_cfg.condition_column,
    )

    labels = events[cond_col].astype(str)
    included_mask, included_labels = _resolve_connectivity_condition_selection(
        labels=labels,
        selected_values=conn_cfg.condition_values,
    )
    out = numeric.copy()

    min_n = conn_cfg.min_epochs_per_group
    counts = labels.value_counts()
    if included_labels:
        counts = counts.loc[[lab for lab in included_labels if lab in counts.index]]
    too_small = counts[counts < int(min_n)]
    if not too_small.empty:
        details = ", ".join([f"{k}={int(v)}" for k, v in too_small.items()])
        raise ValueError(
            "Connectivity granularity='condition' requested but some condition groups have too few epochs "
            f"(<{int(min_n)}): {details}."
        )

    # Exclude non-selected condition values (set to NaN).
    if not np.all(included_mask):
        out.loc[~included_mask, :] = np.nan

    for lab in included_labels:
        mask = ((labels == lab).to_numpy() & included_mask)
        grp_mean = numeric.loc[mask].mean(axis=0)
        out.loc[mask] = [grp_mean.values] * int(np.sum(mask))

    out.columns = df.columns
    # Mark as broadcast feature to prevent pseudo-replication
    out.attrs["feature_granularity"] = "condition"
    out.attrs["phase_estimator"] = phase_estimator_effective
    out.attrs["broadcast_warning"] = (
        "These features are condition-level means broadcast to all trials within condition. "
        "Do NOT use as i.i.d. trial observations. Use primary_unit='condition' or aggregate first."
    )
    ctx.logger.info(
        "Connectivity features computed at condition-level and broadcast within groups. "
        "Mark as non-i.i.d. for downstream analysis."
    )
    return out, list(out.columns)


def extract_connectivity_from_precomputed(
    precomputed: Any, # PrecomputedData
    *,
    bands: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    config: Any = None,
    logger: Any = None,
    phase_estimator_override: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute connectivity measures from precomputed analytic signals."""
    if not precomputed.band_data:
        return pd.DataFrame(), []

    config = config or getattr(precomputed, "config", None) or {}
    logger = logger or getattr(precomputed, "logger", None)

    bands_use = list(precomputed.band_data.keys()) if bands is None else [b for b in bands if b in precomputed.band_data]
    if not bands_use:
        return pd.DataFrame(), []

    conn_cfg = ConnectivityConfig.from_dict(config)
    analysis_mode = str(
        get_nested_value(config, "feature_engineering.analysis_mode", "group_stats") or "group_stats"
    ).strip().lower()
    disable_dynamic_state_metrics = (analysis_mode == "trial_ml_safe")
    dynamic_state_skip_warned = False

    # Resolve phase measures from config
    supported_measures = {"wpli", "imcoh", "aec", "plv", "pli"}
    measures_cfg = conn_cfg.measures
    measures = {str(m).strip().lower() for m in measures_cfg}
    unknown = measures - supported_measures
    if unknown and logger is not None:
        logger.warning(
            "Connectivity: unsupported measures %s; ignoring.",
            ",".join(sorted(unknown)),
        )
    measures = measures & supported_measures
    enable_wpli = "wpli" in measures
    enable_imcoh = "imcoh" in measures
    enable_aec = "aec" in measures
    enable_plv = "plv" in measures
    enable_pli = "pli" in measures

    enable_aec_raw = conn_cfg.enable_aec_raw
    enable_aec_z = conn_cfg.enable_aec_z

    phase_measures = [
        m
        for m, enabled in (
            ("wpli", enable_wpli),
            ("imcoh", enable_imcoh),
            ("plv", enable_plv),
            ("pli", enable_pli),
        )
        if enabled
    ]

    dynamic_requested = bool(conn_cfg.dynamic_enabled and conn_cfg.dynamic_measures)

    if not phase_measures and not enable_aec and not dynamic_requested:
        if logger is not None:
            logger.warning("Connectivity: no supported measures selected; skipping extraction.")
        return pd.DataFrame(), []

    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None

    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            seg_mask_map = {target_name: mask}
        else:
            if logger is not None:
                logger.error(
                    "Connectivity: targeted window '%s' has no valid mask; skipping.",
                    target_name,
                )
            return pd.DataFrame(), []
    else:
        masks = get_segment_masks(precomputed.times, windows, precomputed.config)
        seg_mask_map = {k: v for k, v in masks.items() if v is not None}

    segments_use = segments if segments is not None else sorted(seg_mask_map.keys()) or ["full"]

    ch_names = list(getattr(precomputed, "ch_names", []))
    n_channels = len(ch_names)
    if n_channels < 2:
        if logger is not None:
            logger.warning("Connectivity: Fewer than 2 channels available; skipping extraction.")
        return pd.DataFrame(), []

    t_total0 = time.perf_counter()

    from eeg_pipeline.utils.parallel import get_n_jobs

    n_jobs = get_n_jobs(
        config,
        default=-1,
        config_path="feature_engineering.parallel.n_jobs_connectivity",
    )
    if logger is not None:
        logger.debug("Connectivity extraction: n_jobs=%s", n_jobs)
    n_epochs = int(precomputed.data.shape[0])

    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    pair_i, pair_j = np.triu_indices(n_channels, k=1)
    pair_names = [f"{ch_names[i]}-{ch_names[j]}" for i, j in zip(pair_i, pair_j)]
    indices = (pair_i.astype(int), pair_j.astype(int))

    n_freqs_per_band = conn_cfg.n_freqs_per_band
    conn_mode = conn_cfg.mode
    n_cycles = conn_cfg.n_cycles
    decim = conn_cfg.decim
    min_segment_samples = conn_cfg.min_segment_samples
    min_cycles_per_band = conn_cfg.min_cycles_per_band
    min_segment_sec = conn_cfg.min_segment_sec
    phase_estimator = conn_cfg.phase_estimator
    if phase_estimator_override is not None:
        override = str(phase_estimator_override).strip().lower()
        if override in {"within_epoch", "across_epochs"}:
            phase_estimator = override

    if phase_estimator == "across_epochs" and logger is not None:
        logger.info(
            "Connectivity: using across_epochs phase estimator (standard wPLI/PLV/PLI definition); "
            "values will be broadcast to all trials in each group."
        )

    try:
        sfreq = float(getattr(precomputed, "sfreq", None))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Connectivity extraction requires a valid precomputed.sfreq (sampling frequency)."
        ) from exc
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Connectivity extraction requires a valid precomputed.sfreq (sampling frequency).")

    def _is_wavelet_longer_than_signal_error(exc: BaseException) -> bool:
        msg = str(exc).strip().lower()
        return "wavelets is longer than the signal" in msg or "wavelet is longer than the signal" in msg

    def _safe_n_cycles_for_segment(
        base_n_cycles: Optional[float],
        freqs_hz: np.ndarray,
        sfreq_hz: float,
        n_times: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute safe n_cycles and filter out frequencies that would crash MNE.
        
        Returns:
            Tuple of (valid_freqs, n_cycles_for_valid_freqs)
        """
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        
        if n_times <= 0:
            return np.array([]), np.array([])

        default_cycles = 7.0
        try:
            base = float(base_n_cycles) if base_n_cycles is not None else default_cycles
        except (TypeError, ValueError):
            base = default_cycles

        # MNE's Morlet wavelet length check is based on:
        #   W = morlet(sfreq, freqs, n_cycles)
        #   if len(W[0]) > n_times: raise ValueError(...)
        #
        # In MNE, each wavelet uses:
        #   sigma_t = n_cycles / (2*pi*freq)
        #   t = arange(0, 5*sigma_t, 1/sfreq)
        #   W length = 2*len(t) - 1
        #
        # We compute the maximum n_cycles per frequency that guarantees len(W) <= n_times,
        # with a small safety factor to avoid boundary/rounding edge cases.
        n_sigma = 5.0
        half_len_max = int((n_times + 1) // 2)  # len(t) must be <= half_len_max
        if half_len_max <= 1:
            return np.array([]), np.array([])

        safety_factor = 0.95
        max_cycles_by_wavelet = (
            safety_factor
            * float(half_len_max)
            * (2.0 * np.pi * freqs_hz)
            / (n_sigma * sfreq_hz)
        )

        # For each frequency, cap by base (default 7) and enforce a minimum cycles threshold.
        safe_cycles = np.minimum(base, max_cycles_by_wavelet)

        # Keep at least 1 cycle for phase-based estimation; smaller values are not meaningful.
        min_required_cycles = 1.0
        valid_mask = np.isfinite(safe_cycles) & (safe_cycles >= min_required_cycles)
        return freqs_hz[valid_mask], safe_cycles[valid_mask]

    def _slice_epochs(arr_3d: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if isinstance(mask, np.ndarray):
            if mask.dtype != bool:
                return None
            if not np.any(mask):
                return None
            return arr_3d[:, :, mask]
        return arr_3d

    if phase_measures and spectral_connectivity_time is None:
        raise ImportError(
            "Phase-based connectivity extraction requires 'mne-connectivity'. "
            "Install it with: pip install mne-connectivity"
        )
    needs_envelope = bool(enable_aec or (dynamic_requested and "aec" in conn_cfg.dynamic_measures))
    if needs_envelope and envelope_correlation is None:
        raise ImportError(
            "Envelope-based connectivity extraction requires 'mne-connectivity'. "
            "Install it with: pip install mne-connectivity"
        )

    freq_bands = getattr(precomputed, "frequency_bands", None) or get_frequency_bands(config)

    use_task_parallel = bool(n_jobs > 1)
    inner_n_jobs = 1 if use_task_parallel else n_jobs
    graph_n_jobs = 1 if use_task_parallel else n_jobs
    # Connectivity tasks carry large NumPy arrays; threads avoid per-task process copies.
    task_parallel_backend = "threading"

    output_level = conn_cfg.output_level
    enable_graph_metrics = conn_cfg.enable_graph_metrics

    if logger is not None:
        logger.info(
            "Connectivity extraction setup: epochs=%d, channels=%d, pairs=%d, bands=%d, segments=%d, phase_measures=%d, aec=%s, output_level=%s, graph_metrics=%s",
            n_epochs,
            n_channels,
            int(len(pair_names)),
            int(len(bands_use)),
            int(len(segments_use)),
            int(len(phase_measures)),
            str(bool(enable_aec)),
            output_level,
            str(bool(enable_graph_metrics)),
        )

    def _phase_task(
        seg_name: str,
        band: str,
        method: str,
        seg_data: np.ndarray,
        freqs: np.ndarray,
        fmin: float,
        fmax: float,
        use_n_cycles: Any,
    ) -> pd.DataFrame:
        t0 = time.perf_counter()
        method_use = method
        method_label = method

        def _run(method_to_use: str, use_average: bool = False):
            return spectral_connectivity_time(
                seg_data,
                freqs=freqs,
                method=method_to_use,
                indices=indices,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                average=use_average,
                faverage=True,
                mode=conn_mode,
                n_cycles=use_n_cycles,
                decim=decim,
                n_jobs=inner_n_jobs,
                verbose=False,
            )

        # Determine whether to use across_epochs (average=True) or within_epoch (average=False)
        use_across_epochs = (phase_estimator == "across_epochs")
        
        try:
            con = _run(method_use, use_average=use_across_epochs)
        except ValueError as e:
            # User-facing requirement: warn and continue (do not crash the pipeline).
            #
            # We only suppress the known MNE Morlet constraint error. Other ValueErrors
            # still surface (misconfiguration, shape mismatches, etc.).
            if _is_wavelet_longer_than_signal_error(e):
                if logger is not None:
                    seg_n_times = int(seg_data.shape[-1])
                    seg_sec = float(seg_n_times) / sfreq if sfreq > 0 else np.nan
                    logger.warning(
                        "Connectivity: skipped %s for segment=%s band=%s (%.3fs; %d samples @ %.1f Hz): "
                        "Morlet wavelet longer than signal. Increase segment duration / raise fmin / or set a smaller "
                        "feature_engineering.connectivity.n_cycles.",
                        method_use,
                        seg_name,
                        band,
                        seg_sec,
                        seg_n_times,
                        float(sfreq),
                    )
                return pd.DataFrame()
            raise

        con_data = np.asarray(con.get_data())
        
        # Handle across_epochs mode: broadcast single result to all epochs
        if use_across_epochs:
            # con_data shape is (n_pairs,) or (n_pairs, n_freqs) when average=True
            if con_data.ndim == 1:
                con_vals = np.tile(con_data[None, :], (n_epochs, 1))
            elif con_data.ndim == 2:
                con_vals = np.tile(np.nanmean(con_data, axis=-1)[None, :], (n_epochs, 1))
            else:
                raise ValueError(
                    f"Connectivity: unexpected across_epochs connectivity shape {con_data.shape} "
                    f"for method='{method_use}', segment='{seg_name}', band='{band}'."
                )
        else:
            # within_epoch mode: per-trial connectivity
            if con_data.ndim == 2:
                con_data = con_data[None, :, :]
            if con_data.ndim == 3 and con_data.shape[-1] > 1:
                con_vals = np.nanmean(con_data, axis=-1)
            elif con_data.ndim == 3:
                con_vals = con_data[:, :, 0]
            else:
                raise ValueError(
                    f"Connectivity: unexpected within_epoch connectivity shape {con_data.shape} "
                    f"for method='{method_use}', segment='{seg_name}', band='{band}'."
                )
            if con_vals.shape[0] != n_epochs:
                raise ValueError(
                    f"Connectivity: connectivity epochs mismatch for method='{method_use}', segment='{seg_name}', "
                    f"band='{band}' (got {con_vals.shape[0]} epochs, expected {n_epochs})."
                )

        parts: List[pd.DataFrame] = []
        if output_level == "full":
            prefix = f"conn_{seg_name}_{band}_chpair_"
            suffix = f"_{method_label}"
            cols = [f"{prefix}{pair_name}{suffix}" for pair_name in pair_names]
            parts.append(pd.DataFrame(con_vals, columns=cols))

        glob_col = f"conn_{seg_name}_{band}_global_{method_label}_mean"
        parts.append(pd.DataFrame({glob_col: np.nanmean(con_vals, axis=1)}))

        if conn_cfg.enable_graph_metrics:
            graph_df = _compute_graph_metrics_for_epochs(
                con_vals, pair_i, pair_j, n_channels, method_label, band, seg_name,
                conn_cfg, graph_n_jobs, logger
            )
            parts.append(graph_df)

        df_out = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        if logger is not None:
            logger.debug("Connectivity task phase/%s/%s/%s finished in %.2fs (cols=%d)", seg_name, band, method, time.perf_counter() - t0, int(df_out.shape[1]))
        return df_out

    def _aec_task(seg_name: str, band: str, analytic_seg: np.ndarray) -> pd.DataFrame:
        t0 = time.perf_counter()
        aec_mode = conn_cfg.aec_mode
        orthogonalize = "pairwise"
        if aec_mode in {"none", "raw", "no"}:
            orthogonalize = False
        elif aec_mode in {"sym", "symmetric"}:
            orthogonalize = "sym"
        ec = envelope_correlation(
            analytic_seg,
            orthogonalize=orthogonalize,
            log=False,
            absolute=bool(conn_cfg.aec_absolute),
        )
        ec_data = np.asarray(ec.get_data())
        if ec_data.ndim >= 1 and ec_data.shape[-1] == 1:
            ec_data = np.squeeze(ec_data, axis=-1)

        expected_packed = int(n_channels * (n_channels + 1) // 2)
        dense: Optional[np.ndarray] = None

        if ec_data.ndim == 4 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == n_channels and ec_data.shape[2] == n_channels:
            dense = np.nanmean(ec_data, axis=-1)
        elif ec_data.ndim == 3 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == n_channels and ec_data.shape[2] == n_channels:
            dense = ec_data
        elif ec_data.ndim == 3 and ec_data.shape[0] == n_channels and ec_data.shape[1] == n_channels and ec_data.shape[2] == n_epochs:
            dense = np.moveaxis(ec_data, -1, 0)
        elif ec_data.ndim == 3 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == expected_packed:
            packed = np.nanmean(ec_data, axis=-1)
            tril = np.tril_indices(n_channels, k=0)
            dense = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
            dense[:, tril[0], tril[1]] = packed
            dense[:, tril[1], tril[0]] = packed
        elif ec_data.ndim == 2 and ec_data.shape[0] == n_epochs and ec_data.shape[1] == expected_packed:
            tril = np.tril_indices(n_channels, k=0)
            dense = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
            dense[:, tril[0], tril[1]] = ec_data
            dense[:, tril[1], tril[0]] = ec_data
        elif ec_data.ndim == 1 and ec_data.shape[0] == expected_packed:
            tril = np.tril_indices(n_channels, k=0)
            dense = np.zeros((1, n_channels, n_channels), dtype=float)
            dense[:, tril[0], tril[1]] = ec_data[None, :]
            dense[:, tril[1], tril[0]] = ec_data[None, :]
            dense = np.repeat(dense, n_epochs, axis=0)

        if dense is None or dense.shape[0] != n_epochs:
            return pd.DataFrame()

        aec_vals = dense[:, pair_i, pair_j]
        
        # Compute Fisher-z transform: z = atanh(r)
        # This is scientifically correct for averaging correlations across trials/subjects
        # Note: clip to avoid infinity at +/-1
        aec_vals_z = None
        if enable_aec_z:
            aec_clipped = np.clip(aec_vals, -0.9999, 0.9999)
            aec_vals_z = np.arctanh(aec_clipped)

        parts: List[pd.DataFrame] = []
        if output_level == "full":
            prefix = f"conn_{seg_name}_{band}_chpair_"
            
            # Raw AEC (r values)
            if enable_aec_raw:
                suffix = "_aec"
                cols = [f"{prefix}{pair_name}{suffix}" for pair_name in pair_names]
                parts.append(pd.DataFrame(aec_vals, columns=cols))
            
            # Fisher-z AEC (z values)
            if enable_aec_z and aec_vals_z is not None:
                suffix_z = "_aec_z"
                cols_z = [f"{prefix}{pair_name}{suffix_z}" for pair_name in pair_names]
                parts.append(pd.DataFrame(aec_vals_z, columns=cols_z))

        # Global means
        if enable_aec_raw:
            glob_col = f"conn_{seg_name}_{band}_global_aec_mean"
            parts.append(pd.DataFrame({glob_col: np.nanmean(aec_vals, axis=1)}))
        
        if enable_aec_z and aec_vals_z is not None:
            glob_col_z = f"conn_{seg_name}_{band}_global_aec_z_mean"
            parts.append(pd.DataFrame({glob_col_z: np.nanmean(aec_vals_z, axis=1)}))

        if conn_cfg.enable_graph_metrics:
            graph_df = _compute_graph_metrics_for_epochs(
                aec_vals, pair_i, pair_j, n_channels, "aec", band, seg_name,
                conn_cfg, graph_n_jobs, logger
            )
            parts.append(graph_df)

        return pd.concat(parts, axis=1) if parts else pd.DataFrame()

    tasks: List[Tuple[str, Tuple[Any, ...]]] = []
    for seg_name in segments_use:
        seg_mask = seg_mask_map.get(seg_name)
        if seg_mask is None and seg_name == "full":
            seg_data = precomputed.data
        else:
            seg_data = _slice_epochs(precomputed.data, seg_mask)
        if seg_data is None:
            continue

        seg_n_times = int(seg_data.shape[-1])
        if seg_n_times < min_segment_samples:
            continue
        seg_duration = float(seg_n_times) / sfreq
        if min_segment_sec > 0 and seg_duration < min_segment_sec:
            continue
        req_cycles = max(float(min_cycles_per_band), 1.0) if np.isfinite(min_cycles_per_band) else 1.0
        min_viable_freq = (req_cycles / seg_duration) if seg_duration > 0 else np.inf

        for band in bands_use:
            if band not in freq_bands:
                continue
            if band in precomputed.band_data and getattr(precomputed.band_data[band], "fmin", None) is not None:
                fmin = float(precomputed.band_data[band].fmin)
                fmax = float(precomputed.band_data[band].fmax)
            else:
                fmin, fmax = freq_bands[band]
            try:
                fmin = float(fmin)
                fmax = float(fmax)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
                continue

            # Skip band if segment is too short for required cycles in this band
            if fmax < min_viable_freq:
                if logger is not None:
                    logger.debug(f"Connectivity: skipping band {band} for segment {seg_name} (fmax {fmax} < min_viable {min_viable_freq:.2f}Hz)")
                continue
            
            if not _validate_segment_duration_for_connectivity(seg_duration, fmin, min_cycles_per_band, band, logger):
                continue

            freqs = np.linspace(fmin, fmax, max(n_freqs_per_band, 2))
            
            # Filter freqs to those that actually fit
            freqs = freqs[np.asarray(freqs) >= min_viable_freq]
            if freqs.size < 2:
                if logger is not None:
                    logger.debug(f"Connectivity: skipping band {band} for segment {seg_name} (not enough valid frequencies after filtering)")
                continue

            use_n_cycles = n_cycles
            if conn_mode == "cwt_morlet":
                # Apply strict pre-flight check to filter frequencies that would crash MNE
                valid_freqs, valid_cycles = _safe_n_cycles_for_segment(n_cycles, freqs, sfreq, seg_n_times)
                if valid_freqs.size < 2:
                    if logger is not None:
                        logger.warning(
                            f"Connectivity: skipping band '{band}' for segment '{seg_name}' - "
                            f"segment too short ({seg_n_times} samples = {seg_n_times/sfreq:.2f}s) for reliable "
                            f"phase connectivity at these frequencies (need longer epochs or higher fmin)."
                        )
                    continue
                freqs = valid_freqs
                use_n_cycles = valid_cycles
            
            for method in phase_measures:
                tasks.append(("phase", (seg_name, band, method, seg_data, freqs, fmin, fmax, use_n_cycles)))

            if enable_aec and band in precomputed.band_data:
                analytic_full = precomputed.band_data[band].analytic
                analytic_seg = _slice_epochs(analytic_full, seg_mask)
                if analytic_seg is None:
                    continue
                if analytic_seg.shape[-1] < min_segment_samples:
                    continue
                tasks.append(("aec", (seg_name, band, analytic_seg)))

    def _run_task(task: Tuple[str, Tuple[Any, ...]]) -> pd.DataFrame:
        kind, args = task
        if kind == "phase":
            return _phase_task(*args)
        return _aec_task(*args)
    roi_pair_map: Dict[str, np.ndarray] = {}
    if conn_cfg.dynamic_enabled and conn_cfg.dynamic_include_roi_pairs and output_level == "full":
        roi_pair_map = _build_roi_pair_index_map(
            config=config,
            ch_names=ch_names,
            pair_i=pair_i,
            pair_j=pair_j,
        )

    def _dynamic_task(
        seg_name: str,
        band: str,
        method: str,
        analytic_seg: np.ndarray,
        windows_slices: List[Tuple[int, int]],
    ) -> pd.DataFrame:
        nonlocal dynamic_state_skip_warned
        n_windows = int(len(windows_slices))
        if n_windows < int(conn_cfg.dynamic_min_windows):
            return pd.DataFrame()

        if method == "wpli":
            window_vals = _compute_windowed_wpli(analytic_seg, pair_i, pair_j, windows_slices)
        elif method == "aec":
            window_vals = _compute_windowed_aec(
                analytic_seg, pair_i, pair_j, windows_slices, conn_cfg=conn_cfg
            )
        else:
            return pd.DataFrame()

        if window_vals.ndim != 3 or window_vals.shape[0] != n_epochs:
            return pd.DataFrame()

        mean_stat = f"{method}swmean"
        std_stat = f"{method}swstd"
        ac_stat = f"{method}swac{int(conn_cfg.dynamic_autocorr_lag)}"

        edge_mean = np.nanmean(window_vals, axis=1)
        edge_std = np.nanstd(window_vals, axis=1)
        edge_ac = _series_autocorr_lag(
            np.moveaxis(window_vals, 1, 2), lag=int(conn_cfg.dynamic_autocorr_lag)
        )

        parts: List[pd.DataFrame] = []
        if output_level == "full":
            cols_mean = [
                NamingSchema.build(
                    "conn", seg_name, band, "chpair", mean_stat, channel_pair=pair_name
                )
                for pair_name in pair_names
            ]
            cols_std = [
                NamingSchema.build(
                    "conn", seg_name, band, "chpair", std_stat, channel_pair=pair_name
                )
                for pair_name in pair_names
            ]
            cols_ac = [
                NamingSchema.build(
                    "conn", seg_name, band, "chpair", ac_stat, channel_pair=pair_name
                )
                for pair_name in pair_names
            ]
            parts.append(pd.DataFrame(edge_mean, columns=cols_mean))
            parts.append(pd.DataFrame(edge_std, columns=cols_std))
            parts.append(pd.DataFrame(edge_ac, columns=cols_ac))

            if roi_pair_map:
                roi_data: Dict[str, np.ndarray] = {}
                for roi_pair, idxs in roi_pair_map.items():
                    idxs = np.asarray(idxs, dtype=int)
                    if idxs.size == 0:
                        continue
                    roi_series = np.nanmean(window_vals[:, :, idxs], axis=2)
                    roi_data[
                        NamingSchema.build("conn", seg_name, band, "roi", mean_stat, channel=roi_pair)
                    ] = np.nanmean(roi_series, axis=1)
                    roi_data[
                        NamingSchema.build("conn", seg_name, band, "roi", std_stat, channel=roi_pair)
                    ] = np.nanstd(roi_series, axis=1)
                    roi_data[
                        NamingSchema.build("conn", seg_name, band, "roi", ac_stat, channel=roi_pair)
                    ] = _series_autocorr_lag(
                        roi_series, lag=int(conn_cfg.dynamic_autocorr_lag)
                    )
                if roi_data:
                    parts.append(pd.DataFrame(roi_data))

        parts.append(
            pd.DataFrame(
                {
                    NamingSchema.build("conn", seg_name, band, "global", mean_stat): np.nanmean(
                        edge_mean, axis=1
                    ),
                    NamingSchema.build("conn", seg_name, band, "global", std_stat): np.nanmean(
                        edge_std, axis=1
                    ),
                    NamingSchema.build("conn", seg_name, band, "global", ac_stat): np.nanmean(
                        edge_ac, axis=1
                    ),
                }
            )
        )

        topo_stability = _window_vector_adjacent_stability(window_vals)
        parts.append(
            pd.DataFrame(
                {
                    NamingSchema.build(
                        "conn", seg_name, band, "global", f"{method}swtopostab"
                    ): topo_stability
                }
            )
        )

        if conn_cfg.dynamic_state_enabled and n_windows >= int(conn_cfg.dynamic_state_min_windows):
            if disable_dynamic_state_metrics:
                if logger is not None and not dynamic_state_skip_warned:
                    logger.warning(
                        "Connectivity dynamic state-transition metrics are disabled in "
                        "analysis_mode='trial_ml_safe' because state clustering pools "
                        "across trials and can leak fold information."
                    )
                    dynamic_state_skip_warned = True
            else:
                n_states = min(int(conn_cfg.dynamic_state_n_states), max(2, n_windows - 1))
                random_state = (
                    int(conn_cfg.dynamic_state_random_state)
                    if conn_cfg.dynamic_state_random_state is not None
                    else 0
                )
                state_labels = _fit_dynamic_state_labels(
                    window_vals,
                    n_states=n_states,
                    random_state=random_state,
                )
                if state_labels is not None:
                    switch_rate, dwell_sec, entropy = _state_metrics_per_epoch(
                        state_labels,
                        n_states=n_states,
                        step_sec=float(conn_cfg.sliding_window_step),
                    )
                    parts.append(
                        pd.DataFrame(
                            {
                                NamingSchema.build(
                                    "conn", seg_name, band, "global", f"{method}swswitch"
                                ): switch_rate,
                                NamingSchema.build(
                                    "conn", seg_name, band, "global", f"{method}swdwellsec"
                                ): dwell_sec,
                                NamingSchema.build(
                                    "conn", seg_name, band, "global", f"{method}swstateent"
                                ): entropy,
                            }
                        )
                    )

        return pd.concat(parts, axis=1) if parts else pd.DataFrame()

    dynamic_tasks: List[Tuple[str, Tuple[Any, ...]]] = []
    if conn_cfg.dynamic_enabled:
        if conn_cfg.dynamic_state_enabled and (KMeans is None or StandardScaler is None) and logger is not None:
            logger.warning(
                "Connectivity dynamic states requested, but scikit-learn is unavailable; "
                "state-transition metrics will be skipped."
            )
        for seg_name in segments_use:
            seg_mask = seg_mask_map.get(seg_name)
            if seg_mask is None and seg_name == "full":
                seg_data = precomputed.data
            else:
                seg_data = _slice_epochs(precomputed.data, seg_mask)
            if seg_data is None:
                continue
            seg_n_times = int(seg_data.shape[-1])
            if seg_n_times < min_segment_samples:
                continue
            seg_duration = float(seg_n_times) / sfreq
            if min_segment_sec > 0 and seg_duration < min_segment_sec:
                continue

            windows_slices = _build_sliding_window_slices(
                seg_n_times,
                sfreq,
                float(conn_cfg.sliding_window_len),
                float(conn_cfg.sliding_window_step),
                min_segment_samples,
            )
            if len(windows_slices) < int(conn_cfg.dynamic_min_windows):
                continue

            for band in bands_use:
                if band not in precomputed.band_data:
                    continue
                analytic_full = precomputed.band_data[band].analytic
                analytic_seg = _slice_epochs(analytic_full, seg_mask)
                if analytic_seg is None or analytic_seg.shape[-1] != seg_n_times:
                    continue
                for method in conn_cfg.dynamic_measures:
                    dynamic_tasks.append(
                        ("dynamic", (seg_name, band, method, analytic_seg, windows_slices))
                    )

    task_times: Dict[str, float] = {"phase": 0.0, "aec": 0.0, "dynamic": 0.0}
    task_counts: Dict[str, int] = {"phase": 0, "aec": 0, "dynamic": 0}

    if logger is not None:
        logger.info(
            "Running connectivity tasks: n_static=%d, n_dynamic=%d (threaded=%s)",
            int(len(tasks)),
            int(len(dynamic_tasks)),
            str(bool(use_task_parallel and (len(tasks) + len(dynamic_tasks)) > 1)),
        )

    def _timed_run_static(task: Tuple[str, Tuple[Any, ...]]) -> pd.DataFrame:
        kind, _ = task
        t0 = time.perf_counter()
        df_task = _run_task(task)
        dt = time.perf_counter() - t0
        if kind in task_times:
            task_times[kind] += dt
            task_counts[kind] += 1
        return df_task

    static_dfs: List[pd.DataFrame] = []
    if tasks:
        if use_task_parallel and len(tasks) > 1:
            static_dfs = Parallel(n_jobs=n_jobs, backend=task_parallel_backend)(
                delayed(_timed_run_static)(task) for task in tasks
            )
        else:
            static_dfs = [_timed_run_static(task) for task in tasks]

    def _run_dynamic(task: Tuple[str, Tuple[Any, ...]]) -> pd.DataFrame:
        _, args = task
        return _dynamic_task(*args)

    def _timed_run_dynamic(task: Tuple[str, Tuple[Any, ...]]) -> pd.DataFrame:
        kind, _ = task
        t0 = time.perf_counter()
        df_task = _run_dynamic(task)
        dt = time.perf_counter() - t0
        if kind in task_times:
            task_times[kind] += dt
            task_counts[kind] += 1
        return df_task

    dynamic_dfs: List[pd.DataFrame] = []
    if dynamic_tasks:
        if use_task_parallel and len(dynamic_tasks) > 1:
            dynamic_dfs = Parallel(n_jobs=n_jobs, backend=task_parallel_backend)(
                delayed(_timed_run_dynamic)(task) for task in dynamic_tasks
            )
        else:
            dynamic_dfs = [_timed_run_dynamic(task) for task in dynamic_tasks]

    dfs = [
        df_task
        for df_task in (static_dfs + dynamic_dfs)
        if df_task is not None and not df_task.empty
    ]
    if not dfs:
        return pd.DataFrame(), []

    t_concat0 = time.perf_counter()
    df = pd.concat(dfs, axis=1)
    if logger is not None:
        logger.info(
            "Connectivity post-processing: task_time_phase=%.2fs (%d), task_time_aec=%.2fs (%d), task_time_dynamic=%.2fs (%d), concat=%.2fs, total=%.2fs, out_shape=(%d,%d)",
            float(task_times["phase"]),
            int(task_counts["phase"]),
            float(task_times["aec"]),
            int(task_counts["aec"]),
            float(task_times["dynamic"]),
            int(task_counts["dynamic"]),
            time.perf_counter() - t_concat0,
            time.perf_counter() - t_total0,
            int(df.shape[0]),
            int(df.shape[1]),
        )

    return df, list(df.columns)


###################################################################
# DIRECTED CONNECTIVITY (PSI, DTF, PDC)
###################################################################


def _compute_cross_spectrum(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_fft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-spectral density matrix using Welch's method.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    fmin : float
        Minimum frequency of interest
    fmax : float
        Maximum frequency of interest
    n_fft : int, optional
        FFT length. If None, uses n_times.
        
    Returns
    -------
    csd : np.ndarray
        Cross-spectral density of shape (n_epochs, n_channels, n_channels, n_freqs)
    freqs : np.ndarray
        Frequency vector
    """
    from scipy.signal import csd as scipy_csd
    
    n_epochs, n_channels, n_times = data.shape
    
    if n_fft is None:
        n_fft = min(n_times, int(sfreq * 2))
    n_fft = max(n_fft, 64)
    
    n_overlap = n_fft // 2
    
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[freq_mask]
    n_freqs = len(freqs_band)
    
    if n_freqs < 2:
        return np.array([]), freqs_band
    
    csd_matrix = np.zeros((n_epochs, n_channels, n_channels, n_freqs), dtype=complex)
    
    for ep_idx in range(n_epochs):
        for i in range(n_channels):
            for j in range(n_channels):
                _, csd_ij = scipy_csd(
                    data[ep_idx, i],
                    data[ep_idx, j],
                    fs=sfreq,
                    nperseg=n_fft,
                    noverlap=n_overlap,
                    return_onesided=True,
                )
                csd_matrix[ep_idx, i, j, :] = csd_ij[freq_mask]
    
    return csd_matrix, freqs_band


def _compute_psi_imaginary(
    csd: np.ndarray,
) -> np.ndarray:
    """
    Compute Phase Slope Index using imaginary part of coherency.
    
    This is a more robust version that uses the imaginary part of coherency
    to reduce sensitivity to volume conduction (zero-lag effects).
    
    Parameters
    ----------
    csd : np.ndarray
        Cross-spectral density of shape (n_epochs, n_channels, n_channels, n_freqs)
        
    Returns
    -------
    psi : np.ndarray
        Phase slope index of shape (n_epochs, n_channels, n_channels)
    """
    n_epochs, n_channels, _, n_freqs = csd.shape
    
    if n_freqs < 2:
        return np.full((n_epochs, n_channels, n_channels), np.nan)
    
    psi = np.zeros((n_epochs, n_channels, n_channels))
    
    for ep_idx in range(n_epochs):
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    continue
                
                csd_ij = csd[ep_idx, i, j, :]
                
                norm = np.sqrt(
                    np.abs(csd[ep_idx, i, i, :]) * np.abs(csd[ep_idx, j, j, :]) + 1e-12
                )
                coherency = csd_ij / norm
                
                # Standard PSI definition uses adjacent-frequency coherency products:
                #   PSI = Im( sum_f conj(C(f)) * C(f+1) )
                # Constant phase-lag across frequency should produce ~0 PSI.
                psi_sum = np.sum(np.conj(coherency[:-1]) * coherency[1:])
                psi[ep_idx, i, j] = np.imag(psi_sum)
    
    return psi


def _fit_mvar_model(
    data: np.ndarray,
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Multivariate Autoregressive (MVAR) model using Yule-Walker equations.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_channels, n_times)
    order : int
        Model order (number of lags)
        
    Returns
    -------
    A : np.ndarray
        AR coefficients of shape (order, n_channels, n_channels)
    sigma : np.ndarray
        Residual covariance of shape (n_channels, n_channels)
    """
    n_channels, n_times = data.shape
    
    # Scientific validity: MVAR fits are unstable with too few samples relative
    # to model dimensionality. This guard is intentionally conservative.
    min_required = int(max(order * n_channels + 1, 3 * order * n_channels))
    if n_times < min_required:
        return np.array([]), np.array([])
    
    data_centered = data - data.mean(axis=1, keepdims=True)
    
    R = np.zeros((order + 1, n_channels, n_channels))
    for lag in range(order + 1):
        if lag == 0:
            R[0] = np.dot(data_centered, data_centered.T) / n_times
        else:
            R[lag] = np.dot(data_centered[:, lag:], data_centered[:, :-lag].T) / (n_times - lag)
    
    block_size = n_channels
    R_matrix = np.zeros((order * block_size, order * block_size))
    r_vector = np.zeros((order * block_size, block_size))
    
    for i in range(order):
        for j in range(order):
            lag = abs(i - j)
            if i >= j:
                R_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = R[lag]
            else:
                R_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = R[lag].T
        r_vector[i*block_size:(i+1)*block_size, :] = R[i + 1]
    
    try:
        A_flat = np.linalg.solve(R_matrix, r_vector)
    except np.linalg.LinAlgError:
        return np.array([]), np.array([])

    # Reject numerically ill-conditioned systems (DTF/PDC will be unreliable).
    try:
        cond = float(np.linalg.cond(R_matrix))
        if not np.isfinite(cond) or cond > 1e12:
            return np.array([]), np.array([])
    except Exception:
        return np.array([]), np.array([])
    
    A = np.zeros((order, n_channels, n_channels))
    for i in range(order):
        A[i] = A_flat[i*block_size:(i+1)*block_size, :].T
    
    sigma = R[0].copy()
    for i in range(order):
        sigma -= np.dot(A[i], R[i + 1].T)
    
    return A, sigma


def _compute_dtf_from_mvar(
    A: np.ndarray,
    sigma: np.ndarray,
    freqs: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Compute Directed Transfer Function from MVAR coefficients.
    
    DTF measures the causal influence from channel j to channel i at each frequency.
    
    Reference: Kaminski & Blinowska (1991) "A new method of the description of the 
    information flow in the brain structures"
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients of shape (order, n_channels, n_channels)
    sigma : np.ndarray
        Residual covariance of shape (n_channels, n_channels)
    freqs : np.ndarray
        Frequency vector
    sfreq : float
        Sampling frequency
        
    Returns
    -------
    dtf : np.ndarray
        Directed transfer function of shape (n_channels, n_channels, n_freqs)
        DTF[i, j, f] = causal influence from j to i at frequency f
    """
    if A.size == 0:
        return np.array([])
    
    order, n_channels, _ = A.shape
    n_freqs = len(freqs)
    
    H = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
    
    for f_idx, freq in enumerate(freqs):
        A_f = np.eye(n_channels, dtype=complex)
        for k in range(order):
            A_f -= A[k] * np.exp(-2j * np.pi * freq * (k + 1) / sfreq)
        
        try:
            H[:, :, f_idx] = np.linalg.inv(A_f)
        except np.linalg.LinAlgError:
            H[:, :, f_idx] = np.nan
    
    dtf = np.zeros((n_channels, n_channels, n_freqs))
    
    for f_idx in range(n_freqs):
        H_f = H[:, :, f_idx]
        
        for i in range(n_channels):
            norm = np.sqrt(np.sum(np.abs(H_f[i, :]) ** 2) + 1e-12)
            for j in range(n_channels):
                dtf[i, j, f_idx] = np.abs(H_f[i, j]) / norm
    
    return dtf


def _compute_pdc_from_mvar(
    A: np.ndarray,
    sigma: np.ndarray,
    freqs: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Compute Partial Directed Coherence from MVAR coefficients.
    
    PDC measures the direct causal influence from channel j to channel i,
    partialling out indirect effects through other channels.
    
    Reference: Baccala & Sameshima (2001) "Partial directed coherence: a new 
    concept in neural structure determination"
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients of shape (order, n_channels, n_channels)
    sigma : np.ndarray
        Residual covariance of shape (n_channels, n_channels)
    freqs : np.ndarray
        Frequency vector
    sfreq : float
        Sampling frequency
        
    Returns
    -------
    pdc : np.ndarray
        Partial directed coherence of shape (n_channels, n_channels, n_freqs)
        PDC[i, j, f] = direct causal influence from j to i at frequency f
    """
    if A.size == 0:
        return np.array([])
    
    order, n_channels, _ = A.shape
    n_freqs = len(freqs)
    
    pdc = np.zeros((n_channels, n_channels, n_freqs))
    
    for f_idx, freq in enumerate(freqs):
        A_f = np.eye(n_channels, dtype=complex)
        for k in range(order):
            A_f -= A[k] * np.exp(-2j * np.pi * freq * (k + 1) / sfreq)
        
        for j in range(n_channels):
            norm = np.sqrt(np.sum(np.abs(A_f[:, j]) ** 2) + 1e-12)
            for i in range(n_channels):
                pdc[i, j, f_idx] = np.abs(A_f[i, j]) / norm
    
    return pdc


def _compute_directed_connectivity_epoch(
    ep_idx: int,
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_freqs: int,
    mvar_order: int,
    methods: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute directed connectivity for a single epoch.
    
    Parameters
    ----------
    ep_idx : int
        Epoch index (for logging)
    data : np.ndarray
        EEG data of shape (n_channels, n_times)
    sfreq : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    n_freqs : int
        Number of frequency bins
    mvar_order : int
        MVAR model order
    methods : List[str]
        List of methods to compute ('psi', 'dtf', 'pdc')
        
    Returns
    -------
    results : Dict[str, np.ndarray]
        Dictionary mapping method names to connectivity matrices
    """
    results = {}
    n_channels = data.shape[0]
    n_times = int(data.shape[1])
    
    freqs = np.linspace(fmin, fmax, n_freqs)
    
    if "psi" in methods:
        csd, freqs_csd = _compute_cross_spectrum(
            data[np.newaxis, :, :], sfreq, fmin, fmax
        )
        if csd.size > 0:
            psi = _compute_psi_imaginary(csd)
            results["psi"] = psi[0]
        else:
            results["psi"] = np.full((n_channels, n_channels), np.nan)
    
    if "dtf" in methods or "pdc" in methods:
        # Additional adequacy guard: DTF/PDC via MVAR needs substantially more data
        # than PSI and is highly sensitive to short segments.
        # (PSI uses cross-spectrum; MVAR estimates many parameters.)
        if n_times < int(max(3 * mvar_order * n_channels, mvar_order + 2)):
            if "dtf" in methods:
                results["dtf"] = np.full((n_channels, n_channels), np.nan)
            if "pdc" in methods:
                results["pdc"] = np.full((n_channels, n_channels), np.nan)
            return results

        A, sigma = _fit_mvar_model(data, mvar_order)
        
        if A.size > 0:
            if "dtf" in methods:
                dtf = _compute_dtf_from_mvar(A, sigma, freqs, sfreq)
                if dtf.size > 0:
                    results["dtf"] = np.nanmean(dtf, axis=2)
                else:
                    results["dtf"] = np.full((n_channels, n_channels), np.nan)
            
            if "pdc" in methods:
                pdc = _compute_pdc_from_mvar(A, sigma, freqs, sfreq)
                if pdc.size > 0:
                    results["pdc"] = np.nanmean(pdc, axis=2)
                else:
                    results["pdc"] = np.full((n_channels, n_channels), np.nan)
        else:
            if "dtf" in methods:
                results["dtf"] = np.full((n_channels, n_channels), np.nan)
            if "pdc" in methods:
                results["pdc"] = np.full((n_channels, n_channels), np.nan)
    
    return results


def extract_directed_connectivity_from_precomputed(
    precomputed: Any,
    *,
    bands: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    config: Any = None,
    logger: Any = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract directed connectivity features from precomputed data.
    
    Parameters
    ----------
    precomputed : PrecomputedData
        Precomputed intermediate data with band-filtered signals
    bands : List[str], optional
        Frequency bands to process
    segments : List[str], optional
        Time segments to process
    config : Any
        Configuration object
    logger : Any
        Logger instance
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with directed connectivity features
    columns : List[str]
        List of feature column names
    """
    if not precomputed.band_data:
        return pd.DataFrame(), []
    
    config = config or getattr(precomputed, "config", None) or {}
    logger = logger or getattr(precomputed, "logger", None)
    
    bands_use = (
        list(precomputed.band_data.keys()) 
        if bands is None 
        else [b for b in bands if b in precomputed.band_data]
    )
    if not bands_use:
        return pd.DataFrame(), []
    
    if config is None:
        directed_cfg = {}
    elif hasattr(config, "get") and not isinstance(config, dict):
        directed_cfg = config.get("feature_engineering.directedconnectivity", {}) or {}
    else:
        directed_cfg = get_nested_value(config, "feature_engineering.directedconnectivity", {}) or {}
    
    enable_psi = bool(directed_cfg.get("enable_psi", True))
    enable_dtf = bool(directed_cfg.get("enable_dtf", False))
    enable_pdc = bool(directed_cfg.get("enable_pdc", False))
    
    methods = [m for m, enabled in (("psi", enable_psi), ("dtf", enable_dtf), ("pdc", enable_pdc)) if enabled]
    
    if not methods:
        if logger is not None:
            logger.info("Directed connectivity: no methods enabled; skipping extraction.")
        return pd.DataFrame(), []
    
    output_level = str(directed_cfg.get("output_level", "full")).strip().lower()
    if output_level not in {"full", "global_only"}:
        output_level = "full"
    
    mvar_order = int(directed_cfg.get("mvar_order", 10))
    n_freqs = int(directed_cfg.get("n_freqs", 16))
    min_segment_samples = int(directed_cfg.get("min_segment_samples", 100))
    min_samples_per_mvar_parameter = int(
        directed_cfg.get("min_samples_per_mvar_parameter", 10)
    )
    min_samples_per_mvar_parameter = max(3, min_samples_per_mvar_parameter)
    
    sfreq = float(getattr(precomputed, "sfreq", None))
    
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Directed connectivity extraction requires a valid precomputed.sfreq (sampling frequency).")
    
    ch_names = list(getattr(precomputed, "ch_names", []))
    n_channels = len(ch_names)
    if n_channels < 2:
        if logger is not None:
            logger.warning("Directed connectivity: fewer than 2 channels; skipping.")
        return pd.DataFrame(), []
    
    pair_i, pair_j = np.triu_indices(n_channels, k=1)
    pair_names = [f"{ch_names[i]}-{ch_names[j]}" for i, j in zip(pair_i, pair_j)]
    
    freq_bands = getattr(precomputed, "frequency_bands", None) or get_frequency_bands(config)
    
    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    
    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            seg_mask_map = {target_name: mask}
        else:
            if logger is not None:
                logger.error(
                    "Directed connectivity: targeted window '%s' has no valid mask; skipping.",
                    target_name,
                )
            return pd.DataFrame(), []
    else:
        masks = get_segment_masks(precomputed.times, windows, precomputed.config)
        seg_mask_map = {k: v for k, v in masks.items() if v is not None}
    
    segments_use = segments if segments is not None else sorted(seg_mask_map.keys()) or ["full"]
    
    n_epochs = int(precomputed.data.shape[0])
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
    
    if logger is not None:
        logger.info(
            "Directed connectivity extraction: epochs=%d, channels=%d, bands=%d, "
            "segments=%d, methods=%s",
            n_epochs, n_channels, len(bands_use), len(segments_use), methods
        )
    
    t0 = time.perf_counter()
    
    for seg_name in segments_use:
        seg_mask = seg_mask_map.get(seg_name)
        if seg_mask is None and seg_name == "full":
            seg_data = precomputed.data
        elif seg_mask is not None and np.any(seg_mask):
            seg_data = precomputed.data[:, :, seg_mask]
        else:
            continue
        
        if seg_data.shape[-1] < min_segment_samples:
            continue
        
        for band in bands_use:
            if band not in freq_bands:
                continue
            
            fmin, fmax = freq_bands[band]
            try:
                fmin = float(fmin)
                fmax = float(fmax)
            except (TypeError, ValueError):
                continue
            
            if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
                continue

            n_times_seg = int(seg_data.shape[-1])
            max_stable_order = int(
                max(1, n_times_seg // max(1, min_samples_per_mvar_parameter * n_channels))
            )
            mvar_order_eff = int(max(1, min(mvar_order, max_stable_order)))
            if logger is not None and mvar_order_eff < mvar_order:
                logger.warning(
                    "Directed connectivity: reducing MVAR order from %d to %d for segment='%s', band='%s' "
                    "(n_times=%d, n_channels=%d, min_samples_per_mvar_parameter=%d).",
                    mvar_order,
                    mvar_order_eff,
                    seg_name,
                    band,
                    n_times_seg,
                    n_channels,
                    min_samples_per_mvar_parameter,
                )
            
            for ep_idx in range(n_epochs):
                epoch_data = seg_data[ep_idx]
                
                results = _compute_directed_connectivity_epoch(
                    ep_idx,
                    epoch_data,
                    sfreq,
                    fmin,
                    fmax,
                    n_freqs,
                    mvar_order_eff,
                    methods,
                )
                
                for method, conn_matrix in results.items():
                    if conn_matrix is None or not np.isfinite(conn_matrix).any():
                        continue
                    
                    if output_level == "full":
                        for idx, (i, j) in enumerate(zip(pair_i, pair_j)):
                            col_fwd = NamingSchema.build(
                                "dconn", seg_name, band, "chpair",
                                f"{method}_fwd", channel_pair=pair_names[idx]
                            )
                            col_bwd = NamingSchema.build(
                                "dconn", seg_name, band, "chpair",
                                f"{method}_bwd", channel_pair=pair_names[idx]
                            )
                            records[ep_idx][col_fwd] = float(conn_matrix[i, j])
                            records[ep_idx][col_bwd] = float(conn_matrix[j, i])
                    
                    upper_vals = conn_matrix[pair_i, pair_j]
                    lower_vals = conn_matrix[pair_j, pair_i]
                    
                    col_mean_fwd = NamingSchema.build(
                        "dconn", seg_name, band, "global", f"{method}_fwd_mean"
                    )
                    col_mean_bwd = NamingSchema.build(
                        "dconn", seg_name, band, "global", f"{method}_bwd_mean"
                    )
                    col_asymmetry = NamingSchema.build(
                        "dconn", seg_name, band, "global", f"{method}_asymmetry"
                    )
                    
                    records[ep_idx][col_mean_fwd] = float(np.nanmean(upper_vals))
                    records[ep_idx][col_mean_bwd] = float(np.nanmean(lower_vals))
                    
                    asymmetry = np.nanmean(upper_vals) - np.nanmean(lower_vals)
                    records[ep_idx][col_asymmetry] = float(asymmetry)
    
    if logger is not None:
        logger.info(
            "Directed connectivity extraction completed in %.2fs",
            time.perf_counter() - t0
        )
    
    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    
    df = pd.DataFrame(records)
    return df, list(df.columns)


def extract_directed_connectivity_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract directed connectivity features from FeatureContext.
    
    This is the main entry point for the features pipeline.
    
    Parameters
    ----------
    ctx : FeatureContext
        Feature extraction context
    bands : List[str]
        Frequency bands to process
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with directed connectivity features
    columns : List[str]
        List of feature column names
    """
    if not bands:
        return pd.DataFrame(), []
    expected_transform = _get_spatial_transform_type(
        ctx.config, feature_family="directedconnectivity"
    )
    
    precomputed = None
    getter = getattr(ctx, "get_precomputed_for_family", None)
    if callable(getter):
        precomputed = getter("directedconnectivity")
        if precomputed is None:
            precomputed = getter("connectivity")
    if precomputed is None:
        precomputed = getattr(ctx, "precomputed", None)
    if precomputed is not None:
        compatible, reason = _is_connectivity_precomputed_compatible(
            precomputed,
            expected_transform=expected_transform,
            bands=bands,
        )
        if not compatible:
            if ctx.logger is not None:
                ctx.logger.warning(
                    "Directed connectivity: existing precomputed intermediates are incompatible (%s); "
                    "recomputing directed-connectivity-specific precomputed data.",
                    reason,
                )
            precomputed = None
    if precomputed is None:
        if getattr(ctx, "epochs", None) is None:
            return pd.DataFrame(), []
        
        if not getattr(ctx.epochs, "preload", False):
            ctx.logger.info("Preloading epochs data...")
            ctx.epochs.load_data()
        
        precomputed = precompute_data(
            ctx.epochs,
            bands,
            ctx.config,
            ctx.logger,
            windows_spec=ctx.windows,
            feature_family="directedconnectivity",
            train_mask=getattr(ctx, "train_mask", None),
            analysis_mode=getattr(ctx, "analysis_mode", None),
        )
        setter = getattr(ctx, "set_precomputed_for_family", None)
        if callable(setter):
            setter("directedconnectivity", precomputed)
            setter("connectivity", precomputed)
        else:
            ctx.set_precomputed(precomputed)
    
    ctx_name = getattr(ctx, "name", None)
    segments: List[str] = []
    if ctx_name:
        segments = [ctx_name]
    elif getattr(ctx, "windows", None) is not None:
        for key in ("active", "plateau"):
            mask = ctx.windows.get_mask(key)
            if mask is not None and np.any(mask):
                segments = [key]
                break
        if not segments:
            mask_names = [k for k in ctx.windows.masks.keys() if k != "baseline"]
            if mask_names:
                segments = [mask_names[0]]
    if not segments:
        segments = ["full"]
    
    df, cols = extract_directed_connectivity_from_precomputed(
        precomputed,
        bands=bands,
        segments=segments,
        config=ctx.config,
        logger=ctx.logger,
    )
    
    return df, cols


###################################################################
# PUBLIC API
###################################################################


__all__ = [
    # Undirected connectivity
    "extract_connectivity_features",
    "extract_connectivity_from_precomputed",
    # Directed connectivity
    "extract_directed_connectivity_features",
    "extract_directed_connectivity_from_precomputed",
]
