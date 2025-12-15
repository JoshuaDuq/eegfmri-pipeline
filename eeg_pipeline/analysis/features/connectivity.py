"""
Connectivity Feature Extraction
================================

Computes functional connectivity features from EEG data:
- Phase-based: wPLI, PLI, imCoh
- Amplitude-based: AEC, AEC-orth (orthogonalized)
- Graph metrics: clustering, efficiency, participation, small-world

All measures are computed per frequency band and trial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging
import time
import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert
import networkx as nx
from joblib import Parallel, delayed

try:
    from mne_connectivity import envelope_correlation, spectral_connectivity_time
except Exception:
    envelope_correlation = None
    spectral_connectivity_time = None

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.utils.config.loader import get_frequency_bands, get_fisher_z_clip_values
from eeg_pipeline.utils.data.loading import flatten_lower_triangles
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.analysis.graph_metrics import (
    symmetrize_adjacency as _symmetrize_and_clip,
    compute_global_efficiency_weighted as _global_efficiency_weighted,
    compute_small_world_sigma,
    compute_legacy_graph_summaries,
    threshold_adjacency as _threshold_adjacency,
)

# --- Helpers ---

def _load_schaefer_rsn_lookup() -> Dict[str, str]:
    return {} 

def _infer_community_map(labels: np.ndarray) -> Dict[str, str]:
    return {}

def _compute_wpli_epoch(epoch_data: np.ndarray) -> np.ndarray:
    """Compute wPLI matrix for a single epoch."""
    # epoch_data: (n_ch, n_times) - complex analytic signal
    cross = epoch_data[:, None, :] * np.conj(epoch_data[None, :, :])
    imag_cross = np.imag(cross)
    denom = np.mean(np.abs(imag_cross), axis=-1)
    numer = np.abs(np.mean(imag_cross, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        wpli = np.where(denom > 0, numer / denom, 0.0)
    wpli = 0.5 * (wpli + wpli.T)
    np.fill_diagonal(wpli, 0.0)
    return wpli

def _compute_wpli_matrices(analytic: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """Compute wPLI matrices for all epochs in parallel."""
    n_epochs = analytic.shape[0]
    mats = Parallel(n_jobs=n_jobs)(
        delayed(_compute_wpli_epoch)(analytic[ep]) for ep in range(n_epochs)
    )
    return np.array(mats)

def _compute_aec_orth_epoch(data: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Compute AEC orthogonalized matrix for a single epoch."""
    n_channels = data.shape[0]
    ep_aec = np.eye(n_channels, dtype=float)
    
    for i in range(n_channels):
        xi = data[i]
        for j in range(i + 1, n_channels):
            xj = data[j]
            # Orthogonalize xi with respect to xj
            xj_norm_sq = np.sum(np.abs(xj) ** 2) + epsilon
            beta_ij = np.sum(xi * np.conj(xj)) / xj_norm_sq
            xi_orth = xi - beta_ij * xj
            
            # And vice versa
            xi_norm_sq = np.sum(np.abs(xi) ** 2) + epsilon
            beta_ji = np.sum(xj * np.conj(xi)) / xi_norm_sq
            xj_orth = xj - beta_ji * xi
            
            env_i = np.abs(xi_orth)
            env_j = np.abs(xj_orth)
            std_i = env_i.std()
            std_j = env_j.std()
            
            if std_i < epsilon or std_j < epsilon:
                r = np.nan
            else:
                env_i = (env_i - env_i.mean()) / std_i
                env_j = (env_j - env_j.mean()) / std_j
                r = np.corrcoef(env_i, env_j)[0, 1]
            ep_aec[i, j] = ep_aec[j, i] = r
            
    return ep_aec

def _compute_aec_orth_matrices(analytic: np.ndarray, epsilon: float = 1e-12, n_jobs: int = 1) -> np.ndarray:
    """Compute AEC matrices for all epochs in parallel."""
    n_epochs = analytic.shape[0]
    mats = Parallel(n_jobs=n_jobs)(
        delayed(_compute_aec_orth_epoch)(analytic[ep], epsilon) for ep in range(n_epochs)
    )
    return np.array(mats)

def _bandpass_hilbert_trials(data, sfreq, fmin, fmax, logger, n_jobs=1):
    try:
        # Simple filter -> hilbert
        # Using MNE filter with n_jobs
        flat_data = data.reshape(-1, data.shape[-1])
        filtered = mne.filter.filter_data(
            flat_data, 
            sfreq, 
            l_freq=fmin, 
            h_freq=fmax, 
            verbose=False,
            n_jobs=n_jobs
        )
        analytic = hilbert(filtered, axis=-1).reshape(data.shape)
        return analytic
    except Exception as e:
        logger.error(f"Hilbert failed: {e}")
        return None

# --- Main Extraction Feature ---

def extract_connectivity_features(
    ctx: Any, # FeatureContext
    bands: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    if not bands:
        return pd.DataFrame(), []

    if not ctx.ensure_precomputed():
        return pd.DataFrame(), []

    precomputed = ctx.precomputed
    if precomputed is None:
        return pd.DataFrame(), []

    conn_df, conn_cols = extract_connectivity_from_precomputed(
        precomputed,
        bands=bands,
        config=ctx.config,
        logger=ctx.logger,
        segments=["baseline", "ramp", "plateau"],
    )
    return conn_df, conn_cols


# =============================================================================
# Precomputed Data Extractors (Moved from pipeline.py)
# =============================================================================

def _graph_metrics(adj: np.ndarray, measure: str, band: str, conn_cfg: Dict[str, Any]) -> Dict[str, float]:
    adj = np.asarray(adj, dtype=float)
    adj[~np.isfinite(adj)] = 0.0
    np.fill_diagonal(adj, 0.0)

    top_prop = conn_cfg.get("graph_top_prop", 0.1)
    try:
        top_prop = float(top_prop)
    except Exception:
        top_prop = 0.1
    if not np.isfinite(top_prop) or top_prop <= 0 or top_prop > 1:
        top_prop = 0.1

    small_world_n_rand = conn_cfg.get("small_world_n_rand", 100)
    try:
        small_world_n_rand = int(small_world_n_rand)
    except Exception:
        small_world_n_rand = 100
    small_world_n_rand = max(5, small_world_n_rand)

    adj_sym = _symmetrize_and_clip(adj)
    adj_abs = np.abs(adj_sym)
    adj_bin = _threshold_adjacency(adj_abs, top_proportion=top_prop)

    geff = _global_efficiency_weighted(adj_abs)

    try:
        clust_vals = nx.clustering(nx.from_numpy_array(adj_bin)).values()
        clust = float(np.mean(list(clust_vals))) if clust_vals else np.nan
    except Exception:
        clust = np.nan

    try:
        smallworld = float(compute_small_world_sigma(adj_bin, n_rand=small_world_n_rand))
    except Exception:
        smallworld = np.nan

    return {
        f"{measure}_{band}_geff": geff,
        f"{measure}_{band}_clust": clust,
        f"{measure}_{band}_pc": np.nan,
        f"{measure}_{band}_smallworld": smallworld,
    }

def _mask_array(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return arr
    if isinstance(mask, np.ndarray) and np.any(mask):
        return arr[:, mask]
    return arr

def extract_connectivity_from_precomputed(
    precomputed: Any, # PrecomputedData
    *,
    bands: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    config: Any = None,
    logger: Any = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute connectivity measures from precomputed analytic signals."""
    if not precomputed.band_data:
        return pd.DataFrame(), []

    config = config or getattr(precomputed, "config", None) or {}
    logger = logger or getattr(precomputed, "logger", None)

    bands_use = list(precomputed.band_data.keys()) if bands is None else [b for b in bands if b in precomputed.band_data]
    if not bands_use:
        return pd.DataFrame(), []

    conn_cfg = config.get("feature_engineering.connectivity", {})

    allow_legacy_fallback = bool(conn_cfg.get("allow_legacy_fallback", False))
    output_level = str(conn_cfg.get("output_level", "full")).strip().lower()
    if output_level not in {"full", "global_only"}:
        output_level = "full"

    enable_wpli = bool(conn_cfg.get("enable_wpli", True))
    enable_aec = bool(conn_cfg.get("enable_aec", True))
    enable_pli_bundle = bool(conn_cfg.get("enable_pli", False))
    enable_plv = bool(conn_cfg.get("enable_plv", enable_pli_bundle))
    enable_pli = bool(conn_cfg.get("enable_pli", False))
    enable_imcoh = bool(conn_cfg.get("enable_imcoh", False))
    enable_graph_metrics = bool(conn_cfg.get("enable_graph_metrics", True))

    phase_measures: List[str] = []
    if enable_wpli:
        phase_measures.append("wpli")
    if enable_plv:
        phase_measures.append("plv")
    if enable_pli:
        phase_measures.append("pli")

    if enable_imcoh and logger is not None:
        logger.warning(
            "Connectivity: enable_imcoh=True is not supported in the trial-wise mne-connectivity backend; skipping. "
            "Consider using ciplv or a connectivity-over-epochs method if you need an imcoh-like metric."
        )

    segments_use = segments if segments is not None else ["baseline", "ramp", "plateau"]
    masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    seg_mask_map = {
        "baseline": masks.get("baseline"),
        "ramp": masks.get("ramp"),
        "plateau": masks.get("plateau"),
    }

    ch_names = list(getattr(precomputed, "ch_names", []))
    n_channels = len(ch_names)
    if n_channels < 2:
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

    n_freqs_per_band = int(conn_cfg.get("n_freqs_per_band", 8))
    conn_mode = str(conn_cfg.get("mode", "cwt_morlet"))
    n_cycles = conn_cfg.get("n_cycles", None)
    try:
        n_cycles = float(n_cycles) if n_cycles is not None else None
    except Exception:
        n_cycles = None
    decim = int(conn_cfg.get("decim", 1))
    min_segment_samples = int(conn_cfg.get("min_segment_samples", 50))

    try:
        sfreq = float(getattr(precomputed, "sfreq", None))
    except Exception as exc:
        raise ValueError("Connectivity extraction requires a valid precomputed.sfreq (sampling frequency).") from exc
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Connectivity extraction requires a valid precomputed.sfreq (sampling frequency).")

    def _safe_n_cycles_for_segment(
        base_n_cycles: Optional[float],
        freqs_hz: np.ndarray,
        sfreq_hz: float,
        n_times: int,
    ) -> np.ndarray:
        if n_times <= 0:
            return np.ones_like(freqs_hz, dtype=float)

        default_cycles = 7.0
        try:
            base = float(base_n_cycles) if base_n_cycles is not None else default_cycles
        except Exception:
            base = default_cycles

        freqs_hz = np.asarray(freqs_hz, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            max_cycles = 0.5 * (float(n_times) * freqs_hz / float(sfreq_hz))
        max_cycles = np.where(np.isfinite(max_cycles), max_cycles, 1.0)
        max_cycles = np.maximum(max_cycles, 1.0)
        return np.minimum(base, max_cycles)

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

    if (spectral_connectivity_time is None) or (envelope_correlation is None):
        if not allow_legacy_fallback:
            raise ImportError(
                "Connectivity extraction requires 'mne-connectivity' for scientifically vetted results. "
                "Install mne-connectivity or explicitly set feature_engineering.connectivity.allow_legacy_fallback=true "
                "to enable the legacy fallback (not recommended)."
            )
        if logger is not None:
            logger.warning(
                "mne-connectivity is not available; falling back to legacy connectivity extraction because "
                "feature_engineering.connectivity.allow_legacy_fallback=true. Results may not be comparable to the "
                "mne-connectivity backend."
            )
        records_legacy: List[Dict[str, float]] = []
        for ep_idx in range(n_epochs):
            record: Dict[str, float] = {}
            for band in bands_use:
                bd = precomputed.band_data[band]
                analytic = bd.analytic[ep_idx]
                phase = bd.phase[ep_idx]
                envelope = bd.envelope[ep_idx]

                for seg_name in segments_use:
                    seg_mask = seg_mask_map.get(seg_name)
                    if seg_mask is None or (isinstance(seg_mask, np.ndarray) and not np.any(seg_mask)):
                        continue

                    analytic_seg = _mask_array(analytic, seg_mask)
                    if analytic_seg.ndim != 2 or analytic_seg.shape[1] < 5:
                        continue

                    mats: Dict[str, np.ndarray] = {}
                    if enable_wpli:
                        mats["wpli"] = _compute_wpli_epoch(analytic_seg)
                    if enable_aec:
                        mats["aec"] = _compute_aec_orth_epoch(analytic_seg)

                    if enable_plv or enable_pli:
                        phase_seg = _mask_array(phase, seg_mask)
                        if phase_seg.ndim == 2 and phase_seg.shape[1] >= 2:
                            if enable_plv:
                                plv_mat = np.zeros((n_channels, n_channels), dtype=float)
                                for i in range(n_channels):
                                    for j in range(i + 1, n_channels):
                                        diff = phase_seg[i] - phase_seg[j]
                                        plv = float(np.abs(np.mean(np.exp(1j * diff))))
                                        plv_mat[i, j] = plv
                                        plv_mat[j, i] = plv
                                mats["plv"] = plv_mat
                            if enable_pli:
                                pli_mat = np.zeros((n_channels, n_channels), dtype=float)
                                for i in range(n_channels):
                                    for j in range(i + 1, n_channels):
                                        diff = phase_seg[i] - phase_seg[j]
                                        pli = float(np.abs(np.mean(np.sign(np.sin(diff)))))
                                        pli_mat[i, j] = pli
                                        pli_mat[j, i] = pli
                                mats["pli"] = pli_mat

                    for measure_name, mat in mats.items():
                        if mat.size == 0:
                            continue

                        if output_level == "full":
                            for i in range(n_channels):
                                for j in range(i + 1, n_channels):
                                    pair_name = f"{ch_names[i]}-{ch_names[j]}"
                                    col = NamingSchema.build(
                                        "conn_legacy",
                                        seg_name,
                                        band,
                                        "chpair",
                                        measure_name,
                                        channel_pair=pair_name,
                                    )
                                    record[col] = float(mat[i, j])

                        tmp = np.asarray(mat, dtype=float).copy()
                        np.fill_diagonal(tmp, np.nan)
                        col_glob = NamingSchema.build(
                            "conn_legacy",
                            seg_name,
                            band,
                            "global",
                            f"{measure_name}_mean",
                        )
                        record[col_glob] = float(np.nanmean(tmp))

                        if enable_graph_metrics and seg_name == "plateau":
                            legacy_graph = _graph_metrics(tmp, measure_name, band, conn_cfg)
                            record.update({f"legacy_{k}": v for k, v in legacy_graph.items()})

            records_legacy.append(record)

        df = pd.DataFrame(records_legacy)
        return df, list(df.columns)

    freq_bands = get_frequency_bands(config)

    use_task_parallel = bool(n_jobs > 1) and (not enable_graph_metrics)
    inner_n_jobs = 1 if use_task_parallel else n_jobs

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

    def _phase_task(seg_name: str, band: str, method: str, seg_data: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float, use_n_cycles: Any) -> pd.DataFrame:
        t0 = time.perf_counter()
        con = spectral_connectivity_time(
            seg_data,
            freqs=freqs,
            method=method,
            indices=indices,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            average=False,
            faverage=True,
            mode=conn_mode,
            n_cycles=use_n_cycles,
            decim=decim,
            n_jobs=inner_n_jobs,
            verbose=False,
        )

        con_data = np.asarray(con.get_data())
        if con_data.ndim == 2:
            con_data = con_data[None, :, :]
        if con_data.ndim == 3 and con_data.shape[-1] > 1:
            con_vals = np.nanmean(con_data, axis=-1)
        elif con_data.ndim == 3:
            con_vals = con_data[:, :, 0]
        else:
            return pd.DataFrame()
        if con_vals.shape[0] != n_epochs:
            return pd.DataFrame()

        parts: List[pd.DataFrame] = []
        if output_level == "full":
            prefix = f"conn_{seg_name}_{band}_chpair_"
            suffix = f"_{method}"
            cols = [f"{prefix}{pair_name}{suffix}" for pair_name in pair_names]
            parts.append(pd.DataFrame(con_vals, columns=cols))

        glob_col = f"conn_{seg_name}_{band}_global_{method}_mean"
        parts.append(pd.DataFrame({glob_col: np.nanmean(con_vals, axis=1)}))

        if enable_graph_metrics and seg_name == "plateau":
            if logger is not None:
                logger.info(
                    "Connectivity graph metrics (phase): seg=%s band=%s method=%s (epochs=%d, channels=%d, small_world_n_rand=%s)",
                    seg_name,
                    band,
                    method,
                    int(n_epochs),
                    int(n_channels),
                    str(conn_cfg.get("small_world_n_rand", 100)),
                )
            def _graph_row(ep_idx: int) -> Dict[str, float]:
                adj = np.zeros((n_channels, n_channels), dtype=float)
                adj[pair_i, pair_j] = con_vals[ep_idx]
                adj[pair_j, pair_i] = con_vals[ep_idx]
                return _graph_metrics(adj, method, band, conn_cfg)

            graph_rows = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_graph_row)(ep_idx) for ep_idx in range(n_epochs)
            )
            parts.append(pd.DataFrame(graph_rows))

        df_out = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        if logger is not None:
            logger.debug("Connectivity task phase/%s/%s/%s finished in %.2fs (cols=%d)", seg_name, band, method, time.perf_counter() - t0, int(df_out.shape[1]))
        return df_out

    def _aec_task(seg_name: str, band: str, analytic_seg: np.ndarray) -> pd.DataFrame:
        t0 = time.perf_counter()
        ec = envelope_correlation(
            analytic_seg,
            orthogonalize="pairwise",
            log=False,
            absolute=True,
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

        parts: List[pd.DataFrame] = []
        if output_level == "full":
            prefix = f"conn_{seg_name}_{band}_chpair_"
            suffix = "_aec"
            cols = [f"{prefix}{pair_name}{suffix}" for pair_name in pair_names]
            parts.append(pd.DataFrame(aec_vals, columns=cols))

        glob_col = f"conn_{seg_name}_{band}_global_aec_mean"
        parts.append(pd.DataFrame({glob_col: np.nanmean(aec_vals, axis=1)}))

        if enable_graph_metrics and seg_name == "plateau":
            if logger is not None:
                logger.info(
                    "Connectivity graph metrics (aec): seg=%s band=%s (epochs=%d, channels=%d, small_world_n_rand=%s)",
                    seg_name,
                    band,
                    int(n_epochs),
                    int(n_channels),
                    str(conn_cfg.get("small_world_n_rand", 100)),
                )
            def _graph_row(ep_idx: int) -> Dict[str, float]:
                adj = np.zeros((n_channels, n_channels), dtype=float)
                adj[pair_i, pair_j] = aec_vals[ep_idx]
                adj[pair_j, pair_i] = aec_vals[ep_idx]
                return _graph_metrics(adj, "aec", band, conn_cfg)

            graph_rows = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_graph_row)(ep_idx) for ep_idx in range(n_epochs)
            )
            parts.append(pd.DataFrame(graph_rows))

        return pd.concat(parts, axis=1) if parts else pd.DataFrame()

    tasks: List[Tuple[str, Tuple[Any, ...]]] = []
    for seg_name in segments_use:
        seg_mask = seg_mask_map.get(seg_name)
        seg_data = _slice_epochs(precomputed.data, seg_mask)
        if seg_data is None:
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
            except Exception:
                continue
            if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
                continue

            freqs = np.linspace(fmin, fmax, max(n_freqs_per_band, 2))
            use_n_cycles = n_cycles
            if conn_mode == "cwt_morlet":
                use_n_cycles = _safe_n_cycles_for_segment(n_cycles, freqs, sfreq, int(seg_data.shape[-1]))

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

    if not tasks:
        return pd.DataFrame(), []

    def _run_task(task: Tuple[str, Tuple[Any, ...]]) -> pd.DataFrame:
        kind, args = task
        if kind == "phase":
            return _phase_task(*args)
        return _aec_task(*args)

    if logger is not None:
        logger.info("Running connectivity tasks: n_tasks=%d (threaded=%s)", int(len(tasks)), str(bool(use_task_parallel and len(tasks) > 1)))

    task_times: Dict[str, float] = {"phase": 0.0, "aec": 0.0}
    task_counts: Dict[str, int] = {"phase": 0, "aec": 0}

    def _timed_run_task(task: Tuple[str, Tuple[Any, ...]]) -> pd.DataFrame:
        kind, _ = task
        t0 = time.perf_counter()
        df_task = _run_task(task)
        dt = time.perf_counter() - t0
        if kind in task_times:
            task_times[kind] += dt
            task_counts[kind] += 1
        return df_task

    if use_task_parallel and len(tasks) > 1:
        dfs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_timed_run_task)(task) for task in tasks
        )
    else:
        dfs = [_timed_run_task(task) for task in tasks]

    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame(), []

    t_concat0 = time.perf_counter()
    df = pd.concat(dfs, axis=1)
    if logger is not None:
        logger.info(
            "Connectivity post-processing: task_time_phase=%.2fs (%d), task_time_aec=%.2fs (%d), concat=%.2fs, total=%.2fs, out_shape=(%d,%d)",
            float(task_times["phase"]),
            int(task_counts["phase"]),
            float(task_times["aec"]),
            int(task_counts["aec"]),
            time.perf_counter() - t_concat0,
            time.perf_counter() - t_total0,
            int(df.shape[0]),
            int(df.shape[1]),
        )

    return df, list(df.columns)
