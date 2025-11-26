from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert
import networkx as nx

from eeg_pipeline.utils.analysis.windowing import sliding_window_centers
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.data.loading import flatten_lower_triangles


###################################################################
# Connectivity Feature Extraction
###################################################################


def _symmetrize_and_clip(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square; got shape {adj.shape}")
    adj = 0.5 * (adj + adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


@lru_cache(maxsize=1)
def _load_schaefer_rsn_lookup() -> Dict[str, str]:
    """
    Map ROI Name -> RSN label using the Schaefer 2018 100-parcel, 7-network CSV.
    Returns empty dict if file not found.
    """
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    lookup = {}
    for _, row in df.iterrows():
        name = str(row.get("ROI Name", "")).strip()
        tokens = name.split("_")
        if len(tokens) >= 3:
            network = tokens[2]
            lookup[name] = network
    return lookup


def _infer_community_map(labels: np.ndarray) -> Dict[str, str]:
    """
    Attempt to map node labels to RSNs using Schaefer ROI names.
    If mapping is sparse, returns empty dict to avoid misleading participation metrics.
    """
    lookup = _load_schaefer_rsn_lookup()
    if not lookup:
        return {}
    mapping: Dict[str, str] = {}
    for lbl in labels:
        lbl_str = str(lbl)
        if lbl_str in lookup:
            mapping[lbl_str] = lookup[lbl_str]
            continue
        for key, net in lookup.items():
            if key.endswith(lbl_str):
                mapping[lbl_str] = net
                break
    if len(set(mapping.values())) < 2:
        return {}
    return mapping


def _participation_coeff(adj: np.ndarray, labels: np.ndarray, community_map: Dict[str, str]) -> np.ndarray:
    if not community_map:
        return np.full(adj.shape[0], np.nan, dtype=float)
    comms = [community_map.get(str(l), None) for l in labels]
    unique_comms = [c for c in sorted(set(comms)) if c is not None]
    if len(unique_comms) < 2:
        return np.full(adj.shape[0], np.nan, dtype=float)

    adj = np.maximum(adj, 0.0)
    deg = adj.sum(axis=1)
    pc = np.full(adj.shape[0], np.nan, dtype=float)
    for i in range(adj.shape[0]):
        k_i = deg[i]
        if k_i <= 0:
            continue
        accum = 0.0
        for comm in unique_comms:
            idx = [j for j, c in enumerate(comms) if c == comm]
            if not idx:
                continue
            k_ic = np.sum(adj[i, idx])
            accum += (k_ic / k_i) ** 2
        pc[i] = 1.0 - accum
    return pc


def _global_efficiency_weighted(adj: np.ndarray, eps: float = 1e-9) -> float:
    G = nx.from_numpy_array(adj)
    lengths = {}
    for u, v, data in G.edges(data=True):
        w = abs(data.get("weight", 0.0))
        lengths[(u, v)] = 1.0 / (w + eps)
    nx.set_edge_attributes(G, lengths, "length")
    try:
        return float(nx.global_efficiency(G, weight="length"))
    except TypeError:
        # networkx versions without weighted global_efficiency support
        try:
            sp_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
            n = G.number_of_nodes()
            if n <= 1:
                return np.nan
            inv_dist = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = sp_lengths.get(i, {}).get(j, np.inf)
                    if np.isfinite(d) and d > 0:
                        inv_dist.append(1.0 / d)
            return float((2.0 / (n * (n - 1))) * np.sum(inv_dist)) if inv_dist else np.nan
        except Exception:
            return np.nan
    except ZeroDivisionError:
        return np.nan


def _small_world_sigma(adj_bin: np.ndarray, n_rand: int = 100) -> float:
    G = nx.from_numpy_array(adj_bin)
    if nx.number_of_nodes(G) < 3 or nx.number_of_edges(G) == 0:
        return np.nan
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        C = nx.average_clustering(G)
        L = nx.average_shortest_path_length(G)
    except Exception:
        return np.nan
    if C == 0 or L == 0:
        return np.nan

    n = G.number_of_nodes()
    p = nx.density(G)
    C_rand_list = []
    L_rand_list = []
    for _ in range(n_rand):
        Gr = nx.gnp_random_graph(n, p)
        if nx.number_of_edges(Gr) == 0:
            continue
        if not nx.is_connected(Gr):
            Gr = Gr.subgraph(max(nx.connected_components(Gr), key=len)).copy()
        try:
            C_rand_list.append(nx.average_clustering(Gr))
            L_rand_list.append(nx.average_shortest_path_length(Gr))
        except Exception:
            continue
    if not C_rand_list or not L_rand_list:
        return np.nan
    C_rand = float(np.mean(C_rand_list))
    L_rand = float(np.mean(L_rand_list))
    if C_rand == 0 or L_rand == 0:
        return np.nan
    return (C / C_rand) / (L / L_rand)


def _threshold_adjacency(
    adj: np.ndarray,
    proportional_keep: float,
    min_abs_weight: float,
    drop_negative: bool,
    use_abs_for_threshold: bool,
) -> np.ndarray:
    """
    Apply proportional and absolute-value thresholding to an adjacency matrix.
    """
    adj = _symmetrize_and_clip(adj)
    if drop_negative:
        adj = np.where(adj > 0, adj, 0.0)

    weight_basis = np.abs(adj) if use_abs_for_threshold else adj
    if min_abs_weight > 0:
        adj = np.where(weight_basis >= min_abs_weight, adj, 0.0)
        weight_basis = np.where(weight_basis >= min_abs_weight, weight_basis, 0.0)

    lower = weight_basis[np.tril_indices_from(weight_basis, k=-1)]
    lower = lower[np.isfinite(lower) & (lower > 0)]
    if lower.size == 0 or proportional_keep <= 0:
        np.fill_diagonal(adj, 0.0)
        return adj

    keep = max(1, int(np.ceil(proportional_keep * lower.size)))
    thresh = np.partition(lower, -keep)[-keep]
    adj = np.where(weight_basis >= thresh, adj, 0.0)
    np.fill_diagonal(adj, 0.0)
    return adj


def _compute_graph_metrics_block(
    mats: np.ndarray,
    labels: np.ndarray,
    measure: str,
    band: str,
    community_map: Dict[str, str],
    binary_threshold: float = 0.0,
    proportional_keep: float = 0.1,
    drop_negative: bool = False,
    use_abs_for_threshold: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    n_epochs, n_nodes, _ = mats.shape
    records: List[Dict[str, float]] = []
    rsn_sets = sorted(set(community_map.values())) if community_map else []

    for epoch_idx in range(n_epochs):
        adj = _threshold_adjacency(
            mats[epoch_idx],
            proportional_keep=proportional_keep,
            min_abs_weight=binary_threshold,
            drop_negative=drop_negative,
            use_abs_for_threshold=use_abs_for_threshold,
        )
        adj_abs = np.abs(adj)
        if not np.isfinite(adj_abs).any():
            records.append({f"{measure}_{band}_geff": np.nan})
            continue
        record: Dict[str, float] = {}

        record[f"{measure}_{band}_geff"] = _global_efficiency_weighted(adj_abs)

        try:
            clust_vals = nx.clustering(nx.from_numpy_array(adj_abs), weight="weight")
            record[f"{measure}_{band}_clust"] = float(np.mean(list(clust_vals.values())))
        except Exception:
            record[f"{measure}_{band}_clust"] = np.nan

        pc_vals = _participation_coeff(adj_abs, labels, community_map)
        record[f"{measure}_{band}_pc"] = float(np.nanmean(pc_vals)) if np.isfinite(pc_vals).any() else np.nan

        adj_bin = (adj_abs > binary_threshold).astype(float)
        record[f"{measure}_{band}_smallworld"] = _small_world_sigma(adj_bin)

        if rsn_sets:
            strengths = adj_abs.sum(axis=1)
            pc_by_rsn: Dict[str, List[float]] = {r: [] for r in rsn_sets}
            strength_by_rsn: Dict[str, List[float]] = {r: [] for r in rsn_sets}
            for lbl, s_val, pc_val in zip(labels, strengths, pc_vals):
                rsn = community_map.get(str(lbl))
                if rsn is None:
                    continue
                strength_by_rsn[rsn].append(s_val)
                if np.isfinite(pc_val):
                    pc_by_rsn[rsn].append(pc_val)
            for rsn in rsn_sets:
                if strength_by_rsn[rsn]:
                    record[f"{measure}_{band}_rsn_{rsn}_strength"] = float(np.mean(strength_by_rsn[rsn]))
                if pc_by_rsn[rsn]:
                    record[f"{measure}_{band}_rsn_{rsn}_pc"] = float(np.mean(pc_by_rsn[rsn]))

        records.append(record)

    df = pd.DataFrame(records)
    return df, list(df.columns)


def _get_connectivity_time_mask(
    times: np.ndarray, time_window: Tuple[float, float], logger: Any
) -> Optional[np.ndarray]:
    start, end = time_window
    if start >= end:
        logger.error(
            "Connectivity window start (%.3f) must be earlier than end (%.3f).", start, end
        )
        return None

    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        logger.error(
            "No samples found in connectivity window [%.3f, %.3f] s; cannot compute connectivity.",
            start,
            end,
        )
        return None
    return mask


def _bandpass_hilbert_trials(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    logger: Any,
) -> Optional[np.ndarray]:
    try:
        filtered = mne.filter.filter_data(
            data.reshape(-1, data.shape[-1]),
            sfreq,
            l_freq=fmin,
            h_freq=fmax,
            n_jobs=1,
            verbose=False,
        )
        analytic = hilbert(filtered, axis=-1)
    except Exception as exc:  # pragma: no cover - numeric exceptions depend on data
        logger.error(
            "Failed to band-pass/Hilbert filter data for %.1f-%.1f Hz connectivity: %s",
            fmin,
            fmax,
            exc,
        )
        return None
    return analytic.reshape(data.shape)


def _compute_wpli_matrices(analytic: np.ndarray) -> np.ndarray:
    n_epochs, n_channels, n_times = analytic.shape
    if n_times < 2:
        raise ValueError("Cannot compute wPLI with fewer than 2 samples in the window.")

    wpli_mats = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
    for epoch_idx in range(n_epochs):
        epoch_data = analytic[epoch_idx]
        cross = epoch_data[:, None, :] * np.conj(epoch_data[None, :, :])
        imag_cross = np.imag(cross)
        denom = np.mean(np.abs(imag_cross), axis=-1)
        numer = np.abs(np.mean(imag_cross, axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            wpli = np.where(denom > 0, numer / denom, 0.0)
        wpli = 0.5 * (wpli + wpli.T)
        np.fill_diagonal(wpli, 0.0)
        wpli_mats[epoch_idx] = wpli
    return wpli_mats


def _compute_aec_matrices(analytic: np.ndarray, epsilon: float) -> np.ndarray:
    envelopes = np.abs(analytic)
    n_epochs, n_channels, n_times = envelopes.shape
    if n_times < 2:
        raise ValueError("Cannot compute AEC with fewer than 2 samples in the window.")

    aec_mats = np.empty((n_epochs, n_channels, n_channels), dtype=float)
    for epoch_idx in range(n_epochs):
        env = envelopes[epoch_idx]
        env_mean = env.mean(axis=1, keepdims=True)
        env_std = env.std(axis=1, keepdims=True)
        env_std = np.where(env_std < epsilon, epsilon, env_std)
        standardized = (env - env_mean) / env_std
        aec = (standardized @ standardized.T) / float(n_times - 1)
        np.fill_diagonal(aec, 1.0)
        aec_mats[epoch_idx] = aec
    return aec_mats


def _compute_aec_orth_matrices(analytic: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Symmetric pairwise orthogonalization (Brookes-style) before envelope correlation
    to reduce field-spread/zero-lag leakage.
    
    Uses proper regression-based orthogonalization: for each pair (i, j), we regress
    xi onto xj and compute the residual, then vice versa. The envelopes of these
    orthogonalized signals are then correlated.
    
    Reference: Brookes et al. (2012) NeuroImage, Hipp et al. (2012) Nature Neuroscience
    """
    n_epochs, n_channels, n_times = analytic.shape
    if n_times < 2:
        raise ValueError("Cannot compute orthogonalized AEC with fewer than 2 samples.")

    aec_mats = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
    for ep in range(n_epochs):
        ep_aec = np.eye(n_channels, dtype=float)
        data = analytic[ep]
        for i in range(n_channels):
            xi = data[i]
            for j in range(i + 1, n_channels):
                xj = data[j]

                # Proper Brookes-style orthogonalization via regression
                # Orthogonalize xi with respect to xj: xi_orth = xi - proj(xi onto xj)
                xj_norm_sq = np.sum(np.abs(xj) ** 2) + epsilon
                xi_norm_sq = np.sum(np.abs(xi) ** 2) + epsilon
                
                # Projection coefficient (complex regression)
                beta_ij = np.sum(xi * np.conj(xj)) / xj_norm_sq
                beta_ji = np.sum(xj * np.conj(xi)) / xi_norm_sq
                
                # Orthogonalized signals
                xi_orth = xi - beta_ij * xj
                xj_orth = xj - beta_ji * xi

                # Compute envelopes of orthogonalized signals
                env_i = np.abs(xi_orth)
                env_j = np.abs(xj_orth)

                # Standardize envelopes to unit variance to avoid scale bias
                std_i = env_i.std()
                std_j = env_j.std()
                if std_i < epsilon or std_j < epsilon:
                    r = np.nan
                else:
                    env_i = (env_i - env_i.mean()) / std_i
                    env_j = (env_j - env_j.mean()) / std_j
                    try:
                        r = float(np.corrcoef(env_i, env_j)[0, 1])
                    except Exception:
                        r = np.nan
                ep_aec[i, j] = ep_aec[j, i] = r
        aec_mats[ep] = ep_aec
    return aec_mats


def _flatten_connectivity_data(
    arr: np.ndarray,
    labels: Optional[np.ndarray],
    prefix: str,
    logger: Any,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    if arr.ndim != 3:
        logger.warning(f"Unexpected connectivity array shape for {prefix}: {arr.shape}")
        return None

    _, n_i, n_j = arr.shape
    if n_i != n_j:
        logger.error(
            "Connectivity array for %s is not square (shape=%s); skipping flattening.", prefix, arr.shape
        )
        return None
    if labels is not None and len(labels) != n_i:
        logger.error(
            "Connectivity labels length (%d) does not match matrix size (%d) for %s; skipping.",
            len(labels),
            n_i,
            prefix,
        )
        return None

    if not np.allclose(arr, np.transpose(arr, (0, 2, 1)), atol=1e-6, equal_nan=True):
        logger.warning(
            "Connectivity matrix for %s is not symmetric; enforcing symmetry before flattening.", prefix
        )
        arr = 0.5 * (arr + np.transpose(arr, (0, 2, 1)))

    return flatten_lower_triangles(arr, labels, prefix=prefix)


def extract_connectivity_features(
    epochs: mne.Epochs,
    subject: str,
    task: str,
    bands: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for connectivity computation")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    feat_cfg = config.get("feature_engineering.features", {})
    tf_cfg = config.get("time_frequency_analysis", {})
    plateau_default = tf_cfg.get("plateau_window", [epochs.times[0], epochs.times[-1]])
    window_start = float(feat_cfg.get("plateau_start", plateau_default[0]))
    window_end = float(feat_cfg.get("plateau_end", plateau_default[1]))
    time_mask = _get_connectivity_time_mask(np.asarray(epochs.times), (window_start, window_end), logger)
    if time_mask is None:
        return pd.DataFrame(), []

    sfreq = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks)
    labels = np.array([epochs.info["ch_names"][p] for p in picks])
    community_map = _infer_community_map(labels)

    output_dir = deriv_root / f"sub-{subject}" / "eeg"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / f"sub-{subject}_task-{task}_connectivity_labels.npy"
    try:
        np.save(labels_path, labels)
        logger.info("Saved connectivity labels to %s", labels_path)
    except OSError as exc:
        logger.warning("Failed to save connectivity labels to %s: %s", labels_path, exc)

    all_blocks: List[pd.DataFrame] = []
    all_cols: List[str] = []
    epsilon_std = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))
    aec_mode = str(config.get("feature_engineering.connectivity.aec_mode", "orth")).lower()
    enable_aec = aec_mode in {"orth", "standard"}
    graph_top_prop = float(config.get("feature_engineering.connectivity.graph_top_prop", 0.1))

    for band in bands:
        if band not in freq_bands:
            logger.warning(f"Band '{band}' not defined in config; skipping connectivity.")
            continue

        fmin, fmax = freq_bands[band]
        analytic_full = _bandpass_hilbert_trials(data, sfreq, fmin, fmax, logger)
        if analytic_full is None:
            continue

        analytic = analytic_full[..., time_mask]
        try:
            aec_mats = None
            if enable_aec:
                if aec_mode == "orth":
                    aec_mats = _compute_aec_orth_matrices(analytic, epsilon_std)
                elif aec_mode == "standard":
                    aec_mats = _compute_aec_matrices(analytic, epsilon_std)
            wpli_mats = _compute_wpli_matrices(analytic)
        except ValueError as exc:
            logger.error("Skipping connectivity for band %s: %s", band, exc)
            continue

        aec_path = output_dir / f"sub-{subject}_task-{task}_connectivity_aec_{band}_all_trials.npy"
        wpli_path = output_dir / f"sub-{subject}_task-{task}_connectivity_wpli_{band}_all_trials.npy"

        try:
            if aec_mats is not None:
                np.save(aec_path, aec_mats.astype(np.float32))
            np.save(wpli_path, wpli_mats.astype(np.float32))
            logger.info("Saved connectivity arrays for band %s to %s", band, output_dir)
        except OSError as exc:
            logger.warning("Failed to save connectivity arrays for %s: %s", band, exc)

        for measure, arr in (("aec", aec_mats), ("wpli", wpli_mats)):
            if arr is None:
                continue
            result = _flatten_connectivity_data(arr, labels, f"{measure}_{band}", logger)
            if result is None:
                logger.error("Failed to flatten %s connectivity for band %s", measure, band)
                continue
            df_flat, cols = result
            all_blocks.append(df_flat)
            all_cols.extend(cols)

            metrics_df, metrics_cols = _compute_graph_metrics_block(
                arr,
                labels,
                measure=measure,
                band=band,
                community_map=community_map,
                binary_threshold=0.0,
                proportional_keep=graph_top_prop,
                drop_negative=False,
                use_abs_for_threshold=True,
            )
            all_blocks.append(metrics_df)
            all_cols.extend(metrics_cols)

    if not all_blocks:
        logger.warning("Connectivity computation produced no feature blocks.")
        return pd.DataFrame(), []

    block_lengths = {len(df) for df in all_blocks}
    if len(block_lengths) > 1:
        logger.error(
            "Computed connectivity matrices have inconsistent trial counts: %s. "
            "Refusing to concatenate because this would misalign features.",
            ", ".join(str(l) for l in sorted(block_lengths)),
        )
        raise ValueError("Connectivity blocks have mismatched lengths; cannot align reliably.")

    combined_df = pd.concat(all_blocks, axis=1)
    combined_df.columns = all_cols
    return combined_df, all_cols


def compute_sliding_connectivity_features(
    epochs: mne.Epochs,
    config: Any,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, List[str]]:
    """
    Sliding-window connectivity within the plateau window using channel-wise Pearson correlation.
    Returns:
        edges_df: flattened edge features across windows
        edge_cols: column names for edges
        graph_df: node degree per window and modularity per window
        graph_cols: column names for graph metrics
    """
    if epochs is None or len(epochs) == 0:
        return pd.DataFrame(), [], pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for sliding connectivity")
        return pd.DataFrame(), [], pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    times = epochs.times
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    community_map = _infer_community_map(np.array(ch_names))

    plateau_cfg = config.get("feature_engineering.features", {})
    tf_cfg = config.get("time_frequency_analysis", {})
    plateau_default = tf_cfg.get("plateau_window", [times[0], times[-1]])
    plateau_start = float(plateau_cfg.get("plateau_start", plateau_default[0]))
    plateau_end = float(plateau_cfg.get("plateau_end", plateau_default[1]))

    conn_cfg = config.get("feature_engineering.connectivity", {})
    win_len = float(conn_cfg.get("sliding_window_len", 1.0))
    win_step = float(conn_cfg.get("sliding_window_step", 0.5))
    slide_highpass = conn_cfg.get("sliding_highpass_hz", conn_cfg.get("highpass_hz"))
    slide_lowpass = conn_cfg.get("sliding_lowpass_hz", conn_cfg.get("lowpass_hz"))

    # Use shared window centers helper to keep alignment consistent across modules.
    n_windows_est = int(np.floor((plateau_end - plateau_start - win_len) / win_step) + 1) if plateau_end > plateau_start else 0
    centers = sliding_window_centers(config, max(n_windows_est, 0))
    window_starts = centers - (win_len / 2.0)
    if window_starts.size == 0:
        logger.warning("No sliding windows available for connectivity; adjust window length/step")
        return pd.DataFrame(), [], pd.DataFrame(), []

    edge_blocks: List[pd.DataFrame] = []
    graph_blocks: List[pd.DataFrame] = []
    edge_cols_all: List[str] = []
    graph_cols_all: List[str] = []

    corr_threshold = float(config.get("behavior_analysis.statistics.connectivity_min_correlation", 0.3)) if hasattr(config, "get") else 0.3
    graph_top_prop = float(
        config.get(
            "feature_engineering.connectivity.sliding_graph_top_prop",
            config.get("feature_engineering.connectivity.graph_top_prop", 0.1),
        )
    )
    min_samples_per_window = int(
        conn_cfg.get("sliding_min_samples", max(2, 5 * max(1, len(ch_names))))
    )

    # Optional band/high-pass to reduce slow drifts/evoked leakage
    if slide_highpass is not None or slide_lowpass is not None:
        try:
            data = mne.filter.filter_data(
                data.reshape(len(epochs), len(ch_names), -1),
                sfreq=float(epochs.info["sfreq"]),
                l_freq=float(slide_highpass) if slide_highpass is not None else None,
                h_freq=float(slide_lowpass) if slide_lowpass is not None else None,
                n_jobs=1,
                verbose=False,
            ).reshape(data.shape)
            logger.info(
                "Applied sliding-window bandpass: l_freq=%s, h_freq=%s",
                str(slide_highpass),
                str(slide_lowpass),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to bandpass data for sliding connectivity; using raw data: %s", exc)

    for win_idx, win_start in enumerate(window_starts):
        win_end = win_start + win_len
        mask = (times >= win_start) & (times < win_end)
        n_samples_win = int(mask.sum())
        if n_samples_win < min_samples_per_window:
            logger.debug(
                "Skipping sliding window %d: insufficient samples (%d < %d)",
                win_idx,
                n_samples_win,
                min_samples_per_window,
            )
            continue

        corr_mats = []
        for epoch_arr in data:
            seg = epoch_arr[:, mask]
            if seg.shape[1] < 2:
                corr_mats.append(np.full((len(ch_names), len(ch_names)), np.nan))
                continue
            corr_mats.append(np.corrcoef(seg))
        corr_mats = np.stack(corr_mats, axis=0)

        # Fisher z-transform for thresholding; store raw r in outputs
        corr_clipped = np.clip(corr_mats, -0.999999, 0.999999)
        corr_z = np.arctanh(corr_clipped)
        corr_r = np.tanh(corr_z)

        df_edges, cols_edges = flatten_lower_triangles(corr_r, ch_names, prefix=f"sw{win_idx}corr_all")
        edge_blocks.append(df_edges)
        edge_cols_all.extend(cols_edges)

        metrics_df, metrics_cols = _compute_graph_metrics_block(
            corr_r,
            np.array(ch_names),
            measure=f"sw{win_idx}corr_all",
            band="",
            community_map=community_map,
            binary_threshold=corr_threshold,
            proportional_keep=graph_top_prop,
            drop_negative=True,
            use_abs_for_threshold=True,
        )
        graph_blocks.append(metrics_df)
        graph_cols_all.extend(metrics_cols)

        deg_records = []
        mod_records = []
        for corr in corr_r:
            adj = _threshold_adjacency(
                corr,
                proportional_keep=graph_top_prop,
                min_abs_weight=corr_threshold,
                drop_negative=True,
                use_abs_for_threshold=True,
            )
            adj_abs = np.abs(adj)
            deg_records.append(adj_abs.sum(axis=1))

            try:
                G = nx.Graph()
                for i, ch_i in enumerate(ch_names):
                    for j in range(i + 1, len(ch_names)):
                        w = adj_abs[i, j]
                        if np.isfinite(w) and w > 0:
                            G.add_edge(ch_i, ch_names[j], weight=float(w))
                if G.number_of_edges() == 0:
                    mod_records.append(np.nan)
                else:
                    comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
                    mod_records.append(nx.algorithms.community.modularity(G, comms, weight="weight"))
            except Exception:
                mod_records.append(np.nan)

        deg_df = pd.DataFrame(deg_records, columns=[f"sw{win_idx}corr_all_deg_{ch}" for ch in ch_names])
        mod_df = pd.DataFrame({f"sw{win_idx}corr_all_modularity": mod_records})
        graph_blocks.append(pd.concat([deg_df, mod_df], axis=1))
        graph_cols_all.extend(list(deg_df.columns) + list(mod_df.columns))

    if not edge_blocks:
        return pd.DataFrame(), [], pd.DataFrame(), []

    edges_df = pd.concat(edge_blocks, axis=1)
    graph_df = pd.concat(graph_blocks, axis=1) if graph_blocks else pd.DataFrame()
    return edges_df, edge_cols_all, graph_df, graph_cols_all

