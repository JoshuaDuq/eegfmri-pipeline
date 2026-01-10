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

from typing import Optional, List, Dict, Tuple, Any
import time
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

try:
    from mne_connectivity import envelope_correlation, spectral_connectivity_time
except Exception:
    envelope_correlation = None
    spectral_connectivity_time = None

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.analysis.graph_metrics import (
    symmetrize_adjacency as _symmetrize_and_clip,
    compute_global_efficiency_weighted as _global_efficiency_weighted,
    compute_small_world_sigma,
    threshold_adjacency as _threshold_adjacency,
)

from eeg_pipeline.analysis.features.preparation import precompute_data


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
    phase_measures = {"wpli", "wpli2_debiased", "imcoh", "plv", "pli"}
    uses_phase = bool(set(m.lower() for m in measures) & phase_measures)
    
    if not uses_phase:
        return
    
    conn_cfg = config.get("feature_engineering.connectivity", {}) if hasattr(config, "get") else {}
    warn_enabled = bool(conn_cfg.get("warn_if_no_spatial_transform", True))
    if not warn_enabled:
        return
    
    spatial_transform = str(config.get("feature_engineering.spatial_transform", "none")).strip().lower()
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


def _normalize_phase_estimator(value: Any) -> str:
    v = str(value).strip().lower()
    if v in {"across", "across_epochs", "acrossepochs", "across-epochs"}:
        return "across_epochs"
    if v in {"trial", "trialwise", "within_epoch", "within-epoch", "withinepoch"}:
        return "within_epoch"
    return "within_epoch"


def _resolve_phase_measures(conn_cfg: Dict[str, Any]) -> List[str]:
    supported_measures = {"wpli", "wpli2_debiased", "imcoh", "plv", "pli"}
    measures_cfg = conn_cfg.get("measures")
    if isinstance(measures_cfg, (list, tuple)) and measures_cfg:
        measures = {str(m).strip().lower() for m in measures_cfg}
        measures = measures & supported_measures
        return [m for m in ("wpli2_debiased", "wpli", "imcoh", "plv", "pli") if m in measures]
    out: List[str] = []
    if bool(conn_cfg.get("enable_wpli2_debiased", False)):
        out.append("wpli2_debiased")
    if bool(conn_cfg.get("enable_wpli", True)):
        out.append("wpli")
    if bool(conn_cfg.get("enable_imcoh", False)):
        out.append("imcoh")
    if bool(conn_cfg.get("enable_plv", False)):
        out.append("plv")
    if bool(conn_cfg.get("enable_pli", False)):
        out.append("pli")
    return out


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
        if logger is not None:
            logger.warning("Connectivity: mne-connectivity unavailable; cannot compute across-epochs phase estimates.")
        return

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
    try:
        n_cycles = float(n_cycles) if n_cycles is not None else None
    except Exception:
        n_cycles = None
    decim = int(conn_cfg.get("decim", 1))

    min_cycles_per_band = float(conn_cfg.get("min_cycles_per_band", 3.0))
    min_segment_sec = float(conn_cfg.get("min_segment_sec", 0.0))

    try:
        sfreq = float(getattr(precomputed, "sfreq", None))
    except Exception:
        sfreq = np.nan
    if not np.isfinite(sfreq) or sfreq <= 0:
        return

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
        # Prefer across-epochs averaging if supported by installed mne-connectivity.
        try:
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
        except TypeError:
            return spectral_connectivity_time(
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
                except Exception:
                    continue
                if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
                    continue
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
                    try:
                        con = _run(method_use, seg_data, freqs, fmin, fmax, use_n_cycles)
                    except Exception:
                        if method_use == "wpli2_debiased":
                            try:
                                method_use = "wpli"
                                method_label = "wpli"
                                con = _run(method_use, seg_data, freqs, fmin, fmax, use_n_cycles)
                            except Exception:
                                continue
                        else:
                            continue

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


def extract_connectivity_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if not bands:
        return pd.DataFrame(), []

    conn_cfg = ctx.config.get("feature_engineering.connectivity", {}) if hasattr(ctx.config, "get") else {}
    measures_cfg = conn_cfg.get("measures", ["wpli2_debiased", "aec"])
    if isinstance(measures_cfg, (list, tuple)):
        measures = [str(m).strip().lower() for m in measures_cfg]
    else:
        measures = ["wpli2_debiased", "aec"]
    
    _warn_if_phase_connectivity_without_spatial_transform(ctx.config, measures, ctx.logger)

    precomputed = getattr(ctx, "precomputed", None)
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
        )
        try:
            ctx.set_precomputed(precomputed)
        except Exception:
            pass

    # Only process the segment name from the context (fallback to active/full)
    ctx_name = getattr(ctx, "name", None)
    segments: List[str] = []
    if ctx_name:
        segments = [ctx_name]
    elif getattr(ctx, "windows", None) is not None:
        for key in ("active", "active"):
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

    df, cols = extract_connectivity_from_precomputed(
        precomputed,
        bands=bands,
        segments=segments,
        config=ctx.config,
        logger=ctx.logger,
    )
    if df is None or df.empty:
        return pd.DataFrame(), []

    conn_cfg = ctx.config.get("feature_engineering.connectivity", {}) if hasattr(ctx.config, "get") else {}
    granularity = str(conn_cfg.get("granularity", "trial")).strip().lower()
    if granularity not in {"trial", "condition", "subject"}:
        granularity = "trial"

    phase_estimator = _normalize_phase_estimator(conn_cfg.get("phase_estimator", "within_epoch"))
    
    # Guardrail: detect CV/machine learning mode and warn/force within_epoch for phase estimator
    # across_epochs is cross-trial by nature and WILL leak test information in CV
    train_mask = getattr(ctx, "train_mask", None)
    force_within_for_ml = bool(conn_cfg.get("force_within_epoch_for_ml", True))
    
    if train_mask is not None and phase_estimator == "across_epochs":
        if force_within_for_ml:
            if ctx.logger is not None:
                ctx.logger.warning(
                    "Connectivity: train_mask detected (CV/machine learning mode) with phase_estimator='across_epochs'. "
                    "Across-epochs estimates leak test-trial information. "
                    "Forcing phase_estimator='within_epoch' for valid CV. "
                    "Set feature_engineering.connectivity.force_within_epoch_for_ml=false to override."
                )
            phase_estimator = "within_epoch"
        else:
            if ctx.logger is not None:
                ctx.logger.warning(
                    "Connectivity: train_mask detected (CV/machine learning mode) with phase_estimator='across_epochs'. "
                    "CAUTION: Across-epochs estimates leak test-trial information and will inflate machine learning accuracy. "
                    "Consider using phase_estimator='within_epoch' for valid cross-validation."
                )

    if granularity in {"subject", "condition"} and phase_estimator == "across_epochs":
        n_epochs = int(df.shape[0])
        groups_map: Dict[str, np.ndarray] = {"__all__": np.arange(n_epochs, dtype=int)}
        if granularity == "condition":
            events = getattr(ctx, "aligned_events", None)
            if events is not None and not getattr(events, "empty", True) and len(events) == n_epochs:
                cond_col = None
                for candidate in ("condition", "trial_type"):
                    if candidate in events.columns:
                        cond_col = candidate
                        break
                if cond_col is None:
                    candidates = ctx.config.get("event_columns.pain_binary", []) if hasattr(ctx.config, "get") else []
                    if isinstance(candidates, (list, tuple)):
                        for c in candidates:
                            if c in events.columns:
                                cond_col = c
                                break
                if cond_col is not None:
                    labels = events[cond_col].astype(str)
                    min_n = int(conn_cfg.get("min_epochs_per_group", 5))
                    for lab in sorted(labels.unique()):
                        idx = np.where((labels == lab).to_numpy())[0]
                        if idx.size >= min_n:
                            groups_map[f"cond:{lab}"] = idx

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
        return df, cols

    n_epochs = int(df.shape[0])

    if granularity == "subject":
        means = df.apply(pd.to_numeric, errors="coerce").mean(axis=0)
        out = pd.DataFrame(np.tile(means.to_numpy(dtype=float), (n_epochs, 1)), columns=list(means.index))
        return out, list(out.columns)

    # condition-level: broadcast within each condition label
    events = getattr(ctx, "aligned_events", None)
    if events is None or getattr(events, "empty", True) or len(events) != n_epochs:
        ctx.logger.warning("Connectivity granularity=condition requested but aligned_events missing/mismatched; falling back to subject-level.")
        means = df.apply(pd.to_numeric, errors="coerce").mean(axis=0)
        out = pd.DataFrame(np.tile(means.to_numpy(dtype=float), (n_epochs, 1)), columns=list(means.index))
        return out, list(out.columns)

    cond_col = None
    for candidate in ("condition", "trial_type"):
        if candidate in events.columns:
            cond_col = candidate
            break
    if cond_col is None:
        candidates = ctx.config.get("event_columns.pain_binary", []) if hasattr(ctx.config, "get") else []
        if isinstance(candidates, (list, tuple)):
            for c in candidates:
                if c in events.columns:
                    cond_col = c
                    break

    if cond_col is None:
        ctx.logger.warning("Connectivity granularity=condition requested but no condition column found; falling back to subject-level.")
        means = df.apply(pd.to_numeric, errors="coerce").mean(axis=0)
        out = pd.DataFrame(np.tile(means.to_numpy(dtype=float), (n_epochs, 1)), columns=list(means.index))
        return out, list(out.columns)

    labels = events[cond_col].astype(str)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    out = numeric.copy()

    min_n = int(conn_cfg.get("min_epochs_per_group", 5))
    for lab in sorted(labels.unique()):
        mask = (labels == lab).to_numpy()
        n = int(np.sum(mask))
        if n < min_n:
            ctx.logger.warning("Connectivity condition group '%s' has only %d epochs (<%d); using subject mean for this group.", lab, n, min_n)
            grp_mean = numeric.mean(axis=0)
        else:
            grp_mean = numeric.loc[mask].mean(axis=0)
        out.loc[mask] = np.tile(grp_mean.to_numpy(dtype=float), (n, 1))

    out.columns = df.columns
    return out, list(out.columns)

# =============================================================================
# Precomputed Data Extractors (Moved from pipeline.py)
# =============================================================================

def _graph_metrics(
    adj: np.ndarray,
    measure: str,
    band: str,
    segment_name: str,
    conn_cfg: Dict[str, Any],
) -> Dict[str, float]:
    adj = np.asarray(adj, dtype=float)
    adj[~np.isfinite(adj)] = 0.0
    np.fill_diagonal(adj, 0.0)

    top_prop = conn_cfg.get("graph_top_prop", 0.1)
    try:
        top_prop = float(top_prop)
    except (ValueError, TypeError):
        top_prop = 0.1
    if not np.isfinite(top_prop) or top_prop <= 0 or top_prop > 1:
        top_prop = 0.1

    small_world_n_rand = conn_cfg.get("small_world_n_rand", 100)
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

    output_level = str(conn_cfg.get("output_level", "full")).strip().lower()
    if output_level not in {"full", "global_only"}:
        output_level = "full"

    enable_wpli = bool(conn_cfg.get("enable_wpli", True))
    enable_aec = bool(conn_cfg.get("enable_aec", True))
    enable_plv = bool(conn_cfg.get("enable_plv", False))
    enable_pli = bool(conn_cfg.get("enable_pli", False))
    enable_graph_metrics = bool(conn_cfg.get("enable_graph_metrics", False))
    
    # AEC output format: can include "r" (raw correlation) and/or "z" (Fisher-z transform)
    # Fisher-z is atanh(r) and averages correctly across trials/subjects
    aec_output_modes = conn_cfg.get("aec_output", ["r"])
    if isinstance(aec_output_modes, str):
        aec_output_modes = [aec_output_modes]
    aec_output_modes = [str(m).strip().lower() for m in aec_output_modes]
    if not aec_output_modes:
        aec_output_modes = ["r"]
    enable_aec_raw = "r" in aec_output_modes
    enable_aec_z = "z" in aec_output_modes

    # Supported by mne-connectivity (version-dependent); extraction will skip unsupported
    # methods if spectral_connectivity_time raises.
    supported_measures = {"wpli", "wpli2_debiased", "imcoh", "aec", "plv", "pli"}
    measures_cfg = conn_cfg.get("measures")
    if isinstance(measures_cfg, (list, tuple)) and measures_cfg:
        measures = {str(m).strip().lower() for m in measures_cfg}
        unknown = measures - supported_measures
        if unknown and logger is not None:
            logger.warning(
                "Connectivity: unsupported measures %s; ignoring.",
                ",".join(sorted(unknown)),
            )
        measures = measures & supported_measures
        enable_wpli = "wpli" in measures
        enable_wpli2 = "wpli2_debiased" in measures
        enable_imcoh = "imcoh" in measures
        enable_aec = "aec" in measures
        enable_plv = "plv" in measures
        enable_pli = "pli" in measures
    else:
        enable_wpli2 = bool(conn_cfg.get("enable_wpli2_debiased", False))
        enable_imcoh = bool(conn_cfg.get("enable_imcoh", False))
    phase_measures = [
        m
        for m, enabled in (
            ("wpli2_debiased", enable_wpli2),
            ("wpli", enable_wpli),
            ("imcoh", enable_imcoh),
            ("plv", enable_plv),
            ("pli", enable_pli),
        )
        if enabled
    ]

    if not phase_measures and not enable_aec:
        if logger is not None:
            logger.warning("Connectivity: no supported measures selected; skipping extraction.")
        return pd.DataFrame(), []

    target_name = getattr(precomputed.windows, "name", None) if precomputed.windows else None
    
    if target_name:
        seg_mask_map = {target_name: np.ones(len(precomputed.times), dtype=bool)}
    else:
        masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
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

    n_freqs_per_band = int(conn_cfg.get("n_freqs_per_band", 8))
    conn_mode = str(conn_cfg.get("mode", "cwt_morlet"))
    n_cycles = conn_cfg.get("n_cycles", None)
    try:
        n_cycles = float(n_cycles) if n_cycles is not None else None
    except Exception:
        n_cycles = None
    decim = int(conn_cfg.get("decim", 1))
    min_segment_samples = int(conn_cfg.get("min_segment_samples", 50))
    min_cycles_per_band = float(conn_cfg.get("min_cycles_per_band", 3.0))
    min_segment_sec = float(conn_cfg.get("min_segment_sec", 1.0))
    
    # Phase estimator mode: "within_epoch" (per-trial) or "across_epochs" (group-level, broadcast)
    phase_estimator = _normalize_phase_estimator(conn_cfg.get("phase_estimator", "within_epoch"))
    
    if phase_estimator == "across_epochs" and logger is not None:
        logger.info(
            "Connectivity: using across_epochs phase estimator (standard wPLI/PLV/PLI definition); "
            "values will be broadcast to all trials in each group."
        )

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute safe n_cycles and filter out frequencies that would crash MNE.
        
        Returns:
            Tuple of (valid_freqs, n_cycles_for_valid_freqs)
        """
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        
        if n_times <= 0:
            return np.array([]), np.array([])

        duration = float(n_times) / sfreq_hz
        default_cycles = 7.0
        try:
            base = float(base_n_cycles) if base_n_cycles is not None else default_cycles
        except Exception:
            base = default_cycles

        # MNE requirement: n_cycles / freqs < duration (or equivalently: wavelet_length < n_times)
        # Wavelet length = n_cycles / freq * sfreq
        # We use a CONSERVATIVE safety factor of 0.7 to avoid edge cases
        # This is stricter than the 0.9 factor that was causing crashes
        safety_factor = 0.7
        max_cycles = safety_factor * duration * freqs_hz
        
        # For each frequency, compute the safe n_cycles (capped at base)
        safe_cycles = np.minimum(base, max_cycles)
        
        # Filter out frequencies where even 1 cycle doesn't fit
        # (need at least 1 cycle for meaningful phase estimation)
        min_required_cycles = 1.0
        valid_mask = safe_cycles >= min_required_cycles
        
        valid_freqs = freqs_hz[valid_mask]
        valid_cycles = safe_cycles[valid_mask]
        
        return valid_freqs, valid_cycles

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
        raise ImportError(
            "Connectivity extraction requires 'mne-connectivity'. "
            "Install it with: pip install mne-connectivity"
        )

    freq_bands = getattr(precomputed, "frequency_bands", None) or get_frequency_bands(config)

    use_task_parallel = bool(n_jobs > 1)
    inner_n_jobs = 1 if use_task_parallel else n_jobs
    graph_n_jobs = 1 if use_task_parallel else n_jobs

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
            # Handle MNE's "wavelet longer than signal" and similar validation errors gracefully
            error_msg = str(e).lower()
            if "wavelet" in error_msg or "n_cycles" in error_msg or "longer than" in error_msg:
                if logger is not None:
                    logger.warning(
                        f"Connectivity: {method_use} skipped for segment '{seg_name}' band '{band}' - "
                        f"segment too short for wavelet-based connectivity. "
                        f"Consider using longer epochs or excluding low-frequency bands for short segments."
                    )
                return pd.DataFrame()
            # Re-raise other ValueErrors
            raise
        except Exception as e:
            # Graceful fallback for newer method names on older mne-connectivity installs.
            if method_use == "wpli2_debiased":
                if logger is not None:
                    logger.warning(
                        "Connectivity: method '%s' failed for segment '%s' band '%s' (%s); falling back to 'wpli'.",
                        method_use,
                        seg_name,
                        band,
                        e,
                    )
                try:
                    method_use = "wpli"
                    method_label = "wpli"
                    con = _run(method_use, use_average=use_across_epochs)
                except ValueError as e2:
                    error_msg = str(e2).lower()
                    if "wavelet" in error_msg or "n_cycles" in error_msg or "longer than" in error_msg:
                        if logger is not None:
                            logger.warning(
                                f"Connectivity: wpli fallback also skipped for segment '{seg_name}' band '{band}' - "
                                f"segment too short for wavelet-based connectivity."
                            )
                        return pd.DataFrame()
                    raise
                except Exception as e2:
                    if logger is not None:
                        logger.warning(
                            "Connectivity: fallback 'wpli' also failed for segment '%s' band '%s': %s",
                            seg_name,
                            band,
                            e2,
                        )
                    return pd.DataFrame()
            else:
                if logger is not None:
                    logger.warning(f"Connectivity: {method} failed for segment '{seg_name}' band '{band}': {e}")
                return pd.DataFrame()

        con_data = np.asarray(con.get_data())
        
        # Handle across_epochs mode: broadcast single result to all epochs
        if use_across_epochs:
            # con_data shape is (n_pairs,) or (n_pairs, n_freqs) when average=True
            if con_data.ndim == 1:
                con_vals = np.tile(con_data[None, :], (n_epochs, 1))
            elif con_data.ndim == 2:
                con_vals = np.tile(np.nanmean(con_data, axis=-1)[None, :], (n_epochs, 1))
            else:
                return pd.DataFrame()
        else:
            # within_epoch mode: per-trial connectivity
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
            suffix = f"_{method_label}"
            cols = [f"{prefix}{pair_name}{suffix}" for pair_name in pair_names]
            parts.append(pd.DataFrame(con_vals, columns=cols))

        glob_col = f"conn_{seg_name}_{band}_global_{method_label}_mean"
        parts.append(pd.DataFrame({glob_col: np.nanmean(con_vals, axis=1)}))

        if enable_graph_metrics:
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
                return _graph_metrics(adj, method_label, band, seg_name, conn_cfg)

            if graph_n_jobs == 1:
                graph_rows = [_graph_row(ep_idx) for ep_idx in range(n_epochs)]
            else:
                graph_rows = Parallel(n_jobs=graph_n_jobs, backend="loky")(
                    delayed(_graph_row)(ep_idx) for ep_idx in range(n_epochs)
                )
            parts.append(pd.DataFrame(graph_rows))

        df_out = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        if logger is not None:
            logger.debug("Connectivity task phase/%s/%s/%s finished in %.2fs (cols=%d)", seg_name, band, method, time.perf_counter() - t0, int(df_out.shape[1]))
        return df_out

    def _aec_task(seg_name: str, band: str, analytic_seg: np.ndarray) -> pd.DataFrame:
        t0 = time.perf_counter()
        aec_mode = str(conn_cfg.get("aec_mode", "orth")).strip().lower()
        orthogonalize = "pairwise"
        if aec_mode in {"none", "raw", "no"}:
            orthogonalize = False
        elif aec_mode in {"sym", "symmetric"}:
            orthogonalize = "sym"
        ec = envelope_correlation(
            analytic_seg,
            orthogonalize=orthogonalize,
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

        if enable_graph_metrics:
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
                return _graph_metrics(adj, "aec", band, seg_name, conn_cfg)

            if graph_n_jobs == 1:
                graph_rows = [_graph_row(ep_idx) for ep_idx in range(n_epochs)]
            else:
                graph_rows = Parallel(n_jobs=graph_n_jobs, backend="loky")(
                    delayed(_graph_row)(ep_idx) for ep_idx in range(n_epochs)
                )
            parts.append(pd.DataFrame(graph_rows))

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
            except Exception:
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
        dfs = Parallel(n_jobs=n_jobs, backend="loky")(
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
