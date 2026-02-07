"""
EEG Microstate Features
=======================

Per-trial microstate metrics using GFP-peak map extraction and clustering:
- Coverage per class
- Mean duration per class (ms)
- Occurrence rate per class (Hz)
- Transition probabilities between classes

Designed for canonical A-D usage with either fixed templates (recommended) or
subject-level K-means templates fitted on GFP peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_lowercase
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.utils.analysis.windowing import get_segment_masks


_DEFAULT_CLASS_LABELS = ("a", "b", "c", "d")
_DEFAULT_N_STATES = 4
_DEFAULT_MIN_PEAK_DISTANCE_MS = 10.0
_DEFAULT_MAX_GFP_PEAKS_PER_EPOCH = 400
_DEFAULT_MIN_DURATION_MS = 0.0
_DEFAULT_RANDOM_STATE = 42


@dataclass
class _MicrostateConfig:
    n_states: int
    min_peak_distance_ms: float
    max_gfp_peaks_per_epoch: int
    min_duration_ms: float
    gfp_peak_prominence: float
    random_state: int


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _load_microstate_config(config: Any) -> _MicrostateConfig:
    micro_cfg = _cfg_get(config, "feature_engineering.microstates", {}) or {}
    n_states = int(micro_cfg.get("n_states", _DEFAULT_N_STATES))
    n_states = int(np.clip(n_states, 2, 12))

    min_peak_distance_ms = float(
        micro_cfg.get("min_peak_distance_ms", _DEFAULT_MIN_PEAK_DISTANCE_MS)
    )
    min_peak_distance_ms = max(0.0, min_peak_distance_ms)

    max_gfp_peaks_per_epoch = int(
        micro_cfg.get("max_gfp_peaks_per_epoch", _DEFAULT_MAX_GFP_PEAKS_PER_EPOCH)
    )
    max_gfp_peaks_per_epoch = max(10, max_gfp_peaks_per_epoch)

    min_duration_ms = float(micro_cfg.get("min_duration_ms", _DEFAULT_MIN_DURATION_MS))
    min_duration_ms = max(0.0, min_duration_ms)

    gfp_peak_prominence = float(micro_cfg.get("gfp_peak_prominence", 0.0))
    gfp_peak_prominence = max(0.0, gfp_peak_prominence)

    random_state = int(
        micro_cfg.get("random_state", _cfg_get(config, "project.random_state", _DEFAULT_RANDOM_STATE))
    )

    return _MicrostateConfig(
        n_states=n_states,
        min_peak_distance_ms=min_peak_distance_ms,
        max_gfp_peaks_per_epoch=max_gfp_peaks_per_epoch,
        min_duration_ms=min_duration_ms,
        gfp_peak_prominence=gfp_peak_prominence,
        random_state=random_state,
    )


def _normalize_topography(topography: np.ndarray) -> np.ndarray:
    vec = np.asarray(topography, dtype=float)
    vec = vec - np.nanmean(vec)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(vec)
    vec = vec / norm
    pivot = int(np.argmax(np.abs(vec)))
    if vec[pivot] < 0:
        vec = -vec
    return vec


def _normalize_topographies(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    out = np.zeros_like(matrix, dtype=float)
    for idx in range(matrix.shape[0]):
        out[idx] = _normalize_topography(matrix[idx])
    return out


def _compute_gfp(epoch_data: np.ndarray) -> np.ndarray:
    data = np.asarray(epoch_data, dtype=float)
    demeaned = data - np.nanmean(data, axis=0, keepdims=True)
    return np.nanstd(demeaned, axis=0)


def _extract_peak_topographies(
    epoch_data: np.ndarray,
    sfreq: float,
    cfg: _MicrostateConfig,
) -> np.ndarray:
    if epoch_data.ndim != 2 or epoch_data.shape[1] < 3:
        return np.empty((0, epoch_data.shape[0]), dtype=float)

    gfp = _compute_gfp(epoch_data)
    distance_samples = max(1, int(round(cfg.min_peak_distance_ms * sfreq / 1000.0)))
    peaks, _ = find_peaks(
        gfp,
        distance=distance_samples,
        prominence=cfg.gfp_peak_prominence if cfg.gfp_peak_prominence > 0 else None,
    )
    if peaks.size == 0:
        peaks = np.array([int(np.nanargmax(gfp))], dtype=int)

    peak_scores = gfp[peaks]
    order = np.argsort(peak_scores)[::-1]
    peaks = peaks[order[: cfg.max_gfp_peaks_per_epoch]]

    topographies = epoch_data[:, peaks].T
    return _normalize_topographies(topographies)


def _resolve_fixed_templates(
    fixed_templates: Optional[np.ndarray],
    fixed_template_ch_names: Optional[Sequence[str]],
    selected_ch_names: Sequence[str],
    n_states: int,
) -> Optional[np.ndarray]:
    if fixed_templates is None:
        return None

    templates = np.asarray(fixed_templates, dtype=float)
    if templates.ndim != 2:
        return None
    if templates.shape[0] < n_states:
        return None

    if fixed_template_ch_names is not None and len(fixed_template_ch_names) > 0:
        ch_to_idx = {str(ch): i for i, ch in enumerate(fixed_template_ch_names)}
        channel_indices = []
        for ch in selected_ch_names:
            if ch not in ch_to_idx:
                return None
            channel_indices.append(ch_to_idx[ch])
        templates = templates[:, channel_indices]
    elif templates.shape[1] != len(selected_ch_names):
        return None

    templates = templates[:n_states]
    return _normalize_topographies(templates)


def _fit_templates_kmeans(
    peak_maps: np.ndarray,
    n_states: int,
    random_state: int,
) -> np.ndarray:
    if peak_maps.shape[0] < n_states:
        repeats = int(np.ceil(n_states / max(1, peak_maps.shape[0])))
        peak_maps = np.tile(peak_maps, (repeats, 1))

    model = KMeans(
        n_clusters=n_states,
        n_init=20,
        random_state=random_state,
    )
    model.fit(peak_maps)
    centers = _normalize_topographies(model.cluster_centers_)
    return centers


def _assign_states(sample_maps: np.ndarray, templates: np.ndarray) -> np.ndarray:
    if sample_maps.size == 0:
        return np.array([], dtype=int)

    normalized = _normalize_topographies(sample_maps)
    sim = np.abs(templates @ normalized.T)
    return np.argmax(sim, axis=0).astype(int)


def _apply_min_duration(states: np.ndarray, min_samples: int) -> np.ndarray:
    if states.size == 0 or min_samples <= 1:
        return states

    out = states.copy()
    runs: List[Tuple[int, int, int]] = []
    start = 0
    for idx in range(1, len(out) + 1):
        if idx == len(out) or out[idx] != out[start]:
            runs.append((start, idx, int(out[start])))
            start = idx

    for run_idx, (run_start, run_end, _run_state) in enumerate(runs):
        run_len = run_end - run_start
        if run_len >= min_samples:
            continue
        prev_state = runs[run_idx - 1][2] if run_idx > 0 else None
        next_state = runs[run_idx + 1][2] if run_idx < len(runs) - 1 else None
        replacement = prev_state if prev_state is not None else next_state
        if replacement is None:
            continue
        out[run_start:run_end] = int(replacement)
    return out


def _run_lengths(states: np.ndarray) -> List[Tuple[int, int]]:
    if states.size == 0:
        return []
    runs: List[Tuple[int, int]] = []
    start = 0
    for idx in range(1, len(states) + 1):
        if idx == len(states) or states[idx] != states[start]:
            runs.append((int(states[start]), idx - start))
            start = idx
    return runs


def _collapsed_states(states: np.ndarray) -> np.ndarray:
    if states.size == 0:
        return states
    keep = np.ones(len(states), dtype=bool)
    keep[1:] = states[1:] != states[:-1]
    return states[keep]


def _compute_epoch_metrics(
    states: np.ndarray,
    sfreq: float,
    n_states: int,
) -> Dict[str, np.ndarray]:
    n_samples = len(states)
    if n_samples == 0:
        return {
            "coverage": np.full(n_states, np.nan, dtype=float),
            "duration_ms": np.full(n_states, np.nan, dtype=float),
            "occurrence_hz": np.full(n_states, np.nan, dtype=float),
            "transitions": np.full((n_states, n_states), np.nan, dtype=float),
        }

    coverage = np.array([(states == k).mean() for k in range(n_states)], dtype=float)

    runs = _run_lengths(states)
    durations_ms = np.full(n_states, np.nan, dtype=float)
    occurrence_hz = np.zeros(n_states, dtype=float)
    epoch_sec = n_samples / float(sfreq)

    for k in range(n_states):
        class_runs = [run_len for state, run_len in runs if state == k]
        if class_runs:
            durations_ms[k] = float(np.mean(class_runs) * 1000.0 / sfreq)
            occurrence_hz[k] = float(len(class_runs) / max(epoch_sec, 1e-12))

    collapsed = _collapsed_states(states)
    counts = np.zeros((n_states, n_states), dtype=float)
    for idx in range(len(collapsed) - 1):
        src = int(collapsed[idx])
        dst = int(collapsed[idx + 1])
        counts[src, dst] += 1.0

    transitions = np.zeros_like(counts)
    row_sums = counts.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    transitions[nonzero] = counts[nonzero] / row_sums[nonzero]

    return {
        "coverage": coverage,
        "duration_ms": durations_ms,
        "occurrence_hz": occurrence_hz,
        "transitions": transitions,
    }


def _class_labels(n_states: int) -> List[str]:
    if n_states <= len(_DEFAULT_CLASS_LABELS):
        return list(_DEFAULT_CLASS_LABELS[:n_states])
    labels = list(_DEFAULT_CLASS_LABELS)
    extra = n_states - len(labels)
    labels.extend(list(ascii_lowercase[len(labels) : len(labels) + extra]))
    return labels


def _build_column_name(segment: str, stat: str) -> str:
    return NamingSchema.build(
        group="microstates",
        segment=segment,
        band="broadband",
        scope="global",
        stat=stat,
    )


def extract_microstate_features(ctx: Any) -> Tuple[pd.DataFrame, List[str]]:
    """Extract per-trial microstate features from EEG epochs."""
    epochs = getattr(ctx, "epochs", None)
    if epochs is None or len(epochs) == 0:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        raise ValueError("Microstates: no EEG channels available.")

    cfg = _load_microstate_config(getattr(ctx, "config", None))
    sfreq = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks)  # shape: (n_epochs, n_channels, n_times)

    windows = getattr(ctx, "windows", None)
    target_name = getattr(ctx, "name", None)
    logger = getattr(ctx, "logger", None)

    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segment_masks = {target_name: mask}
        else:
            if logger is not None:
                logger.error(
                    "Microstates: targeted window '%s' has no valid mask; skipping.",
                    target_name,
                )
            return pd.DataFrame(), []
    else:
        segment_masks = get_segment_masks(epochs.times, windows, getattr(ctx, "config", None))

    if not segment_masks:
        return pd.DataFrame(), []

    # Training maps are pooled across segments for consistent templates.
    pooled_peak_maps: List[np.ndarray] = []
    for epoch_idx in range(data.shape[0]):
        epoch = data[epoch_idx]
        for mask in segment_masks.values():
            if mask is None or not np.any(mask):
                continue
            segment_epoch = epoch[:, mask]
            peak_maps = _extract_peak_topographies(segment_epoch, sfreq, cfg)
            if peak_maps.size > 0:
                pooled_peak_maps.append(peak_maps)

    if not pooled_peak_maps:
        return pd.DataFrame(), []

    pooled_matrix = np.vstack(pooled_peak_maps)
    templates = _resolve_fixed_templates(
        fixed_templates=getattr(ctx, "fixed_templates", None),
        fixed_template_ch_names=getattr(ctx, "fixed_template_ch_names", None),
        selected_ch_names=ch_names,
        n_states=cfg.n_states,
    )
    if templates is None:
        templates = _fit_templates_kmeans(
            pooled_matrix,
            n_states=cfg.n_states,
            random_state=cfg.random_state,
        )

    labels = _class_labels(cfg.n_states)
    min_duration_samples = max(1, int(round(cfg.min_duration_ms * sfreq / 1000.0)))

    all_rows: Dict[str, List[float]] = {}
    n_epochs = data.shape[0]

    for segment_name, mask in segment_masks.items():
        if mask is None or not np.any(mask):
            continue

        for epoch_idx in range(n_epochs):
            epoch = data[epoch_idx][:, mask]
            sample_maps = epoch.T
            states = _assign_states(sample_maps, templates)
            states = _apply_min_duration(states, min_duration_samples)
            metrics = _compute_epoch_metrics(states, sfreq, cfg.n_states)

            for class_idx, label in enumerate(labels):
                cov_col = _build_column_name(segment_name, f"coverage_{label}")
                dur_col = _build_column_name(segment_name, f"duration_ms_{label}")
                occ_col = _build_column_name(segment_name, f"occurrence_hz_{label}")

                if cov_col not in all_rows:
                    all_rows[cov_col] = [np.nan] * n_epochs
                    all_rows[dur_col] = [np.nan] * n_epochs
                    all_rows[occ_col] = [np.nan] * n_epochs

                all_rows[cov_col][epoch_idx] = float(metrics["coverage"][class_idx])
                all_rows[dur_col][epoch_idx] = float(metrics["duration_ms"][class_idx])
                all_rows[occ_col][epoch_idx] = float(metrics["occurrence_hz"][class_idx])

            trans = metrics["transitions"]
            for src_idx, src_label in enumerate(labels):
                for dst_idx, dst_label in enumerate(labels):
                    stat = f"trans_{src_label}_to_{dst_label}_prob"
                    col = _build_column_name(segment_name, stat)
                    if col not in all_rows:
                        all_rows[col] = [np.nan] * n_epochs
                    all_rows[col][epoch_idx] = float(trans[src_idx, dst_idx])

    if not all_rows:
        return pd.DataFrame(), []

    out_df = pd.DataFrame(all_rows)
    return out_df, list(out_df.columns)


__all__ = ["extract_microstate_features"]
