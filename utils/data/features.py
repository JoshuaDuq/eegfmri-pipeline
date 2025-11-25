from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import glob
import logging

import mne
import numpy as np
import pandas as pd

from .loading import (
    register_feature_block,
    validate_feature_block_lengths,
    validate_trial_alignment_manifest,
)
from ..io.general import (
    _pick_target_column,
    deriv_stats_path,
    ensure_dir,
    write_tsv,
    sanitize_label,
)


###################################################################
# Feature Alignment and Saving
###################################################################

def align_feature_dataframes(
    pow_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    conn_df: Optional[pd.DataFrame],
    ms_df: Optional[pd.DataFrame],
    aper_df: Optional[pd.DataFrame],
    y: pd.Series,
    aligned_events: pd.DataFrame,
    features_dir: Path,
    logger: logging.Logger,
    config,
    initial_trial_count: Optional[int] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    pd.Series,
    Dict[str, Any],
]:
    """Align feature blocks and target vector while preserving valid trials."""
    def _block_length(block):
        if block is None or getattr(block, "empty", False):
            return None
        return len(block)

    n_trials = next(
        (l for l in [
            _block_length(pow_df),
            _block_length(baseline_df),
            _block_length(conn_df),
            _block_length(ms_df),
            _block_length(aper_df),
            len(y) if y is not None else None,
        ] if l is not None),
        None,
    )

    drop_mask: Optional[np.ndarray] = None
    if n_trials is not None:
        combined_mask = np.ones(n_trials, dtype=bool)

        if pow_df is not None and not getattr(pow_df, "empty", False):
            min_valid_band_fraction = float(config.get("feature_engineering.features.min_valid_band_fraction", 0.0))
            band_names = sorted({str(c).split("_")[1] for c in pow_df.columns if str(c).startswith("pow_") and len(str(c).split("_")) >= 3})
            band_drop_counts = {}
            for band in band_names:
                band_cols = [c for c in pow_df.columns if str(c).startswith(f"pow_{band}_")]
                if not band_cols:
                    continue
                values = pow_df[band_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                valid_fraction = np.isfinite(values).mean(axis=1)
                band_invalid = valid_fraction <= min_valid_band_fraction
                band_drop_counts[band] = int(band_invalid.sum())
                if np.any(band_invalid):
                    pow_df.loc[band_invalid, band_cols] = np.nan
                    qc_col = f"qc_pow_{band}"
                    if qc_col not in pow_df.columns:
                        pow_df[qc_col] = True
                    pow_df.loc[band_invalid, qc_col] = False
            for band, n_bad in band_drop_counts.items():
                if n_bad > 0:
                    logger.warning(
                        "Nulling %d trials with insufficient valid power samples for band %s (threshold=%.2f); trials retained for other bands.",
                        n_bad, band, min_valid_band_fraction,
                    )

        def _finite_mask(block):
            if block is None or getattr(block, "empty", False):
                return np.ones(n_trials, dtype=bool)
            numeric_block = block.apply(pd.to_numeric, errors="coerce") if isinstance(block, pd.DataFrame) else pd.to_numeric(block, errors="coerce")
            data = numeric_block.to_numpy(dtype=float)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return np.isfinite(data).any(axis=1)

        combined_mask &= _finite_mask(pow_df)
        combined_mask &= _finite_mask(baseline_df)
        combined_mask &= _finite_mask(conn_df)
        combined_mask &= _finite_mask(ms_df)
        combined_mask &= _finite_mask(aper_df)
        combined_mask &= _finite_mask(y)

        if not np.all(combined_mask):
            drop_mask = combined_mask

    if drop_mask is not None:
        if pow_df is not None:
            pow_df = pow_df.loc[drop_mask].reset_index(drop=True)
        if baseline_df is not None and not getattr(baseline_df, "empty", False):
            baseline_df = baseline_df.loc[drop_mask].reset_index(drop=True)
        if conn_df is not None and not getattr(conn_df, "empty", False):
            conn_df = conn_df.loc[drop_mask].reset_index(drop=True)
        if ms_df is not None and not getattr(ms_df, "empty", False) and len(ms_df) == len(drop_mask):
            ms_df = ms_df.loc[drop_mask].reset_index(drop=True)
        if aper_df is not None and not getattr(aper_df, "empty", False):
            aper_df = aper_df.loc[drop_mask].reset_index(drop=True)
        if y is not None:
            y = y.loc[drop_mask].reset_index(drop=True)

    block_registry: Dict[str, pd.DataFrame] = {}
    before_lengths: Dict[str, int] = {}

    def _is_valid_df(df):
        return df if df is not None and (not hasattr(df, "empty") or not df.empty) else None

    register_feature_block("power", _is_valid_df(pow_df), block_registry, before_lengths)
    register_feature_block("baseline", _is_valid_df(baseline_df), block_registry, before_lengths)
    register_feature_block("connectivity", _is_valid_df(conn_df), block_registry, before_lengths)
    register_feature_block("microstates", _is_valid_df(ms_df), block_registry, before_lengths)
    register_feature_block("aperiodic", _is_valid_df(aper_df), block_registry, before_lengths)
    register_feature_block("target", y, block_registry, before_lengths)

    if not block_registry:
        logger.warning("No features extracted; skipping save")
        return None, None, None, None, None, None, None

    validate_feature_block_lengths(before_lengths, logger)

    if aligned_events is not None and len(aligned_events) > 0:
        validate_trial_alignment_manifest(aligned_events, features_dir, logger)

    logger.info(f"Validated feature block lengths ({', '.join(f'{k}={v}' for k, v in before_lengths.items())})")

    pow_df_aligned = block_registry.get("power")
    if pow_df_aligned is not None:
        pow_df_aligned = pow_df_aligned.reset_index(drop=True)

    baseline_df_aligned = block_registry.get("baseline")
    if baseline_df_aligned is not None:
        baseline_df_aligned = baseline_df_aligned.reset_index(drop=True)
    else:
        baseline_df_aligned = baseline_df if baseline_df is not None else pd.DataFrame()

    conn_df_aligned = block_registry.get("connectivity")
    if conn_df_aligned is not None:
        conn_df_aligned = conn_df_aligned.reset_index(drop=True)

    ms_df_aligned = block_registry.get("microstates")
    if ms_df_aligned is not None:
        ms_df_aligned = ms_df_aligned.reset_index(drop=True)

    aper_df_aligned = block_registry.get("aperiodic")
    if aper_df_aligned is not None:
        aper_df_aligned = aper_df_aligned.reset_index(drop=True)

    target_block = block_registry.get("target")
    if target_block is not None:
        y_aligned = target_block.iloc[:, 0] if target_block.shape[1] == 1 else target_block
    else:
        y_aligned = y

    def _get_length(df):
        if df is None:
            return 0
        if hasattr(df, "empty") and df.empty:
            return 0
        return len(df) if hasattr(df, "__len__") else 0

    after_lengths = {
        "power": _get_length(pow_df_aligned),
        "baseline": _get_length(baseline_df_aligned),
        "connectivity": _get_length(conn_df_aligned),
        "microstates": _get_length(ms_df_aligned),
        "aperiodic": _get_length(aper_df_aligned),
        "target": _get_length(y_aligned),
    }

    retention_stats = {
        "initial_trial_count": initial_trial_count,
        "before_alignment": before_lengths,
        "after_alignment": after_lengths,
    }

    return pow_df_aligned, baseline_df_aligned, conn_df_aligned, ms_df_aligned, aper_df_aligned, y_aligned, retention_stats


# Plateau features are now in eeg_pipeline.analysis.features.plateau
from eeg_pipeline.analysis.features.plateau import build_plateau_features


def save_all_features(
    pow_df: pd.DataFrame,
    pow_cols: List[str],
    baseline_df: pd.DataFrame,
    baseline_cols: List[str],
    conn_df: Optional[pd.DataFrame],
    conn_cols: List[str],
    ms_df: Optional[pd.DataFrame],
    ms_cols: List[str],
    aper_df: Optional[pd.DataFrame],
    aper_cols: List[str],
    itpc_df: Optional[pd.DataFrame],
    itpc_cols: List[str],
    pac_df: Optional[pd.DataFrame],
    pac_trials_df: Optional[pd.DataFrame],
    pac_time_df: Optional[pd.DataFrame],
    aper_qc: Optional[Dict[str, Any]],
    plateau_df: pd.DataFrame,
    plateau_cols: List[str],
    y: pd.Series,
    features_dir: Path,
    logger: logging.Logger,
    config,
) -> pd.DataFrame:
    """Save all aligned feature blocks to disk and return combined features."""
    direct_blocks = [pow_df]
    if len(pow_cols) == len(pow_df.columns):
        pow_df.columns = pow_cols
    else:
        logger.warning("Power column mismatch: %d names vs %d columns. Using DataFrame names.", len(pow_cols), len(pow_df.columns))
    direct_cols = list(pow_df.columns)

    if not baseline_df.empty:
        direct_blocks.append(baseline_df)
        if baseline_cols:
            if len(baseline_cols) == len(baseline_df.columns):
                baseline_df.columns = baseline_cols
            else:
                logger.warning("Baseline column mismatch: %d names vs %d columns. Using DataFrame names.", len(baseline_cols), len(baseline_df.columns))
        direct_cols.extend(baseline_df.columns)

    if ms_df is not None and not ms_df.empty:
        direct_blocks.append(ms_df)
        if ms_cols:
            if len(ms_cols) == len(ms_df.columns):
                ms_df.columns = ms_cols
            else:
                logger.warning("Microstate column mismatch: %d names vs %d columns. Using DataFrame names.", len(ms_cols), len(ms_df.columns))
        direct_cols.extend(ms_df.columns)
        ms_path = features_dir / "features_microstates.tsv"
        logger.info("Saving microstate features: %s", ms_path)
        write_tsv(ms_df, ms_path)

    if aper_df is not None and not aper_df.empty:
        direct_blocks.append(aper_df)
        if aper_cols:
            if len(aper_cols) == len(aper_df.columns):
                aper_df.columns = aper_cols
            else:
                logger.warning("Aperiodic column mismatch: %d names vs %d columns. Using DataFrame names.", len(aper_cols), len(aper_df.columns))
        direct_cols.extend(aper_df.columns)

    if itpc_df is not None and not itpc_df.empty:
        if itpc_cols and len(itpc_cols) == len(itpc_df.columns):
            itpc_df.columns = itpc_cols
        itpc_path = features_dir / "features_itpc.tsv"
        logger.info("Saving ITPC features (channel x band x bin): %s", itpc_path)
        write_tsv(itpc_df, itpc_path)

    if pac_df is not None and not pac_df.empty:
        pac_path = features_dir / "features_pac.tsv"
        logger.info("Saving PAC comodulograms: %s", pac_path)
        write_tsv(pac_df, pac_path)
    if pac_trials_df is not None and not pac_trials_df.empty:
        pac_trials_path = features_dir / "features_pac_trials.tsv"
        logger.info("Saving PAC per-trial values: %s", pac_trials_path)
        write_tsv(pac_trials_df, pac_trials_path)
    if pac_time_df is not None and not pac_time_df.empty:
        pac_time_path = features_dir / "features_pac_time.tsv"
        logger.info("Saving PAC time-resolved values: %s", pac_time_path)
        write_tsv(pac_time_df, pac_time_path)

    if aper_qc:
        try:
            subject_name = features_dir.parent.parent.name.replace("sub-", "")
            deriv_root = features_dir.parent.parent.parent
            stats_dir = deriv_stats_path(deriv_root, subject_name)
            stats_dir.mkdir(parents=True, exist_ok=True)
            qc_path = stats_dir / "aperiodic_qc.npz"
            np.savez_compressed(
                qc_path,
                freqs=aper_qc.get("freqs"),
                residual_mean=aper_qc.get("residual_mean"),
                r2=aper_qc.get("r2"),
                slopes=aper_qc.get("slopes"),
                offsets=aper_qc.get("offsets"),
                channel_names=aper_qc.get("channel_names"),
                run_labels=aper_qc.get("run_labels"),
            )
            logger.info("Saved aperiodic QC sidecar to %s", qc_path)
        except Exception as exc:
            logger.warning("Failed to save aperiodic QC npz: %s", exc)

    direct_df = pd.concat(direct_blocks, axis=1)

    eeg_direct_path = features_dir / "features_eeg_direct.tsv"
    eeg_direct_cols_path = features_dir / "features_eeg_direct_columns.tsv"
    logger.info("Saving direct EEG features: %s", eeg_direct_path)
    write_tsv(direct_df, eeg_direct_path)
    write_tsv(pd.Series(direct_cols, name="feature").to_frame(), eeg_direct_cols_path)

    if not plateau_df.empty:
        plateau_path = features_dir / "features_eeg_plateau.tsv"
        plateau_cols_path = features_dir / "features_eeg_plateau_columns.tsv"
        logger.info("Saving plateau-averaged EEG features: %s", plateau_path)
        write_tsv(plateau_df, plateau_path)
        write_tsv(pd.Series(plateau_cols, name="feature").to_frame(), plateau_cols_path)

    if conn_df is not None and not conn_df.empty:
        if conn_cols:
            if len(conn_cols) == len(conn_df.columns):
                conn_df.columns = conn_cols
            else:
                logger.warning("Connectivity column mismatch: %d names vs %d columns. Using DataFrame names.", len(conn_cols), len(conn_df.columns))

        conn_path = features_dir / "features_connectivity.tsv"
        logger.info("Saving connectivity features: %s", conn_path)
        write_tsv(conn_df, conn_path)

    blocks = [direct_df]
    cols_all = list(direct_df.columns)
    if conn_df is not None and not conn_df.empty:
        blocks.append(conn_df)
        cols_all.extend(conn_df.columns)

    combined_df = pd.concat(blocks, axis=1)

    combined_path = features_dir / "features_all.tsv"
    logger.info("Saving combined features: %s", combined_path)
    write_tsv(combined_df, combined_path)

    y_path = features_dir / "target_vas_ratings.tsv"
    rating_columns = config.get("event_columns.rating", ["vas_rating"])
    target_column_name = rating_columns[0] if rating_columns else "vas_rating"
    logger.info("Saving behavioral target vector: %s (column: %s)", y_path, target_column_name)
    write_tsv(y.to_frame(name=target_column_name), y_path)

    return combined_df


def _extract_onset_duration(aligned_events: pd.DataFrame, logger: logging.Logger, n_fallback: int) -> Tuple[np.ndarray, np.ndarray]:
    n_trials = len(aligned_events) if aligned_events is not None else n_fallback
    onset = None
    duration = None
    if aligned_events is not None:
        if "onset" in aligned_events.columns:
            onset = pd.to_numeric(aligned_events["onset"], errors="coerce")
        if "duration" in aligned_events.columns:
            duration = pd.to_numeric(aligned_events["duration"], errors="coerce")
    if onset is None:
        onset = pd.Series(np.arange(n_trials, dtype=float))
        logger.warning("Onset column missing; using 0..n-1 as placeholder onsets for regressors.")
    if duration is None:
        duration = pd.Series(np.zeros(n_trials, dtype=float))
    onset = onset.fillna(method="ffill").fillna(method="bfill").fillna(0.0).astype(float)
    duration = duration.fillna(0.0).astype(float)
    return onset.to_numpy(), duration.to_numpy()


def export_fmri_regressors(
    aligned_events: pd.DataFrame,
    plateau_df: pd.DataFrame,
    plateau_cols: List[str],
    ms_df: Optional[pd.DataFrame],
    pac_trials_df: Optional[pd.DataFrame],
    aper_df: Optional[pd.DataFrame],
    y: pd.Series,
    power_bands: List[str],
    subject: str,
    task: str,
    features_dir: Path,
    config,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Export fMRI regressors derived from EEG features."""
    if aligned_events is None or aligned_events.empty:
        logger.warning("Cannot export fMRI regressors: aligned_events is empty.")
        return None

    n_trials = len(aligned_events)
    onset, duration = _extract_onset_duration(aligned_events, logger, n_trials)
    rating = pd.to_numeric(y, errors="coerce") if y is not None else pd.Series(np.full(n_trials, np.nan))

    reg_df = pd.DataFrame({
        "onset": onset,
        "duration": duration,
        "rating": rating,
    })
    reg_values: Dict[str, pd.Series] = {}

    def _add_regressor(name: str, series: Any):
        if series is None:
            return
        ser = pd.to_numeric(series, errors="coerce")
        if len(ser) != n_trials:
            logger.warning("Regressor %s length mismatch (got %d, expected %d); skipping.", name, len(ser), n_trials)
            return
        reg_df[name] = ser
        reg_values[name] = ser

    for band in power_bands:
        band_cols = [c for c in plateau_cols if c.startswith(f"pow_{band}_")]
        if not band_cols:
            continue
        _add_regressor(f"pow_{band}_mean", plateau_df[band_cols].mean(axis=1))

    if ms_df is not None and not ms_df.empty:
        for col in ms_df.columns:
            if col.startswith("ms_coverage_") or col.startswith("ms_duration_"):
                _add_regressor(col, ms_df[col])

    if aper_df is not None and not aper_df.empty:
        slope_cols = [c for c in aper_df.columns if c.startswith("aper_slope_")]
        if slope_cols:
            _add_regressor("aper_slope_mean", aper_df[slope_cols].mean(axis=1))

    if pac_trials_df is not None and not pac_trials_df.empty and "trial" in pac_trials_df.columns and "pac" in pac_trials_df.columns:
        pac_vals = np.full(n_trials, np.nan, dtype=float)
        for trial_idx, val in pac_trials_df.groupby("trial")["pac"].mean().items():
            try:
                t = int(trial_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= t < n_trials:
                pac_vals[t] = float(val)
        _add_regressor("pac_mean", pac_vals)

    if len(reg_df.columns) <= 3:
        logger.warning("No regressors were added; skipping fMRI regressor export.")
        return None

    reg_dir = features_dir / "fmri_regressors"
    ensure_dir(reg_dir)

    combined_path = reg_dir / f"sub-{subject}_task-{task}_fmri_regressors.tsv"
    write_tsv(reg_df, combined_path)

    for name, series in reg_values.items():
        reg_path = reg_dir / f"sub-{subject}_task-{task}_regressor_{sanitize_label(name)}.tsv"
        reg_out = pd.DataFrame({
            "onset": onset,
            "duration": duration,
            "amplitude": series,
        })
        write_tsv(reg_out, reg_path)

    logger.info("Saved %d fMRI regressors to %s", len(reg_values), reg_dir)
    return reg_df


###################################################################
# Microstate Templates
###################################################################

def save_microstate_templates(
    epochs: mne.Epochs,
    templates: np.ndarray,
    subject: str,
    n_states: int,
    deriv_root: Path,
    logger: logging.Logger,
) -> None:
    if templates is None:
        return

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    template_path = stats_dir / f"microstates_templates_K{n_states}.npz"

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    ch_names = [epochs.info["ch_names"][i] for i in picks]

    np.savez_compressed(
        template_path,
        templates=templates,
        ch_names=np.array(ch_names),
        n_states=n_states
    )
    logger.info("Saved microstate templates to %s", template_path)


def load_group_microstate_templates(
    deriv_root: Path,
    n_states: int,
    logger: logging.Logger,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    group_path = deriv_root / "group" / "eeg" / "stats" / f"microstates_templates_group_K{n_states}.npz"
    if not group_path.exists():
        return None, None
    try:
        data = np.load(group_path, allow_pickle=True)
        templates = data.get("templates")
        ch_names = data.get("ch_names")
        if templates is None or ch_names is None:
            logger.warning("Group microstate templates found but missing templates/ch_names")
            return None, None
        logger.info("Loaded group microstate templates from %s", group_path)
        return templates, list(ch_names)
    except Exception as exc:
        logger.warning("Failed to load group microstate templates from %s: %s", group_path, exc)
        return None, None


def compute_group_microstate_templates(
    deriv_root: Path,
    n_states: int,
    logger: logging.Logger,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    pattern = deriv_root / "sub-*" / "eeg" / "stats" / f"microstates_templates_K{n_states}.npz"
    files = sorted(Path(p) for p in glob.glob(str(pattern)))
    if not files:
        logger.warning("No subject microstate templates found to build group template")
        return None, None

    templates_list = []
    channel_sets = []
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            templ = data.get("templates")
            chs = data.get("ch_names")
            if templ is None or chs is None:
                continue
            templates_list.append((templ, list(chs)))
            channel_sets.append(set(chs.tolist()))
        except Exception as exc:
            logger.warning("Skipping template file %s: %s", f, exc)
            continue

    if not templates_list or not channel_sets:
        logger.warning("No valid microstate templates to build group template")
        return None, None

    common_chs = sorted(set.intersection(*channel_sets))
    if not common_chs:
        logger.warning("No common channels across microstate templates; cannot build group template")
        return None, None

    aligned_templates = []
    for templ, chs in templates_list:
        indices = [chs.index(ch) for ch in common_chs]
        aligned = templ[:, indices]
        signs = np.sign(np.max(np.abs(aligned), axis=1))
        signs[signs == 0] = 1
        aligned_templates.append(aligned * signs[:, np.newaxis])

    stacked = np.stack(aligned_templates, axis=0)
    group_templates = np.nanmean(stacked, axis=0)

    group_dir = deriv_root / "group" / "eeg" / "stats"
    group_dir.mkdir(parents=True, exist_ok=True)
    out_path = group_dir / f"microstates_templates_group_K{n_states}.npz"
    np.savez_compressed(out_path, templates=group_templates, ch_names=np.array(common_chs))
    logger.info("Saved group microstate templates to %s", out_path)
    return group_templates, common_chs


###################################################################
# Trial Alignment Logs
###################################################################

def save_trial_alignment_manifest(
    aligned_events: pd.DataFrame,
    epochs: mne.Epochs,
    manifest_path: Path,
    config,
    logger: logging.Logger,
) -> None:
    if aligned_events is None or len(aligned_events) == 0:
        raise ValueError("Cannot save trial alignment manifest: aligned_events is None or empty")

    if len(aligned_events) != len(epochs):
        raise ValueError(
            f"Cannot save trial alignment manifest: length mismatch "
            f"(aligned_events={len(aligned_events)}, epochs={len(epochs)})"
        )

    manifest = pd.DataFrame({"trial_index": np.arange(len(aligned_events), dtype=int)})

    if "sample" in aligned_events.columns:
        manifest["sample"] = aligned_events["sample"].values
    if "onset" in aligned_events.columns:
        manifest["onset"] = aligned_events["onset"].values

    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    target_col = _pick_target_column(aligned_events, constants=constants)
    if target_col is not None:
        manifest["target_value"] = pd.to_numeric(aligned_events[target_col], errors="coerce").values

    write_tsv(manifest, manifest_path)
    logger.info("Saved trial alignment manifest with %d trials to %s", len(manifest), manifest_path)


def save_dropped_trials_log(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    drop_log_path: Path,
    logger: logging.Logger,
) -> None:
    selection = getattr(epochs, "selection", None)
    if selection is None:
        raise AttributeError("epochs.selection missing")

    selection_arr = np.asarray(selection, dtype=int)
    valid_mask = (selection_arr >= 0) & (selection_arr < len(events_df))

    if not np.all(valid_mask):
        logger.warning("Epoch selection contains indices outside events range; restricting to valid entries")
        selection_arr = selection_arr[valid_mask]

    kept_indices = set(int(idx) for idx in selection_arr.tolist())
    dropped_indices = [idx for idx in range(len(events_df)) if idx not in kept_indices]

    if not dropped_indices:
        write_tsv(pd.DataFrame(columns=["original_index", "drop_reason"]), drop_log_path)
        logger.info("No dropped trials detected; wrote empty drop log to %s", drop_log_path)
        return

    dropped_events = events_df.iloc[dropped_indices].copy()
    drop_log = getattr(epochs, "drop_log", None)

    if isinstance(drop_log, (list, tuple)) and len(drop_log) == len(events_df):
        drop_reasons = [
            ";".join(str(x) for x in entry if x) if isinstance(entry, (list, tuple)) else str(entry) if entry else ""
            for idx in dropped_indices
            for entry in [drop_log[idx]]
        ]
    else:
        drop_reasons = [""] * len(dropped_indices)

    dropped_events.insert(0, "original_index", dropped_indices)
    dropped_events["drop_reason"] = drop_reasons
    write_tsv(dropped_events, drop_log_path)
    logger.info("Saved drop log with %d dropped trials to %s", len(dropped_events), drop_log_path)
