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
from ..io.columns import _pick_target_column
from ..io.paths import deriv_stats_path, ensure_dir
from ..io.tsv import write_tsv
from ..io.formatting import sanitize_label


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
    critical_features: Optional[List[str]] = None,
    extra_blocks: Optional[Dict[str, pd.DataFrame]] = None,
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

            def _power_band(col: Any) -> Optional[str]:
                name = str(col)
                if name.startswith("power_"):
                    parts = name.split("_")
                    if len(parts) >= 3:
                        return parts[2]
                    return None
                if name.startswith("pow_"):
                    parts = name.split("_")
                    if len(parts) >= 2:
                        return parts[1]
                    return None
                return None

            band_names = sorted({b for b in (_power_band(c) for c in pow_df.columns) if b})
            band_drop_counts = {}
            for band in band_names:
                band_cols = [c for c in pow_df.columns if _power_band(c) == band]
                if not band_cols:
                    continue
                values = pow_df[band_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                valid_fraction = np.isfinite(values).mean(axis=1)
                band_invalid = valid_fraction <= min_valid_band_fraction
                band_drop_counts[band] = int(band_invalid.sum())
                if np.any(band_invalid):
                    pow_df.loc[band_invalid, band_cols] = np.nan
                    qc_col = f"qc_power_{band}"
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
        n_mask = len(drop_mask)
        
        def _safe_mask_apply(df, name: str):
            """Apply mask only if DataFrame length matches mask length."""
            if df is None or getattr(df, "empty", False):
                return df
            if len(df) != n_mask:
                logger.warning(
                    f"Length mismatch for {name}: DataFrame has {len(df)} rows but mask has {n_mask}. "
                    f"Skipping mask application to avoid misalignment."
                )
                return df
            return df.loc[drop_mask].reset_index(drop=True)
        
        pow_df = _safe_mask_apply(pow_df, "power")
        baseline_df = _safe_mask_apply(baseline_df, "baseline")
        conn_df = _safe_mask_apply(conn_df, "connectivity")
        ms_df = _safe_mask_apply(ms_df, "microstates")
        aper_df = _safe_mask_apply(aper_df, "aperiodic")
        if y is not None and len(y) == n_mask:
            y = y.loc[drop_mask].reset_index(drop=True)
        elif y is not None:
            logger.warning(
                f"Length mismatch for target: Series has {len(y)} rows but mask has {n_mask}. "
                f"Skipping mask application to avoid misalignment."
            )

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
    
    if extra_blocks:
        for block_name, block_df in extra_blocks.items():
            register_feature_block(block_name, _is_valid_df(block_df), block_registry, before_lengths)

    if not block_registry:
        logger.warning("No features extracted; skipping save")
        return None, None, None, None, None, None, None

    validate_feature_block_lengths(before_lengths, logger, critical_features=critical_features)

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

    extra_aligned: Dict[str, Optional[pd.DataFrame]] = {}
    if extra_blocks:
        for block_name in extra_blocks.keys():
            block = block_registry.get(block_name)
            if block is not None:
                block = block.reset_index(drop=True)
            extra_aligned[block_name] = block
            after_lengths[block_name] = _get_length(block)

    retention_stats = {
        "initial_trial_count": initial_trial_count,
        "before_alignment": before_lengths,
        "after_alignment": after_lengths,
        "mask": drop_mask,
        "extra_aligned": extra_aligned,
    }

    return pow_df_aligned, baseline_df_aligned, conn_df_aligned, ms_df_aligned, aper_df_aligned, y_aligned, retention_stats


# Plateau features are now in eeg_pipeline.utils.data.manipulation
from .manipulation import build_plateau_features


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
    plateau_df: Optional[pd.DataFrame],
    plateau_cols: Optional[List[str]],
    y: pd.Series,
    features_dir: Path,
    logger: logging.Logger,
    config,
    # New features
    comp_df: Optional[pd.DataFrame] = None,
    comp_cols: List[str] = None,
    dynamics_df: Optional[pd.DataFrame] = None,
    dynamics_cols: List[str] = None,
    cfc_df: Optional[pd.DataFrame] = None,
    cfc_cols: List[str] = None,
    precomputed_df: Optional[pd.DataFrame] = None,
    precomputed_cols: Optional[List[str]] = None,
    feature_qc: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Save all aligned feature blocks to disk and return combined features."""
    import json
    from eeg_pipeline.utils.analysis.features.metadata import generate_manifest

    direct_blocks = []
    direct_cols = []
    
    if pow_df is not None:
        direct_blocks.append(pow_df)
        if len(pow_cols) == len(pow_df.columns):
            pow_df.columns = pow_cols
        else:
            logger.warning("Power column mismatch: %d names vs %d columns. Using DataFrame names.", len(pow_cols), len(pow_df.columns))
        direct_cols.extend(list(pow_df.columns))

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
        aper_path = features_dir / "features_aperiodic.tsv"
        logger.info("Saving aperiodic features: %s", aper_path)
        write_tsv(aper_df, aper_path)

    if itpc_df is not None and not itpc_df.empty:
        if itpc_cols and len(itpc_cols) == len(itpc_df.columns):
            itpc_df.columns = itpc_cols
        # Add ITPC to direct blocks so it's included in features_all.tsv
        direct_blocks.append(itpc_df)
        direct_cols.extend(itpc_df.columns)
        itpc_path = features_dir / "features_itpc.tsv"
        logger.info("Saving ITPC features (channel x band x segment): %s", itpc_path)
        write_tsv(itpc_df, itpc_path)

    if pac_df is not None and not pac_df.empty:
        pac_path = features_dir / "features_pac.tsv"
        logger.info("Saving PAC comodulograms: %s", pac_path)
        write_tsv(pac_df, pac_path)
    if pac_trials_df is not None and not pac_trials_df.empty:
        pac_trials_path = features_dir / "features_pac_trials.tsv"
        logger.info("Saving PAC per-trial values: %s", pac_trials_path)
        write_tsv(pac_trials_df, pac_trials_path)
        # Include PAC per-trial values in features_all.tsv
        direct_blocks.append(pac_trials_df)
        direct_cols.extend(list(pac_trials_df.columns))
    if pac_time_df is not None and not pac_time_df.empty:
        pac_time_path = features_dir / "features_pac_time.tsv"
        logger.info("Saving PAC time-resolved values: %s", pac_time_path)
        write_tsv(pac_time_df, pac_time_path)
        # Only include PAC time-resolved features in features_all.tsv if they are trial-aligned
        try:
            n_trials = len(y) if y is not None else None
            if n_trials is not None and len(pac_time_df) == n_trials:
                direct_blocks.append(pac_time_df)
                direct_cols.extend(list(pac_time_df.columns))
            else:
                logger.info(
                    "PAC time-resolved output is not trial-aligned (rows=%d, trials=%s); excluding from features_all.tsv",
                    len(pac_time_df),
                    str(n_trials) if n_trials is not None else "unknown",
                )
        except Exception as exc:
            logger.warning("Failed to evaluate PAC time alignment; excluding from features_all.tsv: %s", exc)

    # Complexity features
    if comp_df is not None and not comp_df.empty:
        direct_blocks.append(comp_df)
        if comp_cols:
            if len(comp_cols) == len(comp_df.columns):
                comp_df.columns = comp_cols
        direct_cols.extend(comp_df.columns)
        comp_path = features_dir / "features_complexity.tsv"
        logger.info("Saving complexity features: %s", comp_path)
        write_tsv(comp_df, comp_path)

    # Dynamics features
    if dynamics_df is not None and not dynamics_df.empty:
        # direct_blocks.append(dynamics_df)  # Assuming trial-wise
        # Wait, extract_precomputed_features returns trial-wise? Yes.
        direct_blocks.append(dynamics_df)
        if dynamics_cols:
            if len(dynamics_cols) == len(dynamics_df.columns):
                dynamics_df.columns = dynamics_cols
        direct_cols.extend(dynamics_df.columns)
        dyn_path = features_dir / "features_dynamics.tsv"
        logger.info("Saving dynamics features: %s", dyn_path)
        write_tsv(dynamics_df, dyn_path)

    # CFC features
    if cfc_df is not None and not cfc_df.empty:
        direct_blocks.append(cfc_df)
        if cfc_cols:
            if len(cfc_cols) == len(cfc_df.columns):
                cfc_df.columns = cfc_cols
        direct_cols.extend(cfc_df.columns)
        cfc_path = features_dir / "features_cfc.tsv"
        logger.info("Saving CFC features: %s", cfc_path)
        write_tsv(cfc_df, cfc_path)

    # Precomputed features
    if precomputed_df is not None and not precomputed_df.empty:
        if precomputed_cols:
            if len(precomputed_cols) == len(precomputed_df.columns):
                precomputed_df.columns = precomputed_cols
            else:
                logger.warning(
                    "Precomputed column mismatch: %d names vs %d columns. Using DataFrame names.",
                    len(precomputed_cols),
                    len(precomputed_df.columns),
                )
        precomputed_path = features_dir / "features_precomputed.tsv"
        precomputed_cols_path = features_dir / "features_precomputed_columns.tsv"
        logger.info("Saving precomputed features: %s", precomputed_path)
        write_tsv(precomputed_df, precomputed_path)
        write_tsv(pd.Series(list(precomputed_df.columns), name="feature").to_frame(), precomputed_cols_path)

    if aper_qc and aper_qc.get("freqs") is not None and aper_qc.get("slopes") is not None and aper_qc.get("offsets") is not None and aper_qc.get("r2") is not None:
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
        except (OSError, IOError, TypeError, KeyError) as exc:
            logger.warning("Failed to save aperiodic QC npz: %s", exc)
    elif aper_qc:
        logger.info("Aperiodic QC payload present but incomplete; skipping aperiodic_qc.npz")

    if direct_blocks:
        direct_df = pd.concat(direct_blocks, axis=1)
    else:
        direct_df = pd.DataFrame()
        logger.info("No direct feature blocks to concatenate (connectivity/precomputed-only run)")

    if not direct_df.empty:
        eeg_direct_path = features_dir / "features_eeg_direct.tsv"
        eeg_direct_cols_path = features_dir / "features_eeg_direct_columns.tsv"
        logger.info("Saving direct EEG features: %s", eeg_direct_path)
        write_tsv(direct_df, eeg_direct_path)
        write_tsv(pd.Series(direct_cols, name="feature").to_frame(), eeg_direct_cols_path)

    if plateau_df is not None and not plateau_df.empty:
        plateau_path = features_dir / "features_eeg_plateau.tsv"
        plateau_cols_path = features_dir / "features_eeg_plateau_columns.tsv"
        logger.info("Saving plateau-averaged EEG features: %s", plateau_path)
        write_tsv(plateau_df, plateau_path)
        write_tsv(pd.Series(plateau_cols or [], name="feature").to_frame(), plateau_cols_path)

    if conn_df is not None and not conn_df.empty:
        if conn_cols:
            if len(conn_cols) == len(conn_df.columns):
                conn_df.columns = conn_cols
            else:
                logger.warning("Connectivity column mismatch: %d names vs %d columns. Using DataFrame names.", len(conn_cols), len(conn_df.columns))

        conn_path = features_dir / "features_connectivity.tsv"
        logger.info("Saving connectivity features: %s", conn_path)
        write_tsv(conn_df, conn_path)

    blocks = []
    cols_all = []
    if not direct_df.empty:
        blocks.append(direct_df)
        cols_all.extend(direct_df.columns)
    if conn_df is not None and not conn_df.empty:
        blocks.append(conn_df)
        cols_all.extend(conn_df.columns)
    if precomputed_df is not None and not precomputed_df.empty:
        blocks.append(precomputed_df)
        cols_all.extend(precomputed_df.columns)

    if blocks:
        combined_df = pd.concat(blocks, axis=1)
    else:
        combined_df = pd.DataFrame()
        logger.warning("No feature blocks available for combined output")

    combined_path = features_dir / "features_all.tsv"
    logger.info("Saving combined features: %s", combined_path)
    write_tsv(combined_df, combined_path)

    # --- Generate JSON Sidecar ---
    try:
        subject_str = features_dir.parts[-3].replace("sub-", "") if len(features_dir.parts) > 3 else "unknown"
        sidecar_path = features_dir / "features.json"
        manifest = generate_manifest(
            feature_columns=list(combined_df.columns),
            config=config,
            subject=subject_str,
            task=config.get("project.task") if config is not None else None,
            qc=feature_qc,
        )
        with open(sidecar_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved feature metadata sidecar: %s", sidecar_path)
    except Exception as e:
        logger.warning(f"Failed to generate feature sidecar: {e}")

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
    onset = onset.ffill().bfill().fillna(0.0).astype(float)
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
    except (OSError, IOError, ValueError, KeyError) as exc:
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
        except (OSError, IOError, ValueError, KeyError) as exc:
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


def iterate_feature_columns(
    feature_df: pd.DataFrame,
    col_prefix: Optional[str] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Get filtered feature columns by prefix."""
    if feature_df is None or feature_df.empty:
        return [], pd.DataFrame()
    if col_prefix:
        cols = [c for c in feature_df.columns if str(c).startswith(col_prefix)]
        return (cols, feature_df[cols]) if cols else ([], pd.DataFrame())
    return list(feature_df.columns), feature_df


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
