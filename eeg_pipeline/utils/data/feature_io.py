"""
Feature I/O Utilities.

Consolidated module for loading and saving feature data including:
- Feature bundle loading (power, microstates, connectivity, etc.)
- Feature saving and export functions
- fMRI regressor exports
- Microstate template management
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import glob
import json
import logging
import time

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.formatting import sanitize_label
from eeg_pipeline.infra.paths import (
    deriv_features_path,
    deriv_stats_path,
    ensure_dir,
    find_connectivity_features_path,
)
from eeg_pipeline.infra.tsv import read_table, read_tsv, write_parquet, write_tsv

from .features import infer_power_band
from .manipulation import build_plateau_features


###################################################################
# READING UTILITIES
###################################################################

def _safe_read_table(
    path: Path,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return read_table(path)
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def load_subject_features(
    subjects: List[str],
    deriv_root: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if not subjects:
        return {}

    subject_features: Dict[str, pd.DataFrame] = {}
    for subject in subjects:
        feature_path = deriv_features_path(deriv_root, subject) / "features_power.tsv"
        if not feature_path.exists():
            logger.warning(f"Missing features for sub-{subject}: {feature_path}")
            continue
        subject_features[subject] = read_table(feature_path)

    return subject_features


def _load_features_and_targets(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    epochs: Optional[Any] = None,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], pd.Series, Any]:
    feats_dir = deriv_features_path(deriv_root, subject)
    temporal_path = feats_dir / "features_power.tsv"
    plateau_path = feats_dir / "features_power_plateau.tsv"
    conn_path = find_connectivity_features_path(deriv_root, subject)
    y_path = feats_dir / "target_vas_ratings.tsv"

    power_path = plateau_path if plateau_path.exists() else temporal_path
    if not power_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing features or targets for sub-{subject}. Expected at {feats_dir}")

    temporal_df = read_table(temporal_path) if temporal_path.exists() else None
    plateau_df = read_table(power_path)
    conn_df = read_table(conn_path) if conn_path.exists() else None
    y_df = read_table(y_path)

    if y_df.shape[1] == 1:
        y = pd.to_numeric(y_df.iloc[:, 0], errors="coerce")
    else:
        numeric_cols = y_df.select_dtypes(exclude=["object"]).columns
        if len(numeric_cols) == 0:
            raise ValueError(f"No numeric target columns found in {y_path}")
        y = pd.to_numeric(y_df[numeric_cols[0]], errors="coerce")

    if epochs is None:
        epochs, _ = load_epochs_for_analysis(
            subject,
            task,
            align="strict",
            preload=False,
            deriv_root=deriv_root,
            bids_root=getattr(config, "bids_root", None),
            config=config,
        )
        if epochs is None:
            raise FileNotFoundError(f"Could not locate clean epochs for sub-{subject}, task-{task}")

    n_samples = len(y)
    if len(plateau_df) != n_samples:
        raise ValueError(
            f"Length mismatch: plateau features ({len(plateau_df)} rows) != target ratings ({n_samples} rows) "
            f"for sub-{subject}, task-{task}"
        )

    if temporal_df is not None and len(temporal_df) != n_samples:
        raise ValueError(
            f"Length mismatch: temporal features ({len(temporal_df)} rows) != target ratings ({n_samples} rows) "
            f"for sub-{subject}, task-{task}"
        )

    if conn_df is not None and len(conn_df) != n_samples:
        raise ValueError(
            f"Length mismatch: connectivity features ({len(conn_df)} rows) != target ratings ({n_samples} rows) "
            f"for sub-{subject}, task-{task}"
        )

    return temporal_df, plateau_df, conn_df, y, getattr(epochs, "info", None)





@dataclass
class FeatureBundle:
    """Unified container for all feature tables loaded for a subject."""

    power_df: Optional[pd.DataFrame] = None
    microstate_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    pac_df: Optional[pd.DataFrame] = None
    pac_trials_df: Optional[pd.DataFrame] = None
    pac_time_df: Optional[pd.DataFrame] = None
    itpc_df: Optional[pd.DataFrame] = None
    complexity_df: Optional[pd.DataFrame] = None
    dynamics_df: Optional[pd.DataFrame] = None
    all_features_df: Optional[pd.DataFrame] = None
    targets: Optional[pd.Series] = None

    @property
    def n_trials(self) -> int:
        if self.power_df is not None:
            return len(self.power_df)
        return 0

    @property
    def empty(self) -> bool:
        return self.power_df is None


def load_feature_bundle(
    subject: str,
    deriv_root: Path,
    logger: Optional[logging.Logger] = None,
    include_targets: bool = False,
    config: Optional[Any] = None,
) -> FeatureBundle:
    """Canonical loader for all feature tables for a subject."""
    if logger is None:
        logger = logging.getLogger(__name__)

    features_dir = deriv_features_path(deriv_root, subject)

    bundle = FeatureBundle(
        power_df=_safe_read_table(features_dir / "features_power.tsv", logger),
        microstate_df=_safe_read_table(features_dir / "features_microstates.tsv", logger),
        connectivity_df=_safe_read_table(find_connectivity_features_path(deriv_root, subject), logger),
        aperiodic_df=_safe_read_table(features_dir / "features_aperiodic.tsv", logger),
        pac_df=_safe_read_table(features_dir / "features_pac.tsv", logger),
        pac_trials_df=_safe_read_table(features_dir / "features_pac_trials.tsv", logger),
        pac_time_df=_safe_read_table(features_dir / "features_pac_time.tsv", logger),
        itpc_df=_safe_read_table(features_dir / "features_itpc.tsv", logger),
        complexity_df=_safe_read_table(features_dir / "features_complexity.tsv", logger),
        dynamics_df=_safe_read_table(features_dir / "features_dynamics.tsv", logger),
        all_features_df=_safe_read_table(features_dir / "features_all.tsv", logger),
    )

    if include_targets:
        targets_df = _safe_read_table(features_dir / "target_vas_ratings.tsv", logger)
        if targets_df is not None:
            if targets_df.shape[1] == 1:
                bundle.targets = pd.to_numeric(targets_df.iloc[:, 0], errors="coerce")
            else:
                constants = {
                    "TARGET_COLUMNS": (
                        config.get("event_columns.rating", [])
                        if config is not None and hasattr(config, "get")
                        else []
                    )
                }
                target_col = pick_target_column(targets_df, constants=constants)
                if target_col is None:
                    numeric_cols = targets_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) == 0:
                        raise ValueError(
                            "No numeric target columns found in target_vas_ratings.tsv. "
                            f"Available columns: {list(targets_df.columns)}"
                        )
                    if len(numeric_cols) > 1:
                        logger.warning(
                            "Multiple numeric target columns found in target_vas_ratings.tsv; using '%s'. Candidates=%s",
                            str(numeric_cols[0]),
                            ",".join(str(c) for c in numeric_cols),
                        )
                    target_col = str(numeric_cols[0])
                bundle.targets = pd.to_numeric(targets_df[target_col], errors="coerce")

    return bundle


def load_feature_dfs_for_subjects(
    subjects: List[str],
    deriv_root: Path,
    input_filename_key: str,
    logger: logging.Logger,
    config,
) -> pd.DataFrame:
    input_filename = config.get(input_filename_key)
    if not input_filename:
        logger.warning(f"Input filename not found in config: {input_filename_key}")
        return pd.DataFrame()

    all_dfs = []
    for subject in subjects:
        features_dir = deriv_features_path(deriv_root, subject)
        file_path = features_dir / input_filename

        if not file_path.exists():
            logger.debug(f"Features not found for sub-{subject} at {file_path}")
            continue

        try:
            df = read_tsv(file_path)
            if df.empty:
                continue
            df.insert(0, "subject", subject)
            all_dfs.append(df)
        except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
            logger.warning(f"Failed to read features for sub-{subject} at {file_path}: {exc}")
            continue

    if not all_dfs:
        logger.warning(f"No features found for {input_filename_key} across any subject")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


###################################################################
# SAVING UTILITIES
###################################################################


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
    comp_df: Optional[pd.DataFrame] = None,
    comp_cols: Optional[List[str]] = None,
    spectral_df: Optional[pd.DataFrame] = None,
    spectral_cols: Optional[List[str]] = None,
    feature_qc: Optional[Dict[str, Any]] = None,
    export_all: bool = True,
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    from eeg_pipeline.domain.features.naming import generate_manifest

    def _dedupe_identical_duplicate_columns(
        df: pd.DataFrame,
        *,
        df_label: str,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        dup_mask = df.columns.duplicated(keep=False)
        if not bool(np.any(dup_mask)):
            return df

        dup_names = pd.Index(df.columns[dup_mask]).unique().tolist()
        cols_to_drop: List[int] = []
        for col_name in dup_names:
            idxs = [i for i, c in enumerate(df.columns) if c == col_name]
            if len(idxs) <= 1:
                continue

            base = df.iloc[:, idxs[0]]
            for idx in idxs[1:]:
                other = df.iloc[:, idx]

                if base.equals(other):
                    cols_to_drop.append(idx)
                    continue

                try:
                    base_num = pd.to_numeric(base, errors="coerce")
                    other_num = pd.to_numeric(other, errors="coerce")
                except Exception:
                    base_num = base
                    other_num = other

                if (
                    isinstance(base_num, pd.Series)
                    and isinstance(other_num, pd.Series)
                    and base_num.equals(other_num)
                ):
                    cols_to_drop.append(idx)
                    continue

                raise ValueError(
                    "Duplicate feature column name with non-identical values detected while building "
                    f"{df_label}: '{col_name}'. This indicates the same feature was computed more than once "
                    "with conflicting values; aborting to preserve scientific validity."
                )

        if not cols_to_drop:
            return df

        drop_mask = np.ones(df.shape[1], dtype=bool)
        drop_mask[np.array(cols_to_drop, dtype=int)] = False
        logger.warning(
            "Dropping %d duplicate feature columns with identical values while building %s. Examples=%s",
            len(cols_to_drop),
            df_label,
            ",".join(str(c) for c in dup_names[:5]),
        )
        return df.iloc[:, drop_mask]

    direct_blocks = []
    direct_cols: List[str] = []

    if pow_df is not None:
        direct_blocks.append(pow_df)
        if len(pow_cols) == len(pow_df.columns):
            pow_df.columns = pow_cols
        else:
            logger.warning(
                "Power column mismatch: %d names vs %d columns. Using DataFrame names.",
                len(pow_cols),
                len(pow_df.columns),
            )
        direct_cols.extend(list(pow_df.columns))

    if baseline_df is not None and not baseline_df.empty:
        direct_blocks.append(baseline_df)
        if baseline_cols:
            if len(baseline_cols) == len(baseline_df.columns):
                baseline_df.columns = baseline_cols
            else:
                logger.warning(
                    "Baseline column mismatch: %d names vs %d columns. Using DataFrame names.",
                    len(baseline_cols),
                    len(baseline_df.columns),
                )
        direct_cols.extend(list(baseline_df.columns))

    if ms_df is not None and not ms_df.empty:
        direct_blocks.append(ms_df)
        if ms_cols:
            if len(ms_cols) == len(ms_df.columns):
                ms_df.columns = ms_cols
            else:
                logger.warning(
                    "Microstate column mismatch: %d names vs %d columns. Using DataFrame names.",
                    len(ms_cols),
                    len(ms_df.columns),
                )
        direct_cols.extend(list(ms_df.columns))
        ms_name = f"features_microstates_{suffix}.tsv" if suffix else "features_microstates.tsv"
        ms_path = features_dir / ms_name
        logger.info("Saving microstate features: %s", ms_path)
        write_tsv(ms_df, ms_path)

    if aper_df is not None and not aper_df.empty:
        direct_blocks.append(aper_df)
        if aper_cols:
            if len(aper_cols) == len(aper_df.columns):
                aper_df.columns = aper_cols
            else:
                logger.warning(
                    "Aperiodic column mismatch: %d names vs %d columns. Using DataFrame names.",
                    len(aper_cols),
                    len(aper_df.columns),
                )
        direct_cols.extend(list(aper_df.columns))
        aper_name = f"features_aperiodic_{suffix}.tsv" if suffix else "features_aperiodic.tsv"
        aper_path = features_dir / aper_name
        logger.info("Saving aperiodic features: %s", aper_path)
        write_tsv(aper_df, aper_path)

    if itpc_df is not None and not itpc_df.empty:
        if itpc_cols and len(itpc_cols) == len(itpc_df.columns):
            itpc_df.columns = itpc_cols
        direct_blocks.append(itpc_df)
        direct_cols.extend(list(itpc_df.columns))
        itpc_name = f"features_itpc_{suffix}.tsv" if suffix else "features_itpc.tsv"
        itpc_path = features_dir / itpc_name
        logger.info("Saving ITPC features (channel x band x segment): %s", itpc_path)
        write_tsv(itpc_df, itpc_path)

    if pac_df is not None and not pac_df.empty:
        pac_name = f"features_pac_{suffix}.tsv" if suffix else "features_pac.tsv"
        pac_path = features_dir / pac_name
        logger.info("Saving PAC comodulograms: %s", pac_path)
        write_tsv(pac_df, pac_path)

    if pac_trials_df is not None and not pac_trials_df.empty:
        pac_trials_name = f"features_pac_trials_{suffix}.tsv" if suffix else "features_pac_trials.tsv"
        pac_trials_path = features_dir / pac_trials_name
        logger.info("Saving PAC per-trial values: %s", pac_trials_path)
        write_tsv(pac_trials_df, pac_trials_path)
        direct_blocks.append(pac_trials_df)
        direct_cols.extend(list(pac_trials_df.columns))

    if pac_time_df is not None and not pac_time_df.empty:
        pac_time_name = f"features_pac_time_{suffix}.tsv" if suffix else "features_pac_time.tsv"
        pac_time_path = features_dir / pac_time_name
        logger.info("Saving PAC time-resolved values: %s", pac_time_path)
        write_tsv(pac_time_df, pac_time_path)

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
            logger.warning(
                "Failed to evaluate PAC time alignment; excluding from features_all.tsv: %s",
                exc,
            )

    if comp_df is not None and not comp_df.empty:
        direct_blocks.append(comp_df)
        if comp_cols:
            if len(comp_cols) == len(comp_df.columns):
                comp_df.columns = comp_cols
        direct_cols.extend(list(comp_df.columns))
        comp_name = f"features_complexity_{suffix}.tsv" if suffix else "features_complexity.tsv"
        comp_path = features_dir / comp_name
        logger.info("Saving complexity features: %s", comp_path)
        write_tsv(comp_df, comp_path)

    if spectral_df is not None and not spectral_df.empty:
        direct_blocks.append(spectral_df)
        if spectral_cols:
            if len(spectral_cols) == len(spectral_df.columns):
                spectral_df.columns = spectral_cols
        direct_cols.extend(list(spectral_df.columns))
        spec_name = f"features_spectral_{suffix}.tsv" if suffix else "features_spectral.tsv"
        spec_path = features_dir / spec_name
        logger.info("Saving spectral features (IAF): %s", spec_path)
        write_tsv(spectral_df, spec_path)


    if aper_qc and aper_qc.get("freqs") is not None and aper_qc.get("slopes") is not None and aper_qc.get("offsets") is not None and aper_qc.get("r2") is not None:
        try:
            subject_name = features_dir.parent.parent.name.replace("sub-", "")
            deriv_root = features_dir.parent.parent.parent
            stats_dir = deriv_stats_path(deriv_root, subject_name)
            stats_dir.mkdir(parents=True, exist_ok=True)
            qc_name = f"aperiodic_qc_{suffix}.npz" if suffix else "aperiodic_qc.npz"
            qc_path = stats_dir / qc_name
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
        direct_df = _dedupe_identical_duplicate_columns(direct_df, df_label="features_power.tsv")
    else:
        direct_df = pd.DataFrame()
        logger.info("No direct feature blocks to concatenate (connectivity/precomputed-only run)")

    if not direct_df.empty:
        direct_name = f"features_power_{suffix}.tsv" if suffix else "features_power.tsv"
        cols_name = f"features_power_columns_{suffix}.tsv" if suffix else "features_power_columns.tsv"
        eeg_direct_path = features_dir / direct_name
        eeg_direct_cols_path = features_dir / cols_name
        direct_cols = list(direct_df.columns)
        logger.info("Saving power features: %s", eeg_direct_path)
        write_tsv(direct_df, eeg_direct_path)
        write_tsv(pd.Series(direct_cols, name="feature").to_frame(), eeg_direct_cols_path)

    if plateau_df is not None and not plateau_df.empty:
        plateau_name = f"features_power_plateau_{suffix}.tsv" if suffix else "features_power_plateau.tsv"
        p_cols_name = f"features_power_plateau_columns_{suffix}.tsv" if suffix else "features_power_plateau_columns.tsv"
        plateau_path = features_dir / plateau_name
        plateau_cols_path = features_dir / p_cols_name
        logger.info("Saving plateau-averaged EEG features: %s", plateau_path)
        write_tsv(plateau_df, plateau_path)
        write_tsv(pd.Series(plateau_cols or [], name="feature").to_frame(), plateau_cols_path)

    if conn_df is not None and not conn_df.empty:
        if conn_cols:
            if len(conn_cols) == len(conn_df.columns):
                conn_df.columns = conn_cols
            else:
                logger.warning(
                    "Connectivity column mismatch: %d names vs %d columns. Using DataFrame names.",
                    len(conn_cols),
                    len(conn_df.columns),
                )

        conn_name = f"features_connectivity_{suffix}.parquet" if suffix else "features_connectivity.parquet"
        conn_path = features_dir / conn_name
        logger.info("Saving connectivity features: %s", conn_path)
        t0 = time.perf_counter()
        write_parquet(conn_df, conn_path)
        logger.info(
            "Saved connectivity Parquet in %.2fs (rows=%d, cols=%d)",
            time.perf_counter() - t0,
            len(conn_df),
            len(conn_df.columns),
        )

    blocks = []
    cols_all: List[str] = []
    if not direct_df.empty:
        blocks.append(direct_df)
        cols_all.extend(list(direct_df.columns))

    if blocks:
        combined_df = pd.concat(blocks, axis=1)
        combined_df = _dedupe_identical_duplicate_columns(combined_df, df_label="features_all.tsv")
    else:
        combined_df = pd.DataFrame()
        logger.warning("No feature blocks available for combined output")

    if not export_all:
        logger.debug("Skipping features_all.tsv creation (export_all=False)")
    else:
        all_name = f"features_all_{suffix}.tsv" if suffix else "features_all.tsv"
        combined_path = features_dir / all_name
        logger.info("Saving combined features: %s", combined_path)
        t0 = time.perf_counter()
        write_tsv(combined_df, combined_path)
        logger.info(
            "Saved combined TSV in %.2fs (rows=%d, cols=%d)",
            time.perf_counter() - t0,
            len(combined_df),
            len(combined_df.columns),
        )

    try:
        subject_str = features_dir.parts[-3].replace("sub-", "") if len(features_dir.parts) > 3 else "unknown"
        json_name = f"features_{suffix}.json" if suffix else "features.json"
        sidecar_path = features_dir / json_name
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
    except (OSError, IOError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Failed to generate feature sidecar: %s", exc)

    y_path = features_dir / "target_vas_ratings.tsv"
    rating_columns = config.get("event_columns.rating", ["vas_rating"])
    target_column_name = rating_columns[0] if rating_columns else "vas_rating"
    logger.info("Saving behavioral target vector: %s (column: %s)", y_path, target_column_name)
    write_tsv(y.to_frame(name=target_column_name), y_path)

    return combined_df


def _extract_onset_duration(
    aligned_events: pd.DataFrame,
    logger: logging.Logger,
    n_fallback: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    if aligned_events is None or aligned_events.empty:
        logger.warning("Cannot export fMRI regressors: aligned_events is empty.")
        return None

    n_trials = len(aligned_events)
    onset, duration = _extract_onset_duration(aligned_events, logger, n_trials)
    rating = pd.to_numeric(y, errors="coerce") if y is not None else pd.Series(np.full(n_trials, np.nan))

    reg_df = pd.DataFrame({"onset": onset, "duration": duration, "rating": rating})
    reg_values: Dict[str, pd.Series] = {}

    def _add_regressor(name: str, series: Any) -> None:
        if series is None:
            return
        ser = pd.to_numeric(series, errors="coerce")
        if len(ser) != n_trials:
            logger.warning(
                "Regressor %s length mismatch (got %d, expected %d); skipping.",
                name,
                len(ser),
                n_trials,
            )
            return
        reg_df[name] = ser
        reg_values[name] = ser

    for band in power_bands:
        band_cols = [
            c
            for c in plateau_cols
            if infer_power_band(c, bands=power_bands) == band
        ]
        if not band_cols:
            continue
        _add_regressor(f"pow_{band}_mean", plateau_df[band_cols].mean(axis=1))

    if ms_df is not None and not ms_df.empty:
        for col in ms_df.columns:
            col_str = str(col)
            if col_str.startswith("ms_coverage_") or col_str.startswith("ms_duration_"):
                _add_regressor(col_str, ms_df[col])
                continue
            if col_str.startswith("microstates_") and ("_coverage_state" in col_str or "_duration_state" in col_str):
                _add_regressor(col_str, ms_df[col])

    if aper_df is not None and not aper_df.empty:
        slope_cols_legacy = [c for c in aper_df.columns if str(c).startswith("aper_slope_")]
        if slope_cols_legacy:
            _add_regressor("aper_slope_mean", aper_df[slope_cols_legacy].mean(axis=1))

        slope_cols_v2 = [
            c
            for c in aper_df.columns
            if str(c).startswith("aperiodic_") and str(c).endswith("_slope") and "_ch_" in str(c)
        ]
        if slope_cols_v2:
            _add_regressor("aperiodic_slope_mean", aper_df[slope_cols_v2].mean(axis=1))

        offset_cols_v2 = [
            c
            for c in aper_df.columns
            if str(c).startswith("aperiodic_") and str(c).endswith("_offset") and "_ch_" in str(c)
        ]
        if offset_cols_v2:
            _add_regressor("aperiodic_offset_mean", aper_df[offset_cols_v2].mean(axis=1))

    if (
        pac_trials_df is not None
        and not pac_trials_df.empty
        and "trial" in pac_trials_df.columns
        and "pac" in pac_trials_df.columns
    ):
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
        reg_out = pd.DataFrame({"onset": onset, "duration": duration, "amplitude": series})
        write_tsv(reg_out, reg_path)

    logger.info("Saved %d fMRI regressors to %s", len(reg_values), reg_dir)
    return reg_df


###################################################################
# MICROSTATE TEMPLATE MANAGEMENT
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

    np.savez_compressed(template_path, templates=templates, ch_names=np.array(ch_names), n_states=n_states)
    logger.info("Saved microstate templates to %s", template_path)


def load_group_microstate_templates(
    deriv_root: Path,
    n_states: int,
    logger: logging.Logger,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    group_path = (
        deriv_root / "group" / "eeg" / "stats" / f"microstates_templates_group_K{n_states}.npz"
    )
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
        logger.warning(
            "No common channels across microstate templates; cannot build group template"
        )
        return None, None

    aligned_templates = []
    for templ, chs in templates_list:
        indices = [chs.index(ch) for ch in common_chs]
        aligned_templates.append(templ[:, indices])

    stacked = np.stack(aligned_templates, axis=0)
    group_templates = np.nanmean(stacked, axis=0)

    out_dir = deriv_root / "group" / "eeg" / "stats"
    ensure_dir(out_dir)
    out_path = out_dir / f"microstates_templates_group_K{n_states}.npz"
    np.savez_compressed(out_path, templates=group_templates, ch_names=np.array(common_chs), n_states=n_states)
    logger.info("Saved group microstate templates to %s", out_path)

    return group_templates, common_chs


###################################################################
# TRIAL MANAGEMENT
###################################################################

def save_trial_alignment_manifest(
    aligned_events: pd.DataFrame,
    epochs: mne.Epochs,
    manifest_path: Path,
    config,
    logger: logging.Logger,
) -> None:
    manifest: Dict[str, Any] = {
        "n_epochs": int(len(epochs)) if epochs is not None else None,
        "n_events": int(len(aligned_events)) if aligned_events is not None else None,
        "epoch_times": {
            "tmin": float(epochs.tmin) if epochs is not None else None,
            "tmax": float(epochs.tmax) if epochs is not None else None,
        },
        "config_task": config.get("project.task") if config is not None else None,
    }

    if aligned_events is not None:
        for col in ["trial_type", "condition", "run", "block"]:
            if col in aligned_events.columns:
                manifest[f"events_{col}"] = aligned_events[col].astype(str).tolist()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def save_dropped_trials_log(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    drop_log_path: Path,
    logger: logging.Logger,
) -> None:
    if epochs is None or events_df is None:
        return

    kept = getattr(epochs, "selection", None)
    if kept is None:
        logger.info("Epoch selection missing; cannot reconstruct dropped trials.")
        return

    drop_log_path.parent.mkdir(parents=True, exist_ok=True)

    dropped = np.setdiff1d(np.arange(len(events_df)), kept)
    if dropped.size == 0:
        return

    out = pd.DataFrame({"original_index": dropped})
    write_tsv(out, drop_log_path)


def iterate_feature_columns(
    feature_df: pd.DataFrame,
    col_prefix: Optional[str] = None,
) -> Tuple[List[str], pd.DataFrame]:
    if feature_df is None or feature_df.empty:
        return [], pd.DataFrame()
    if col_prefix:
        cols = [c for c in feature_df.columns if str(c).startswith(col_prefix)]
        if not cols:
            return [], pd.DataFrame()
        return cols, feature_df[cols]
    return list(feature_df.columns), feature_df


def combine_all_features(features_dir: Path, config=None) -> Optional[pd.DataFrame]:
    """
    Combine all individual feature files into features_all.tsv.
    
    This utility loads all feature TSV files from a subject's features directory
    and merges them into a single combined file.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    features_dir = Path(features_dir)
    if not features_dir.exists():
        logger.warning("Features directory does not exist: %s", features_dir)
        return None
    
    blocks = []
    
    # Load direct EEG features
    direct_path = features_dir / "features_power.tsv"
    if direct_path.exists():
        df = _safe_read_table(direct_path, logger)
        if df is not None and not df.empty:
            blocks.append(df)
    
    
    if not blocks:
        logger.warning("No feature files found to combine in %s", features_dir)
        return None
    
    # Combine all blocks
    combined_df = pd.concat(blocks, axis=1)
    
    # Remove exact duplicate columns
    dup_mask = combined_df.columns.duplicated(keep="first")
    if dup_mask.any():
        logger.info("Removing %d duplicate columns", dup_mask.sum())
        combined_df = combined_df.loc[:, ~dup_mask]
    
    # Save combined file
    combined_path = features_dir / "features_all.tsv"
    write_tsv(combined_df, combined_path)
    logger.info("Saved combined features: %s (rows=%d, cols=%d)", 
                combined_path, len(combined_df), len(combined_df.columns))
    
    return combined_df


__all__ = [
    # Reading
    "FeatureBundle",
    "load_feature_bundle",
    "load_feature_dfs_for_subjects",
    "load_subject_features",
    "_load_features_and_targets",
    # Saving
    "build_plateau_features",
    "combine_all_features",
    "compute_group_microstate_templates",
    "export_fmri_regressors",
    "iterate_feature_columns",
    "load_group_microstate_templates",
    "save_all_features",
    "save_dropped_trials_log",
    "save_microstate_templates",
    "save_trial_alignment_manifest",
]
