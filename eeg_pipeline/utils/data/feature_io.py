"""
Feature I/O Utilities.

Consolidated module for loading and saving feature data including:
- Feature bundle loading (power, connectivity, etc.)
- Feature saving and export functions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import time

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.infra.paths import (
    deriv_features_path,
    deriv_stats_path,
    find_connectivity_features_path,
)
from eeg_pipeline.infra.tsv import read_table, read_tsv, write_tsv


###################################################################
# READING UTILITIES
###################################################################


def _safe_read_table(
    path: Path,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Safely read a table file, returning None if missing or invalid."""
    if not path.exists():
        return None
    try:
        from eeg_pipeline.infra.tsv import read_table
        return read_table(path)
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _safe_read_feature_table(
    features_dir: Path,
    base_name: str,
    logger: logging.Logger,
    extension: str = ".tsv",
) -> Optional[pd.DataFrame]:
    """Read feature table, checking subfolder first."""
    folder = _get_folder_for_feature(base_name)
    filename = f"{base_name}{extension}"
    
    if folder:
        subfolder_path = features_dir / folder / filename
        if subfolder_path.exists():
            return _safe_read_table(subfolder_path, logger)
            
    return _safe_read_table(features_dir / filename, logger)


def load_subject_features(
    subjects: List[str],
    deriv_root: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """Load power features for multiple subjects."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if not subjects:
        return {}

    subject_features: Dict[str, pd.DataFrame] = {}
    for subject in subjects:
        features_dir = deriv_features_path(deriv_root, subject)
        feature_path = features_dir / "power" / "features_power.tsv"
        if not feature_path.exists():
            feature_path = features_dir / "features_power.tsv"
        if not feature_path.exists():
            logger.warning("Missing features for sub-%s: %s", subject, feature_path)
            continue
        subject_features[subject] = read_table(feature_path)

    return subject_features


def _extract_target_series(target_df: pd.DataFrame, target_path: Path) -> pd.Series:
    """Extract target series from target dataframe."""
    if target_df.shape[1] == 1:
        return pd.to_numeric(target_df.iloc[:, 0], errors="coerce")
    
    numeric_cols = target_df.select_dtypes(exclude=["object"]).columns
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric target columns found in {target_path}")
    return pd.to_numeric(target_df[numeric_cols[0]], errors="coerce")


def _validate_feature_lengths(
    subject: str,
    task: str,
    target_length: int,
    active_df: pd.DataFrame,
    temporal_df: Optional[pd.DataFrame] = None,
    conn_df: Optional[pd.DataFrame] = None,
) -> None:
    """Validate that all feature dataframes have matching lengths."""
    if len(active_df) != target_length:
        raise ValueError(
            f"Length mismatch: active features ({len(active_df)} rows) != target ratings "
            f"({target_length} rows) for sub-{subject}, task-{task}"
        )

    if temporal_df is not None and len(temporal_df) != target_length:
        raise ValueError(
            f"Length mismatch: temporal features ({len(temporal_df)} rows) != target ratings "
            f"({target_length} rows) for sub-{subject}, task-{task}"
        )

    if conn_df is not None and len(conn_df) != target_length:
        raise ValueError(
            f"Length mismatch: connectivity features ({len(conn_df)} rows) != target ratings "
            f"({target_length} rows) for sub-{subject}, task-{task}"
        )


def _load_features_and_targets(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    epochs: Optional[Any] = None,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], pd.Series, Any]:
    """Load features and targets for a subject, validating alignment."""
    feats_dir = deriv_features_path(deriv_root, subject)
    # Try new organized paths first then fallback to root
    temporal_path = feats_dir / "power" / "features_power.tsv"
    if not temporal_path.exists():
        temporal_path = feats_dir / "features_power.tsv"
        
    active_path = feats_dir / "power" / "features_power_active.tsv"
    if not active_path.exists():
        active_path = feats_dir / "features_power_active.tsv"
        
    conn_path = find_connectivity_features_path(deriv_root, subject)
    
    target_path = feats_dir / "behavior" / "target_vas_ratings.tsv"
    if not target_path.exists():
        target_path = feats_dir / "target_vas_ratings.tsv"

    power_path = active_path if active_path.exists() else temporal_path
    if not power_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            f"Missing features or targets for sub-{subject}. Expected at {feats_dir}"
        )

    temporal_df = read_table(temporal_path) if temporal_path.exists() else None
    active_df = read_table(power_path)
    conn_df = read_table(conn_path) if conn_path.exists() else None
    target_df = read_table(target_path)

    target_series = _extract_target_series(target_df, target_path)

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
            raise FileNotFoundError(
                f"Could not locate clean epochs for sub-{subject}, task-{task}"
            )

    _validate_feature_lengths(
        subject, task, len(target_series), active_df, temporal_df, conn_df
    )

    return temporal_df, active_df, conn_df, target_series, getattr(epochs, "info", None)





@dataclass
class FeatureBundle:
    """Unified container for all feature tables loaded for a subject."""

    power_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    erp_df: Optional[pd.DataFrame] = None
    pac_df: Optional[pd.DataFrame] = None
    pac_trials_df: Optional[pd.DataFrame] = None
    pac_time_df: Optional[pd.DataFrame] = None
    itpc_df: Optional[pd.DataFrame] = None
    complexity_df: Optional[pd.DataFrame] = None
    bursts_df: Optional[pd.DataFrame] = None
    quality_df: Optional[pd.DataFrame] = None
    erds_df: Optional[pd.DataFrame] = None
    spectral_df: Optional[pd.DataFrame] = None
    ratios_df: Optional[pd.DataFrame] = None
    asymmetry_df: Optional[pd.DataFrame] = None
    temporal_df: Optional[pd.DataFrame] = None
    targets: Optional[pd.Series] = None

    @property
    def n_trials(self) -> int:
        if self.power_df is not None:
            return len(self.power_df)
        return 0

    @property
    def empty(self) -> bool:
        return self.power_df is None


def _extract_targets_from_dataframe(
    targets_df: pd.DataFrame,
    config: Optional[Any],
    logger: logging.Logger,
) -> pd.Series:
    """Extract target series from targets dataframe with config-aware column selection."""
    if targets_df.shape[1] == 1:
        return pd.to_numeric(targets_df.iloc[:, 0], errors="coerce")
    
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
    
    return pd.to_numeric(targets_df[target_col], errors="coerce")


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
        power_df=_safe_read_feature_table(features_dir, "features_power", logger),
        connectivity_df=_safe_read_table(
            find_connectivity_features_path(deriv_root, subject), logger
        ),
        aperiodic_df=_safe_read_feature_table(features_dir, "features_aperiodic", logger),
        erp_df=_safe_read_feature_table(features_dir, "features_erp", logger),
        pac_df=_safe_read_feature_table(features_dir, "features_pac", logger),
        pac_trials_df=_safe_read_feature_table(features_dir, "features_pac_trials", logger),
        pac_time_df=_safe_read_feature_table(features_dir, "features_pac_time", logger),
        itpc_df=_safe_read_feature_table(features_dir, "features_itpc", logger),
        complexity_df=_safe_read_feature_table(features_dir, "features_complexity", logger),
        bursts_df=_safe_read_feature_table(features_dir, "features_bursts", logger),
        quality_df=_safe_read_feature_table(features_dir, "features_quality", logger),
        erds_df=_safe_read_feature_table(features_dir, "features_erds", logger),
        spectral_df=_safe_read_feature_table(features_dir, "features_spectral", logger),
        ratios_df=_safe_read_feature_table(features_dir, "features_ratios", logger),
        asymmetry_df=_safe_read_feature_table(features_dir, "features_asymmetry", logger),
        temporal_df=_safe_read_feature_table(features_dir, "features_temporal", logger),
    )

    if include_targets:
        targets_df = _safe_read_feature_table(features_dir, "target_vas_ratings", logger)
        if targets_df is not None:
            bundle.targets = _extract_targets_from_dataframe(
                targets_df, config, logger
            )

    return bundle


def load_feature_dfs_for_subjects(
    subjects: List[str],
    deriv_root: Path,
    input_filename_key: str,
    logger: logging.Logger,
    config: Any,
) -> pd.DataFrame:
    """Load feature dataframes for multiple subjects and concatenate them."""
    input_filename = config.get(input_filename_key)
    if not input_filename:
        logger.warning("Input filename not found in config: %s", input_filename_key)
        return pd.DataFrame()

    all_dfs = []
    for subject in subjects:
        features_dir = deriv_features_path(deriv_root, subject)
        file_path = features_dir / input_filename

        if not file_path.exists():
            logger.debug("Features not found for sub-%s at %s", subject, file_path)
            continue

        try:
            df = read_tsv(file_path)
            if df.empty:
                continue
            df.insert(0, "subject", subject)
            all_dfs.append(df)
        except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
            logger.warning(
                "Failed to read features for sub-%s at %s: %s", subject, file_path, exc
            )
            continue

    if not all_dfs:
        logger.warning(
            "No features found for %s across any subject", input_filename_key
        )
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


###################################################################
# SAVING UTILITIES
###################################################################


def _assign_columns_safely(
    df: pd.DataFrame,
    column_names: Optional[List[str]],
    feature_type: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Assign column names to dataframe with validation and logging."""
    if column_names is None:
        return df
    if len(column_names) == len(df.columns):
        df = df.copy()
        df.columns = column_names
    else:
        logger.warning(
            "%s column mismatch: %d names vs %d columns. Using DataFrame names.",
            feature_type,
            len(column_names),
            len(df.columns),
        )
    return df


def _build_filename(base_name: str, suffix: Optional[str] = None) -> str:
    """Build filename with optional suffix."""
    if suffix:
        return f"{base_name}_{suffix}.tsv"
    return f"{base_name}.tsv"


def _get_folder_for_feature(base_name: str) -> str:
    """Determine subfolder name from base filename."""
    if base_name.startswith("features_"):
        name = base_name.replace("features_", "")
        # Group related features
        if name.startswith("pac"):
            return "pac"
        if name.startswith("power"):
            return "power"
        return name
    
    if base_name == "aperiodic_qc":
        return "aperiodic"
    if base_name == "features":
        return "metadata"  # Manifest
    if base_name == "features_subject":
        return "subject"
    if base_name == "target_vas_ratings":
        return "behavior"
    
    return ""


def _save_feature_dataframe(
    df: pd.DataFrame,
    base_filename: str,
    features_dir: Path,
    logger: logging.Logger,
    column_names: Optional[List[str]] = None,
    feature_type: str = "Feature",
    suffix: Optional[str] = None,
) -> None:
    """Save a feature dataframe with column assignment and logging."""
    df = _assign_columns_safely(df, column_names, feature_type, logger)
    folder_name = _get_folder_for_feature(base_filename)
    filename = _build_filename(base_filename, suffix)
    file_path = features_dir / folder_name / filename
    logger.info("Saving %s: %s", feature_type, file_path)
    write_tsv(df, file_path)


def _save_feature_metadata(
    df: pd.DataFrame,
    base_filename: str,
    features_dir: Path,
    config: Any,
    logger: logging.Logger,
    suffix: Optional[str] = None,
) -> None:
    """Save feature-specific metadata to its metadata subfolder."""
    if df is None or df.empty:
        return

    try:
        from eeg_pipeline.domain.features.naming import generate_manifest

        folder_name = _get_folder_for_feature(base_filename)
        metadata_dir = features_dir / folder_name / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        subject_str = (
            features_dir.parts[-3].replace("sub-", "")
            if len(features_dir.parts) > 3
            else "unknown"
        )

        json_filename = _build_filename(base_filename, suffix).replace(".tsv", ".json")
        metadata_path = metadata_dir / json_filename

        manifest = generate_manifest(
            feature_columns=list(df.columns),
            config=config,
            subject=subject_str,
            task=config.get("project.task") if config is not None else None,
            qc=None,
        )

        with open(metadata_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved feature metadata: %s", metadata_path)
    except (OSError, IOError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Failed to generate feature metadata for %s: %s", base_filename, exc)


def _dedupe_identical_duplicate_columns(
    df: pd.DataFrame,
    df_label: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Remove duplicate columns with identical values, preserving scientific validity."""
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
                f"Duplicate feature column name with non-identical values detected while building "
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


def _is_aperiodic_qc_complete(aper_qc: Dict[str, Any]) -> bool:
    """Check if aperiodic QC dictionary contains all required fields."""
    required_fields = ["freqs", "slopes", "offsets", "r2"]
    return all(aper_qc.get(field) is not None for field in required_fields)


def _save_aperiodic_qc(
    aper_qc: Dict[str, Any],
    features_dir: Path,
    logger: logging.Logger,
    suffix: Optional[str] = None,
) -> None:
    """Save aperiodic QC data to TSV file."""
    if not _is_aperiodic_qc_complete(aper_qc):
        logger.info("Aperiodic QC payload present but incomplete; skipping aperiodic_qc.tsv")
        return

    try:
        folder_name = _get_folder_for_feature("aperiodic_qc")
        qc_dir = features_dir / folder_name
        qc_dir.mkdir(parents=True, exist_ok=True)

        qc_filename = f"aperiodic_qc_{suffix}.tsv" if suffix else "aperiodic_qc.tsv"
        save_path = qc_dir / qc_filename

        subject_name = features_dir.parent.parent.name.replace("sub-", "")
        deriv_root = features_dir.parent.parent.parent
        stats_dir = deriv_stats_path(deriv_root, subject_name)
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / qc_filename

        freqs = aper_qc.get("freqs")
        residual_mean = aper_qc.get("residual_mean")
        r2 = aper_qc.get("r2")
        slopes = aper_qc.get("slopes")
        offsets = aper_qc.get("offsets")
        channel_names = aper_qc.get("channel_names")
        run_labels = aper_qc.get("run_labels")

        rows = []

        if freqs is not None and residual_mean is not None:
            n_channels = residual_mean.shape[0] if residual_mean.ndim > 1 else 1
            for ch_idx in range(n_channels):
                ch_name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"ch_{ch_idx}"
                resid = residual_mean[ch_idx] if residual_mean.ndim > 1 else residual_mean
                for freq_idx, freq in enumerate(freqs):
                    rows.append({
                        "frequency": freq,
                        "channel": ch_name,
                        "residual_mean": float(resid[freq_idx]) if freq_idx < len(resid) else np.nan
                    })

        if r2 is not None:
            n_channels = len(r2) if r2.ndim == 1 else r2.shape[0]
            for ch_idx in range(n_channels):
                ch_name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"ch_{ch_idx}"
                r2_val = r2[ch_idx] if r2.ndim == 1 else (r2[ch_idx, 0] if r2.ndim > 1 else r2)
                rows.append({
                    "frequency": np.nan,
                    "channel": ch_name,
                    "residual_mean": np.nan,
                    "r2": float(r2_val)
                })

        if slopes is not None and offsets is not None:
            n_trials = slopes.shape[0] if slopes.ndim > 1 else 1
            n_channels = slopes.shape[1] if slopes.ndim > 1 else 1
            for trial_idx in range(n_trials):
                run_label = run_labels[trial_idx] if run_labels and trial_idx < len(run_labels) else f"trial_{trial_idx}"
                for ch_idx in range(n_channels):
                    ch_name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"ch_{ch_idx}"
                    slope_val = slopes[trial_idx, ch_idx] if slopes.ndim > 1 else slopes
                    offset_val = offsets[trial_idx, ch_idx] if offsets.ndim > 1 else offsets
                    rows.append({
                        "frequency": np.nan,
                        "channel": ch_name,
                        "residual_mean": np.nan,
                        "r2": np.nan,
                        "trial": run_label,
                        "slope": float(slope_val),
                        "offset": float(offset_val)
                    })

        df = pd.DataFrame(rows)

        for target_path in [save_path, stats_path]:
            write_tsv(df, target_path)

        logger.info("Saved aperiodic QC sidecar to %s and %s", save_path, stats_path)
    except (OSError, IOError, TypeError, KeyError) as exc:
        logger.warning("Failed to save aperiodic QC TSV: %s", exc)


def _save_feature_manifest(
    direct_df: pd.DataFrame,
    features_dir: Path,
    config: Any,
    feature_qc: Optional[Dict[str, Any]],
    logger: logging.Logger,
    suffix: Optional[str] = None,
) -> None:
    """Save feature metadata manifest as JSON sidecar."""
    from eeg_pipeline.domain.features.naming import generate_manifest

    try:
        subject_str = (
            features_dir.parts[-3].replace("sub-", "")
            if len(features_dir.parts) > 3
            else "unknown"
        )
        folder_name = _get_folder_for_feature("features")
        json_filename = _build_filename("features", suffix).replace(".tsv", ".json")
        sidecar_path = features_dir / folder_name / json_filename
        manifest = generate_manifest(
            feature_columns=list(direct_df.columns),
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


def save_all_features(
    pow_df: pd.DataFrame,
    pow_cols: List[str],
    baseline_df: pd.DataFrame,
    baseline_cols: List[str],
    conn_df: Optional[pd.DataFrame],
    conn_cols: List[str],
    aper_df: Optional[pd.DataFrame],
    aper_cols: List[str],
    erp_df: Optional[pd.DataFrame] = None,
    erp_cols: Optional[List[str]] = None,
    itpc_df: Optional[pd.DataFrame] = None,
    itpc_cols: Optional[List[str]] = None,
    pac_df: Optional[pd.DataFrame] = None,
    pac_trials_df: Optional[pd.DataFrame] = None,
    pac_time_df: Optional[pd.DataFrame] = None,
    aper_qc: Optional[Dict[str, Any]] = None,
    active_df: Optional[pd.DataFrame] = None,
    active_cols: Optional[List[str]] = None,
    y: Optional[pd.Series] = None,
    features_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    config: Any = None,
    comp_df: Optional[pd.DataFrame] = None,
    comp_cols: Optional[List[str]] = None,
    bursts_df: Optional[pd.DataFrame] = None,
    bursts_cols: Optional[List[str]] = None,
    spectral_df: Optional[pd.DataFrame] = None,
    spectral_cols: Optional[List[str]] = None,
    erds_df: Optional[pd.DataFrame] = None,
    erds_cols: Optional[List[str]] = None,
    ratios_df: Optional[pd.DataFrame] = None,
    ratios_cols: Optional[List[str]] = None,
    asymmetry_df: Optional[pd.DataFrame] = None,
    asymmetry_cols: Optional[List[str]] = None,
    quality_df: Optional[pd.DataFrame] = None,
    quality_cols: Optional[List[str]] = None,
    feature_qc: Optional[Dict[str, Any]] = None,
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    """Save all feature dataframes to disk with validation and deduplication."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if features_dir is None:
        raise ValueError("features_dir must be provided")

    direct_blocks = []
    if pow_df is not None and not pow_df.empty:
        pow_df = _assign_columns_safely(pow_df, pow_cols, "Power", logger)
        logger.debug("Adding Power block to direct features: %d columns", len(pow_df.columns))
        direct_blocks.append(pow_df)

    if baseline_df is not None and not baseline_df.empty:
        # Only include baseline columns in the specific baseline file.
        # This keeps the outputs window-specific as requested by the user.
        if isinstance(suffix, str) and suffix.lower() == "baseline":
            baseline_df = _assign_columns_safely(baseline_df, baseline_cols, "Baseline", logger)
            logger.debug("Adding Baseline block to direct features: %d columns", len(baseline_df.columns))
            direct_blocks.append(baseline_df)


    feature_save_configs = [
        (aper_df, aper_cols, "features_aperiodic", "aperiodic features"),
        (erp_df, erp_cols, "features_erp", "ERP/LEP features"),
        (itpc_df, itpc_cols, "features_itpc", "ITPC features (channel x band x segment)"),
        (pac_df, None, "features_pac", "PAC comodulograms"),
        (pac_trials_df, None, "features_pac_trials", "PAC per-trial values"),
        (pac_time_df, None, "features_pac_time", "PAC time-resolved values"),
        (comp_df, comp_cols, "features_complexity", "complexity features"),
        (bursts_df, bursts_cols, "features_bursts", "burst features"),
        (spectral_df, spectral_cols, "features_spectral", "spectral features (IAF)"),
        (erds_df, erds_cols, "features_erds", "ERDS features"),
        (ratios_df, ratios_cols, "features_ratios", "power ratio features"),
        (asymmetry_df, asymmetry_cols, "features_asymmetry", "asymmetry features"),
        (quality_df, quality_cols, "features_quality", "quality metrics"),
    ]

    for df, cols, base_name, description in feature_save_configs:
        if df is not None and not df.empty:
            _save_feature_dataframe(
                df, base_name, features_dir, logger, cols, description, suffix
            )
            _save_feature_metadata(
                df, base_name, features_dir, config, logger, suffix
            )

    if aper_qc:
        _save_aperiodic_qc(aper_qc, features_dir, logger, suffix)

    if direct_blocks:
        direct_df = pd.concat(direct_blocks, axis=1)
        direct_df = _dedupe_identical_duplicate_columns(
            direct_df, "features_power.tsv", logger
        )
    else:
        direct_df = pd.DataFrame()

    if not direct_df.empty:
        folder_name = _get_folder_for_feature("features_power")
        direct_filename = _build_filename("features_power", suffix)
        direct_path = features_dir / folder_name / direct_filename
        logger.info("Saving power features: %s", direct_path)
        write_tsv(direct_df, direct_path)
        _save_feature_metadata(
            direct_df, "features_power", features_dir, config, logger, suffix
        )

    if active_df is not None and not active_df.empty:
        folder_name = _get_folder_for_feature("features_power_active")
        active_filename = _build_filename("features_power_active", suffix)
        active_path = features_dir / folder_name / active_filename
        logger.info("Saving active-averaged EEG features: %s", active_path)
        write_tsv(active_df, active_path)
        _save_feature_metadata(
            active_df, "features_power_active", features_dir, config, logger, suffix
        )

    if conn_df is not None and not conn_df.empty:
        conn_df = _assign_columns_safely(conn_df, conn_cols, "Connectivity", logger)
        folder_name = _get_folder_for_feature("features_connectivity")
        conn_filename = _build_filename("features_connectivity", suffix)
        conn_path = features_dir / folder_name / conn_filename
        logger.info("Saving connectivity features: %s", conn_path)
        start_time = time.perf_counter()
        write_tsv(conn_df, conn_path)
        logger.info(
            "Saved connectivity TSV in %.2fs (rows=%d, cols=%d)",
            time.perf_counter() - start_time,
            len(conn_df),
            len(conn_df.columns),
        )
        _save_feature_metadata(
            conn_df, "features_connectivity", features_dir, config, logger, suffix
        )

    _save_feature_manifest(direct_df, features_dir, config, feature_qc, logger, suffix)

    if y is not None:
        target_path = features_dir / "behavior" / "target_vas_ratings.tsv"
        rating_columns = (
            config.get("event_columns.rating", ["vas_rating"])
            if config is not None and hasattr(config, "get")
            else ["vas_rating"]
        )
        target_column_name = rating_columns[0] if rating_columns else "vas_rating"
        logger.info(
            "Saving behavioral target vector: %s (column: %s)",
            target_path,
            target_column_name,
        )
        write_tsv(y.to_frame(name=target_column_name), target_path)

    return direct_df






###################################################################
# TRIAL MANAGEMENT
###################################################################

def save_trial_alignment_manifest(
    aligned_events: pd.DataFrame,
    epochs: mne.Epochs,
    manifest_path: Path,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Save trial alignment manifest with epoch and event metadata."""
    manifest: Dict[str, Any] = {
        "n_epochs": int(len(epochs)) if epochs is not None else None,
        "n_events": int(len(aligned_events)) if aligned_events is not None else None,
        "epoch_times": {
            "tmin": float(epochs.tmin) if epochs is not None else None,
            "tmax": float(epochs.tmax) if epochs is not None else None,
        },
        "config_task": (
            config.get("project.task") if config is not None and hasattr(config, "get") else None
        ),
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
    """Extract feature columns matching optional prefix."""
    if feature_df is None or feature_df.empty:
        return [], pd.DataFrame()
    if col_prefix:
        matching_cols = [
            col for col in feature_df.columns if str(col).startswith(col_prefix)
        ]
        if not matching_cols:
            return [], pd.DataFrame()
        return matching_cols, feature_df[matching_cols]
    return list(feature_df.columns), feature_df


__all__ = [
    "FeatureBundle",
    "load_feature_bundle",
    "load_feature_dfs_for_subjects",
    "load_subject_features",
    "_load_features_and_targets",
    "iterate_feature_columns",
    "save_all_features",
    "save_dropped_trials_log",
    "save_trial_alignment_manifest",
]
