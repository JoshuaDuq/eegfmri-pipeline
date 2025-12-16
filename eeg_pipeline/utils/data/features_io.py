from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.infra.paths import deriv_features_path, find_connectivity_features_path
from eeg_pipeline.infra.tsv import read_tsv, read_table
from eeg_pipeline.utils.data.epochs_loading import load_epochs_for_analysis


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
        feature_path = deriv_features_path(deriv_root, subject) / "features_eeg_direct.tsv"
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
    temporal_path = feats_dir / "features_eeg_direct.tsv"
    plateau_path = feats_dir / "features_eeg_plateau.tsv"
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


def _load_feature_bundle_for_subject(
    subject: str,
    deriv_root: Path,
    logger: Optional[logging.Logger] = None,
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """Legacy tuple-return bundle loader (kept for backward compatibility)."""
    if logger is None:
        logger = logging.getLogger(__name__)

    features_dir = deriv_features_path(deriv_root, subject)
    power_path = features_dir / "features_eeg_direct.tsv"
    if not power_path.exists():
        logger.warning("Power features not found at %s", power_path)
        return None, None, None, None, None, None, None, None

    try:
        pow_df = read_table(power_path)
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        logger.error("Failed to read power features from %s: %s", power_path, exc)
        return None, None, None, None, None, None, None, None

    ms_df = _safe_read_table(features_dir / "features_microstates.tsv", logger)
    conn_df = _safe_read_table(find_connectivity_features_path(deriv_root, subject), logger)
    aper_df = _safe_read_table(features_dir / "features_aperiodic.tsv", logger)
    pac_df = _safe_read_table(features_dir / "features_pac.tsv", logger)
    pac_trials_df = _safe_read_table(features_dir / "features_pac_trials.tsv", logger)
    pac_time_df = _safe_read_table(features_dir / "features_pac_time.tsv", logger)
    itpc_df = _safe_read_table(features_dir / "features_itpc.tsv", logger)

    return pow_df, ms_df, conn_df, aper_df, pac_df, pac_trials_df, pac_time_df, itpc_df


load_feature_bundle_for_subject = _load_feature_bundle_for_subject


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
        power_df=_safe_read_table(features_dir / "features_eeg_direct.tsv", logger),
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
