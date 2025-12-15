from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.io.columns import pick_target_column
from eeg_pipeline.io.paths import deriv_features_path, find_connectivity_features_path
from eeg_pipeline.io.tsv import read_tsv, read_table


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

    def _safe_read(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return read_table(path)
        except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    ms_df = _safe_read(features_dir / "features_microstates.tsv")
    conn_df = _safe_read(find_connectivity_features_path(deriv_root, subject))
    aper_df = _safe_read(features_dir / "features_aperiodic.tsv")
    pac_df = _safe_read(features_dir / "features_pac.tsv")
    pac_trials_df = _safe_read(features_dir / "features_pac_trials.tsv")
    pac_time_df = _safe_read(features_dir / "features_pac_time.tsv")
    itpc_df = _safe_read(features_dir / "features_itpc.tsv")

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

    def _safe_read(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return read_table(path)
        except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    bundle = FeatureBundle(
        power_df=_safe_read(features_dir / "features_eeg_direct.tsv"),
        microstate_df=_safe_read(features_dir / "features_microstates.tsv"),
        connectivity_df=_safe_read(find_connectivity_features_path(deriv_root, subject)),
        aperiodic_df=_safe_read(features_dir / "features_aperiodic.tsv"),
        pac_df=_safe_read(features_dir / "features_pac.tsv"),
        pac_trials_df=_safe_read(features_dir / "features_pac_trials.tsv"),
        pac_time_df=_safe_read(features_dir / "features_pac_time.tsv"),
        itpc_df=_safe_read(features_dir / "features_itpc.tsv"),
        complexity_df=_safe_read(features_dir / "features_complexity.tsv"),
        dynamics_df=_safe_read(features_dir / "features_dynamics.tsv"),
        all_features_df=_safe_read(features_dir / "features_all.tsv"),
    )

    if include_targets:
        targets_df = _safe_read(features_dir / "target_vas_ratings.tsv")
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
