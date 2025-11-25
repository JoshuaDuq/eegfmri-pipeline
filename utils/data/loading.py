from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass
import logging
import warnings
import numpy as np
import pandas as pd
import mne

from ..io.general import (
    _find_clean_epochs_path,
    _load_events_df,
    _pick_target_column,
    deriv_features_path,
    read_tsv,
)
from ..config.loader import load_settings, ConfigDict, get_config_value, ensure_config

EEGConfig = ConfigDict


###################################################################
# Event / Epoch Alignment
###################################################################

def _align_by_selection(events_df: pd.DataFrame, epochs: mne.Epochs, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if not hasattr(epochs, "selection") or epochs.selection is None:
        return None
    
    sel_arr = np.asarray(epochs.selection, dtype=int)
    min_sel = int(np.min(sel_arr))
    max_sel = int(np.max(sel_arr))
    
    if min_sel < 0:
        raise ValueError(f"Invalid epochs.selection: contains negative indices (min={min_sel})")
    
    if max_sel >= len(events_df):
        raise ValueError(f"epochs.selection out of bounds: max={max_sel}, events_len={len(events_df)}")
    
    if len(sel_arr) != len(epochs):
        raise ValueError(
            f"epochs.selection length mismatch: selection={len(sel_arr)}, epochs={len(epochs)}"
        )
    
    aligned = events_df.iloc[sel_arr].reset_index(drop=True)
    logger.info(f"Successfully aligned using epochs.selection ({len(sel_arr)} epochs)")
    return aligned


def _align_by_sample(events_df: pd.DataFrame, epochs: mne.Epochs, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if "sample" not in events_df.columns:
        return None
    
    if not hasattr(epochs, "events") or epochs.events is None:
        return None
    
    epoch_samples = epochs.events[:, 0]
    events_indexed = events_df.set_index("sample")
    aligned = events_indexed.reindex(epoch_samples)
    
    if len(aligned) != len(epochs):
        return None
    
    if aligned.isna().all(axis=1).any():
        nan_count = aligned.isna().all(axis=1).sum()
        raise ValueError(f"Sample-based alignment failed: {nan_count} NaN rows")
    
    aligned_reset = aligned.reset_index()
    logger.info(f"Successfully aligned using sample column ({len(aligned)} epochs)")
    return aligned_reset


def align_events_to_epochs(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    logger = logger or logging.getLogger(__name__)

    if events_df is None or len(events_df) == 0:
        logger.debug("No events DataFrame provided")
        return None

    if len(epochs) == 0:
        logger.debug("No epochs provided")
        return pd.DataFrame()

    logger.info(f"Attempting alignment: {len(events_df)} events to {len(epochs)} epochs")

    aligned = _align_by_selection(events_df, epochs, logger)
    if aligned is not None:
        return aligned
    
    aligned = _align_by_sample(events_df, epochs, logger)
    if aligned is not None:
        return aligned

    error_msg = (
        f"Cannot guarantee events-to-epochs alignment for reliable analysis. "
        f"Events DataFrame ({len(events_df)} rows) cannot be reliably aligned to "
        f"epochs ({len(epochs)} epochs). Explicit alignment is required:\n"
        f"1. Ensure epochs.selection is properly set during epoching, OR\n"
        f"2. Include 'sample' column in events with matching sample indices, OR\n"
        f"3. Use trial_alignment.manifest.tsv created by 03_feature_extraction.py and align via load_epochs_for_analysis."
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


def trim_behavioral_to_events_strict(
    behavioral_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)

    matched = _match_behavioral_to_events(behavioral_df, events_df, logger)
    if matched is not None:
        return matched

    behavioral_len = len(behavioral_df)
    events_len = len(events_df)

    if behavioral_len == events_len:
        logger.warning(
            "Lengths match but could not verify alignment by identifiers. "
            "Using row-order alignment (assumes perfect sequential match)."
        )
        return behavioral_df.reset_index(drop=True)

    if behavioral_len > events_len:
        error_msg = (
            f"Behavioral data longer than events ({behavioral_len} > {events_len}). "
            f"Could not match by identifiers. In strict mode, cannot safely trim "
            f"behavioral data as extra rows may not be at the end, which would cause "
            f"misalignment between behavioral ratings and EEG trials. "
            f"Please verify data alignment or use a non-strict trimming function."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    error_msg = (
        f"Behavioral data ({behavioral_len} rows) is shorter than events "
        f"({events_len} rows). This indicates missing behavioral trials "
        f"that cannot be safely recovered. Check data collection/preprocessing."
    )
    logger.critical(f"CRITICAL: Behavioral data shorter than events. Behavioral: {behavioral_len}, Events: {events_len}. This indicates missing behavioral trials.")
    raise ValueError(error_msg)


def _match_by_trial_identifiers(
    behavioral_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    trial_cols_behavioral = ["trial", "trial_number", "trial_index", "Trial", "TrialNumber"]
    trial_cols_events = ["trial", "trial_number", "trial_index"]
    
    behavioral_trial_col = _pick_first_column(behavioral_df, trial_cols_behavioral)
    events_trial_col = _pick_first_column(events_df, trial_cols_events)
    
    if not behavioral_trial_col or not events_trial_col:
        return None
    
    try:
        beh_trials = pd.to_numeric(behavioral_df[behavioral_trial_col], errors="coerce")
        ev_trials = pd.to_numeric(events_df[events_trial_col], errors="coerce")
        
        if beh_trials.notna().sum() == 0 or ev_trials.notna().sum() == 0:
            return None
        
        ev_trial_set = set(ev_trials[ev_trials.notna()].astype(int))
        matched_indices = []
        for beh_idx, beh_trial in enumerate(beh_trials):
            if pd.isna(beh_trial):
                continue
            beh_trial_int = int(beh_trial)
            if beh_trial_int in ev_trial_set:
                ev_matches = events_df.index[ev_trials == beh_trial_int].tolist()
                if ev_matches:
                    matched_indices.append(beh_idx)
        
        if len(matched_indices) == len(events_df):
            matched = behavioral_df.iloc[matched_indices].reset_index(drop=True)
            logger.info(
                f"Matched {len(matched)} behavioral trials to events using trial identifiers "
                f"({behavioral_trial_col} <-> {events_trial_col})"
            )
            return matched
    except Exception as e:
        logger.debug(f"Trial-based matching failed: {e}")
    
    return None


def _match_by_temperature(
    behavioral_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    temp_cols_behavioral = ["stimulus_temp", "temperature", "thermode_temperature", "Temp", "Temperature"]
    temp_cols_events = ["stimulus_temp", "temperature", "thermode_temperature"]
    onset_cols_events = ["onset", "Onset"]
    
    behavioral_temp_col = _pick_first_column(behavioral_df, temp_cols_behavioral)
    events_temp_col = _pick_first_column(events_df, temp_cols_events)
    events_onset_col = _pick_first_column(events_df, onset_cols_events)
    
    if not behavioral_temp_col or not events_temp_col or not events_onset_col:
        return None
    
    try:
        beh_temp = pd.to_numeric(behavioral_df[behavioral_temp_col], errors="coerce")
        ev_temp = pd.to_numeric(events_df[events_temp_col], errors="coerce")
        ev_onset = pd.to_numeric(events_df[events_onset_col], errors="coerce")
        
        valid_mask = beh_temp.notna() & ev_temp.notna() & ev_onset.notna()
        if valid_mask.sum() == 0:
            return None
        
        matched_indices = []
        ev_temp_onset_pairs = list(zip(ev_temp[valid_mask], ev_onset[valid_mask]))
        
        for beh_idx, beh_t in enumerate(beh_temp):
            if pd.isna(beh_t):
                continue
            matches = [
                i for i, (ev_t, _) in enumerate(ev_temp_onset_pairs)
                if abs(float(ev_t) - float(beh_t)) < 0.1
            ]
            if matches:
                matched_indices.append(beh_idx)
        
        if len(matched_indices) == len(events_df):
            matched = behavioral_df.iloc[matched_indices].reset_index(drop=True)
            logger.info(
                f"Matched {len(matched)} behavioral trials to events using temperature "
                f"({behavioral_temp_col} <-> {events_temp_col})"
            )
            return matched
    except Exception as e:
        logger.debug(f"Temperature-based matching failed: {e}")
    
    return None


def _match_behavioral_to_events(
    behavioral_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    matched = _match_by_trial_identifiers(behavioral_df, events_df, logger)
    if matched is not None:
        return matched
    
    return _match_by_temperature(behavioral_df, events_df, logger)


def align_events_to_epochs_strict(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    return align_events_to_epochs(events_df, epochs, logger=logger)


def align_events_with_policy(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    config,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if config is None:
        raise ValueError("config is required for align_events_with_policy")

    allow_trim = bool(config.get("alignment.allow_misaligned_trim"))
    min_samples = int(config.get("alignment.min_alignment_samples"))

    if events_df is None or len(events_df) == 0:
        logger.debug("No events DataFrame provided")
        return None

    if len(epochs) == 0:
        logger.debug("No epochs provided")
        return pd.DataFrame()

    n_events = len(events_df)
    n_epochs = len(epochs)

    aligned = align_events_to_epochs(events_df, epochs, logger=logger)

    if aligned is None:
        error_msg = f"Alignment failed: could not align {n_events} events to {n_epochs} epochs"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(aligned) != n_epochs:
        diff = abs(len(aligned) - n_epochs)
        if allow_trim and diff <= min_samples:
            n_keep = min(len(aligned), n_epochs)
            logger.warning(
                f"Alignment length mismatch (events={len(aligned)}, epochs={n_epochs}, diff={diff}). "
                f"Trimming to {n_keep} samples (allow_misaligned_trim=True, max_tolerable_mismatch={min_samples})"
            )
            if len(aligned) > n_epochs:
                aligned = aligned.iloc[:n_epochs].reset_index(drop=True)
            else:
                error_msg = f"Cannot trim: aligned has {len(aligned)} rows, epochs has {n_epochs}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            reason = "allow_misaligned_trim=False" if not allow_trim else f"mismatch ({diff}) exceeds max_tolerable_mismatch ({min_samples})"
            error_msg = (
                f"Alignment length mismatch (events={len(aligned)}, epochs={n_epochs}, diff={diff}). "
                f"Cannot proceed: {reason}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    return aligned


def _handle_validation_error(msg: str, strict: bool, logger: logging.Logger) -> bool:
    if strict:
        raise ValueError(msg)
    logger.error(msg)
    return False


def validate_alignment(
    aligned_events: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
    config: Optional[Any] = None,
) -> bool:
    logger = logger or logging.getLogger(__name__)

    if aligned_events is None:
        if strict:
            raise ValueError("aligned_events is None; cannot validate alignment")
        return False

    if len(aligned_events) != len(epochs):
        return _handle_validation_error(
            f"Length mismatch: events={len(aligned_events)}, epochs={len(epochs)}",
            strict, logger
        )

    nan_fraction = aligned_events.isna().all(axis=1).mean()
    if config is None:
        raise ValueError("config is required for max_nan_fraction")
    max_nan_fraction = float(config.get("analysis.data_quality.max_nan_fraction"))
    if nan_fraction > max_nan_fraction:
        return _handle_validation_error(
            f"High NaN fraction in aligned events: {nan_fraction:.1%}",
            strict, logger
        )

    logger.info(
        f"Alignment validation passed: {len(aligned_events)} rows, "
        f"{nan_fraction:.1%} NaN fraction"
    )
    return True


def align_or_raise(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    config,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    aligned = align_events_with_policy(events_df, epochs, config=config, logger=logger)
    if aligned is None or len(aligned) == 0:
        raise ValueError("Alignment produced empty events DataFrame")
    validate_alignment(aligned, epochs, logger=logger, strict=True, config=config)
    return aligned


def get_aligned_events(
    epochs: mne.Epochs,
    subject: str,
    task: str,
    *,
    strict: bool = True,
    logger: Optional[logging.Logger] = None,
    bids_root: Optional[Path] = None,
    config=None,
    constants=None,
) -> Optional[pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if config is None:
        raise ValueError("config is required for get_aligned_events")

    events_df = _load_events_df(subject, task, bids_root=bids_root, config=config, constants=constants)
    if events_df is None:
        if strict:
            raise ValueError(f"Events TSV not found for sub-{subject}, task-{task}. Required when strict=True")
        logger.warning(f"Events TSV not found for sub-{subject}, task-{task}")
        return None

    try:
        aligned_events = align_events_to_epochs(events_df, epochs, logger=logger)
    except ValueError as err:
        if strict:
            raise ValueError(
                f"Alignment failed for sub-{subject}, task-{task} in strict mode: {err}"
            ) from err
        logger.warning(f"Alignment failed for sub-{subject}, task-{task}: {err}")
        return None

    if aligned_events is None:
        if strict:
            raise ValueError(
                f"Could not align events to epochs for sub-{subject}, task-{task}. "
                f"This is required when strict=True."
            )
        return None

    if len(aligned_events) != len(epochs):
        msg = (
            f"Alignment length mismatch for sub-{subject}, task-{task}: "
            f"aligned_events ({len(aligned_events)} rows) != epochs ({len(epochs)} epochs)"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    validate_alignment(aligned_events, epochs, logger, strict=strict, config=config)

    return aligned_events


###################################################################
# Typed Return Values
###################################################################

@dataclass
class DecodingDataResult:
    features: pd.DataFrame
    targets: pd.Series
    groups: np.ndarray
    metadata: pd.DataFrame
    
    @property
    def empty(self) -> bool:
        return len(self.features) == 0 or len(self.targets) == 0


###################################################################
# Helper Functions
###################################################################

def _get_trial_alignment_manifest_path(deriv_root: Path, subject: str) -> Path:
    sub = f"sub-{subject}" if not subject.startswith("sub-") else subject
    return deriv_root / sub / "eeg" / "features" / "trial_alignment.tsv"


def _load_trial_alignment_manifest(manifest_path: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Trial alignment manifest missing: {manifest_path}\n"
            f"This file is required and must be created by 03_feature_extraction.py. "
            f"Run 03_feature_extraction.py first to generate features with proper alignment."
        )
    
    manifest = pd.read_csv(manifest_path, sep="\t")
    if "trial_index" not in manifest.columns:
        raise ValueError(
            f"Invalid trial alignment manifest: missing 'trial_index' column in {manifest_path}"
        )
    
    if len(manifest) == 0:
        raise ValueError(
            f"Trial alignment manifest is empty: {manifest_path}"
        )
    
    logger.debug(f"Loaded trial alignment manifest: {len(manifest)} trials from {manifest_path}")
    return manifest


def _collect_subjects_from_bids(bids_root: Path) -> List[str]:
    if not bids_root.exists():
        return []
    subjects = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if sub_dir.is_dir():
            subjects.append(sub_dir.name[4:])
    return subjects


def _collect_subjects_from_derivatives_epochs(deriv_root: Path, task: str, config: Optional[EEGConfig] = None, constants: Optional[Dict[str, Any]] = None) -> List[str]:
    if not deriv_root.exists():
        return []
    from ..io.general import _find_clean_epochs_path
    subjects = []
    for sub_dir in sorted(deriv_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name[4:]
        epo_path = _find_clean_epochs_path(sub_id, task, deriv_root=deriv_root, config=config, constants=constants)
        if epo_path is not None and epo_path.exists():
            subjects.append(sub_id)
    return subjects


def _collect_subjects_from_features(deriv_root: Path) -> List[str]:
    if not deriv_root.exists():
        return []
    subjects = []
    for sub_dir in sorted(deriv_root.glob("sub-*/eeg/features")):
        eeg_feat = sub_dir / "features_eeg_direct.tsv"
        y_tsv = sub_dir / "target_vas_ratings.tsv"
        if eeg_feat.exists() and y_tsv.exists():
            sub_id = sub_dir.parts[-3].replace("sub-", "")
            subjects.append(sub_id)
    return subjects


def _collect_subject_ids_with_features(deriv_root: Path) -> List[str]:
    return _collect_subjects_from_features(deriv_root)


def _pick_first_column(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    if df is None:
        return None
    from ..io.general import _find_column_in_dataframe
    return _find_column_in_dataframe(df, candidates)


def _canonical_covariate_name(name: Optional[str], config=None) -> Optional[str]:
    if name is None:
        return None
    
    n = str(name).lower()
    
    if config is None:
        try:
            config = load_settings()
        except Exception:
            pass
    
    temp_aliases = {"stimulus_temp", "stimulus_temperature", "temp", "temperature"}
    trial_aliases = {"trial", "trial_number", "trial_index", "run", "block"}
    
    if config is not None:
        temp_cols = config.get("event_columns.temperature", [])
        temp_aliases.update(str(c).lower() for c in temp_cols)
    
    if n in temp_aliases:
        return "temperature"
    if n in trial_aliases:
        return "trial"
    
    return n


def extract_default_covariates(events_df: pd.DataFrame, config) -> List[str]:
    covariates = []
    
    temperature_columns = config.get("event_columns.temperature")
    temperature_column = _pick_first_column(events_df, temperature_columns)
    if temperature_column:
        covariates.append(temperature_column)
    
    trial_column_candidates = ["trial", "trial_number", "trial_index", "run", "block"]
    trial_col = _pick_first_column(events_df, trial_column_candidates)
    if trial_col:
        covariates.append(trial_col)
    
    return covariates


def extract_temperature_data(
    aligned_events: Optional[pd.DataFrame],
    config,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    if aligned_events is None:
        return None, None
    
    psych_temp_columns = config.get("event_columns.temperature")
    temp_col = _pick_first_column(aligned_events, psych_temp_columns)
    if not temp_col:
        return None, None
    
    temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")
    return temp_series, temp_col


def _add_covariate_column(
    covariate_columns: List[str],
    column_name_map: Dict[str, str],
    col_name: str,
    canonical_name: str
) -> None:
    covariate_columns.append(col_name)
    column_name_map[col_name] = canonical_name


def _resolve_covariate_columns(
    df_events: pd.DataFrame,
    partial_covars: Optional[List[str]],
    config: Optional[Any],
) -> Tuple[List[str], Dict[str, str]]:
    covariate_columns: List[str] = []
    column_name_map: Dict[str, str] = {}
    if config is None:
        raise ValueError("config is required")
    temperature_candidates = config.get("event_columns.temperature")

    if partial_covars:
        for covariate in partial_covars:
            if covariate in df_events.columns:
                canonical_name = _canonical_covariate_name(covariate, config=config) or covariate
                _add_covariate_column(covariate_columns, column_name_map, covariate, canonical_name)
                continue
            
            canonical_name = _canonical_covariate_name(covariate, config=config)
            if canonical_name == "temperature":
                temp_column = _pick_first_column(df_events, temperature_candidates)
                if temp_column:
                    _add_covariate_column(covariate_columns, column_name_map, temp_column, canonical_name)
    else:
        temp_column = _pick_first_column(df_events, temperature_candidates)
        if temp_column:
            _add_covariate_column(covariate_columns, column_name_map, temp_column, "temperature")
        
        trial_candidates = ["trial", "trial_number", "trial_index", "run", "block"]
        for candidate in trial_candidates:
            if candidate in df_events.columns:
                canonical_name = _canonical_covariate_name(candidate, config=config) or candidate
                _add_covariate_column(covariate_columns, column_name_map, candidate, canonical_name)
                break

    return covariate_columns, column_name_map


def _build_covariate_matrices(
    df_events: Optional[pd.DataFrame],
    partial_covars: Optional[List[str]],
    temp_col: Optional[str],
    config: Optional[Any] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if df_events is None:
        return None, None

    if config is None:
        try:
            config = load_settings()
        except Exception:
            config = None

    covariate_columns, column_name_map = _resolve_covariate_columns(df_events, partial_covars, config)
    
    if not covariate_columns:
        return None, None

    covariates_df = pd.DataFrame()
    for covariate in covariate_columns:
        if covariate in df_events.columns:
            canonical_name = column_name_map.get(covariate, covariate)
            covariates_df[canonical_name] = pd.to_numeric(df_events[covariate], errors="coerce")

    if covariates_df.empty:
        return None, None

    temp_canonical = _canonical_covariate_name(temp_col, config=config) if temp_col else None
    covariates_without_temp = covariates_df.drop(columns=[temp_canonical], errors="ignore") if temp_canonical else covariates_df.copy()
    
    if covariates_without_temp.shape[1] == 0:
        covariates_without_temp = None

    return covariates_df, covariates_without_temp


def _load_features_and_targets(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    epochs: Optional[mne.Epochs] = None,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], pd.Series, mne.Info]:
    feats_dir = deriv_features_path(deriv_root, subject)
    temporal_path = feats_dir / "features_eeg_direct.tsv"
    plateau_path = feats_dir / "features_eeg_plateau.tsv"
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"

    if not plateau_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing features or targets for sub-{subject}. Expected at {feats_dir}")

    temporal_df = pd.read_csv(temporal_path, sep="\t") if temporal_path.exists() else None
    plateau_df = pd.read_csv(plateau_path, sep="\t")
    conn_df = pd.read_csv(conn_path, sep="\t") if conn_path.exists() else None
    y_df = pd.read_csv(y_path, sep="\t")

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
            bids_root=config.bids_root,
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

    return temporal_df, plateau_df, conn_df, y, epochs.info

def _resolve_subjects_with_policy(
    subjects_from_files: List[str],
    subjects_from_config: List[str],
    policy: Literal["intersection", "union", "config_only"],
    logger: logging.Logger,
) -> List[str]:
    if policy == "config_only":
        resolved = subjects_from_config
        logger.info(f"Using config subjects only: {len(resolved)} subjects")
        return resolved
    
    if policy == "intersection":
        resolved = sorted(list(set(subjects_from_files) & set(subjects_from_config)))
        logger.info(
            f"Using intersection: {len(resolved)} subjects "
            f"(file={len(subjects_from_files)}, config={len(subjects_from_config)})"
        )
        return resolved
    
    if policy == "union":
        resolved = sorted(list(set(subjects_from_files) | set(subjects_from_config)))
        logger.info(
            f"Using union: {len(resolved)} subjects "
            f"(file={len(subjects_from_files)}, config={len(subjects_from_config)})"
        )
        return resolved
    
    raise ValueError(f"Unknown policy: {policy}. Must be 'intersection', 'union', or 'config_only'")


def _validate_event_columns(events_df: pd.DataFrame, config: EEGConfig, logger: logging.Logger) -> None:
    if events_df is None or events_df.empty:
        return
    
    event_cols_config = config.get("event_columns", {})
    if not event_cols_config:
        logger.warning("No event_columns found in config; skipping validation")
        return
    
    missing_columns = []
    for logical_name, candidates in event_cols_config.items():
        if not isinstance(candidates, (list, tuple)):
            continue
        found = any(col in events_df.columns for col in candidates)
        if not found:
            missing_columns.append(f"event_columns.{logical_name} (tried: {candidates})")
    
    if missing_columns:
        available = list(events_df.columns)
        error_msg = (
            f"Required event columns not found in events DataFrame. "
            f"Missing: {', '.join(missing_columns)}. "
            f"Available columns: {available}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


###################################################################
# Main Data Loading Functions
###################################################################

def _validate_load_epochs_params(align: str, config: Any) -> None:
    if align not in ("strict", "warn", "none"):
        raise ValueError(f"align must be one of 'strict', 'warn', 'none', got '{align}'")
    if config is None:
        raise ValueError("config is required for load_epochs_for_analysis")


def _load_epochs_and_events(
    subject: str,
    task: str,
    deriv_root: Optional[Path],
    bids_root: Optional[Path],
    preload: bool,
    config: Any,
    constants: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    epochs_path = _find_clean_epochs_path(subject, task, deriv_root=deriv_root, config=config, constants=constants)
    if epochs_path is None or not epochs_path.exists():
        logger.error(f"Could not find cleaned epochs file for sub-{subject}, task-{task}")
        return None, None
    
    logger.info(f"Loading epochs: {epochs_path}")
    epochs = mne.read_epochs(epochs_path, preload=preload, verbose=False)
    
    events_df = _load_events_df(subject, task, bids_root=bids_root, config=config, constants=constants)
    return epochs, events_df


def _handle_alignment_mismatch(
    aligned_events: pd.DataFrame,
    epochs: mne.Epochs,
    subject: str,
    task: str,
    allow_trim: bool,
    min_alignment_samples: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    n_events = len(aligned_events)
    n_epochs = len(epochs)
    
    if n_events == n_epochs:
        return aligned_events
    
    diff = abs(n_events - n_epochs)
    
    if allow_trim and diff <= min_alignment_samples:
        logger.warning(
            f"Alignment length mismatch for sub-{subject}, task-{task}: "
            f"events={n_events}, epochs={n_epochs}, diff={diff}. "
            f"Trimming enabled (max_tolerable_mismatch={min_alignment_samples})."
        )
        if n_events > n_epochs:
            return aligned_events.iloc[:n_epochs].reset_index(drop=True)
        else:
            from ..io.general import log_and_raise_error
            error_msg = (
                f"Alignment length mismatch for sub-{subject}, task-{task}: "
                f"events={n_events}, epochs={n_epochs}, diff={diff}. "
                f"Cannot trim when events < epochs."
            )
            log_and_raise_error(logger, error_msg)
    else:
        from ..io.general import log_and_raise_error
        reason = "allow_misaligned_trim=False" if not allow_trim else f"mismatch ({diff}) exceeds max_tolerable_mismatch ({min_alignment_samples})"
        error_msg = (
            f"Alignment length mismatch for sub-{subject}, task-{task}: "
            f"events={n_events}, epochs={n_epochs}, diff={diff}. "
            f"Cannot proceed: {reason}"
        )
        log_and_raise_error(logger, error_msg)


def load_epochs_for_analysis(
    subject: str,
    task: str,
    align: str = "strict",
    preload: bool = False,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
    constants=None,
    use_cache: bool = True,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    _validate_load_epochs_params(align, config)
    
    allow_trim = bool(config.get("alignment.allow_misaligned_trim"))
    min_alignment_samples = int(config.get("alignment.min_alignment_samples"))
    
    epochs, events_df = _load_epochs_and_events(
        subject, task, deriv_root, bids_root, preload, config, constants, logger
    )
    if epochs is None:
        return None, None
    
    if events_df is None:
        if align == "strict":
            raise ValueError(f"Events TSV not found for sub-{subject}, task-{task}. Required when align='strict'")
        logger.warning("Events TSV not found; metadata will not be set.")
        return epochs, None
    
    logger.info(f"Loaded events: {len(events_df)} rows")
    logger.debug(f"Alignment parameters: allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}")
    
    _validate_event_columns(events_df, config, logger)
    
    try:
        aligned_events = align_events_to_epochs(events_df, epochs, logger=logger)
    except ValueError as err:
        if align == "strict":
            raise ValueError(
                f"Alignment failed for sub-{subject}, task-{task} in strict mode: {err}. "
                f"Alignment parameters: allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}"
            ) from err
        if align == "warn":
            logger.warning(f"Alignment failed: {err}")
        aligned_events = None
    
    if aligned_events is None:
        if align == "strict":
            raise ValueError(
                f"Could not align events to epochs for sub-{subject}, task-{task}. "
                f"Events have {len(events_df)} rows, epochs have {len(epochs)} epochs. "
                f"Alignment parameters: allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}. "
                f"This is required when align='strict'."
            )
        return epochs, None
    
    aligned_events = _handle_alignment_mismatch(
        aligned_events, epochs, subject, task, allow_trim, min_alignment_samples, logger
    )
    
    if len(aligned_events) != len(epochs):
        error_msg = (
            f"Failed to align events to epochs for sub-{subject}, task-{task}: "
            f"final length mismatch (events={len(aligned_events)}, epochs={len(epochs)}). "
            f"Cannot guarantee alignment."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    validation_result = validate_alignment(aligned_events, epochs, logger, strict=(align == "strict"), config=config)
    if not validation_result:
        if align == "strict":
            raise ValueError(
                f"Alignment validation failed for sub-{subject}, task-{task} with strict mode."
            )
        logger.warning("Alignment validation failed; returning None for aligned events")
        return epochs, None
    
    if use_cache:
        epochs._behavioral = aligned_events  # type: ignore[attr-defined]
    return epochs, aligned_events


def load_epochs_with_aligned_events(
    subject: str,
    task: str,
    config,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    preload: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    epochs, aligned = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=preload,
        deriv_root=deriv_root,
        bids_root=bids_root,
        logger=logger,
        config=config,
    )
    if epochs is None or aligned is None:
        return None, None
    return epochs, aligned


def pick_event_columns(df: pd.DataFrame, config) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"rating": None, "temperature": None, "pain_binary": None}
    rating_cols = config.get("event_columns.rating")
    temp_cols = config.get("event_columns.temperature")
    pain_cols = config.get("event_columns.pain_binary")

    for cand in rating_cols:
        if cand in df.columns:
            out["rating"] = cand
            break
    for cand in temp_cols:
        if cand in df.columns:
            out["temperature"] = cand
            break
    for cand in pain_cols:
        if cand in df.columns:
            out["pain_binary"] = cand
            break
    return out


def resolve_columns(
    df: pd.DataFrame,
    config: Optional[EEGConfig] = None,
    deriv_root: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if config is None:
        if deriv_root is None:
            raise ValueError("Either config or deriv_root must be provided to resolve_columns")
        from ..config.loader import load_settings
        config = load_settings()

    cols = pick_event_columns(df, config)
    return cols["pain_binary"], cols["temperature"], cols["rating"]




###################################################################
# Unified Subject Discovery
###################################################################

def get_available_subjects(
    config: EEGConfig,
    constants: Optional[Dict[str, Any]] = None,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    task: Optional[str] = None,
    discovery_sources: Optional[List[Literal["bids", "derivatives_epochs", "features"]]] = None,
    subject_discovery_policy: Literal["intersection", "union", "config_only"] = "intersection",
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if deriv_root is None:
        deriv_root = config.deriv_root
    
    if bids_root is None:
        bids_root = config.bids_root
    
    if task is None:
        task = config.get("project.task")
    
    if discovery_sources is None:
        discovery_sources = ["derivatives_epochs", "features"]
    
    subjects_from_config = config.subjects or []
    
    discovered_subjects = []
    
    if "bids" in discovery_sources:
        bids_subjects = _collect_subjects_from_bids(bids_root)
        discovered_subjects.append(("bids", bids_subjects))
        logger.debug(f"Discovered {len(bids_subjects)} subjects from BIDS")
    
    if "derivatives_epochs" in discovery_sources:
        epoch_subjects = _collect_subjects_from_derivatives_epochs(deriv_root, task, config=config, constants=constants)
        discovered_subjects.append(("derivatives_epochs", epoch_subjects))
        logger.debug(f"Discovered {len(epoch_subjects)} subjects from derivatives (clean epochs)")
    
    if "features" in discovery_sources:
        feature_subjects = _collect_subjects_from_features(deriv_root)
        discovered_subjects.append(("features", feature_subjects))
        logger.debug(f"Discovered {len(feature_subjects)} subjects from derivatives (features)")
    
    if not discovered_subjects:
        logger.warning("No subjects discovered from any source")
        return []
    
    if subject_discovery_policy == "config_only":
        resolved = subjects_from_config
        logger.info(f"Using config subjects only: {len(resolved)} subjects")
    elif subject_discovery_policy == "intersection":
        all_discovered = [subj for _, subjects in discovered_subjects for subj in subjects]
        if subjects_from_config:
            resolved = sorted(list(set(all_discovered) & set(subjects_from_config)))
            logger.info(
                f"Using intersection: {len(resolved)} subjects "
                f"(discovered={len(set(all_discovered))}, config={len(subjects_from_config)})"
            )
        else:
            if len(discovered_subjects) > 1:
                subject_sets = [set(subjects) for _, subjects in discovered_subjects]
                resolved = sorted(list(set.intersection(*subject_sets)))
                logger.info(
                    f"Using intersection of discovery sources: {len(resolved)} subjects "
                    f"(from {len(discovered_subjects)} sources)"
                )
            else:
                resolved = sorted(list(set(all_discovered)))
                logger.info(
                    f"Using discovered subjects: {len(resolved)} subjects "
                    f"(from {discovered_subjects[0][0]})"
                )
    elif subject_discovery_policy == "union":
        all_discovered = [subj for _, subjects in discovered_subjects for subj in subjects]
        resolved = sorted(list(set(all_discovered) | set(subjects_from_config)))
        logger.info(
            f"Using union: {len(resolved)} subjects "
            f"(discovered={len(set(all_discovered))}, config={len(subjects_from_config)})"
        )
    else:
        raise ValueError(f"Unknown policy: {subject_discovery_policy}. Must be 'intersection', 'union', or 'config_only'")
    
    return resolved


def parse_subject_args(
    args,
    config,
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if deriv_root is None:
        deriv_root = config.deriv_root
    
    if task is None:
        task = config.get("project.task")
    
    subjects: Optional[List[str]] = None
    
    if hasattr(args, 'group') and args.group is not None:
        g = args.group.strip()
        if g.lower() in {"all", "*", "@all"}:
            subjects = get_available_subjects(
                config=config,
                deriv_root=deriv_root,
                task=task,
                discovery_sources=["derivatives_epochs"],
                logger=logger,
            )
        else:
            candidates = [
                s.strip()
                for s in g.replace(";", ",").replace(" ", ",").split(",")
                if s.strip()
            ]
            subjects = []
            for s in candidates:
                if _find_clean_epochs_path(s, task, deriv_root=deriv_root, config=config) is not None:
                    subjects.append(s)
                else:
                    logger.warning(f"--group subject '{s}' has no cleaned epochs; skipping")
    elif hasattr(args, 'all_subjects') and args.all_subjects:
        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            task=task,
            discovery_sources=["derivatives_epochs"],
            logger=logger,
        )
    elif hasattr(args, 'subject') and args.subject:
        subjects = list(dict.fromkeys(args.subject))
    elif hasattr(args, 'subjects') and args.subjects:
        subjects = list(dict.fromkeys(args.subjects))
    
    if subjects is None:
        subjects = config.subjects if hasattr(config, 'subjects') and config.subjects else []
    
    return subjects


def load_decoding_data(
    subject: str,
    deriv_root: Path,
    config: EEGConfig,
    bids_root: Optional[Path] = None,
    task: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if task is None:
        task = config.get("project.task")
    
    if bids_root is None:
        bids_root = config.bids_root
    
    sub = f"sub-{subject}"
    feat_dir = deriv_root / sub / "eeg" / "features"
    X_path = feat_dir / "features_eeg_direct.tsv"
    y_path = feat_dir / "target_vas_ratings.tsv"
    manifest_path = _get_trial_alignment_manifest_path(deriv_root, subject)
    
    if not (X_path.exists() and y_path.exists()):
        raise FileNotFoundError(
            f"Missing features/targets for {sub}: {X_path} or {y_path} not found"
        )
    
    manifest = _load_trial_alignment_manifest(manifest_path, logger)
    expected_n_trials = len(manifest)
    
    X = pd.read_csv(X_path, sep="\t")
    y_df = pd.read_csv(y_path, sep="\t")
    
    if len(X) != expected_n_trials:
        raise ValueError(
            f"Feature count mismatch for subject {sub}, task {task}: "
            f"features have {len(X)} rows but trial_alignment.tsv specifies {expected_n_trials} trials. "
            f"This indicates a misalignment between feature extraction and trial manifest. "
            f"Re-run 03_feature_extraction.py to regenerate features with proper alignment."
        )
    
    if len(y_df) != expected_n_trials:
        raise ValueError(
            f"Target count mismatch for subject {sub}, task {task}: "
            f"targets have {len(y_df)} rows but trial_alignment.tsv specifies {expected_n_trials} trials. "
            f"This indicates a misalignment between target extraction and trial manifest. "
            f"Re-run 03_feature_extraction.py to regenerate features with proper alignment."
        )
    
    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    tgt_col = _pick_target_column(y_df, constants=constants)
    if tgt_col is None:
        pain_col, _, rating_col = resolve_columns(y_df, deriv_root=deriv_root, config=config)
        if rating_col is None:
            raise ValueError(
                f"No suitable target column found in {y_path} for subject {sub}, task {task}. "
                f"Available columns: {list(y_df.columns)}"
            )
        tgt_col = rating_col
    
    y = pd.to_numeric(y_df[tgt_col], errors="coerce")
    
    if len(X) != len(y) or len(X) == 0:
        raise ValueError(
            f"Length mismatch or empty data for subject {sub}, task {task}: X={len(X)}, y={len(y)}"
        )
    
    mask_valid = ~y.isna()
    n_dropped = int((~mask_valid).sum())
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} trials with NaN targets for {sub}")
    
    X = X.loc[mask_valid].reset_index(drop=True)
    y = y.loc[mask_valid].reset_index(drop=True)
    
    groups = np.array([sub] * len(X))
    
    meta = pd.DataFrame({
        "subject_id": [sub] * len(X),
        "trial_id": list(range(len(X))),
    })
    
    logger.info(
        f"Loaded data for {sub}: n_trials={len(X)}, n_features={X.shape[1]}"
    )
    
    return X, y, groups, meta


def load_multiple_subjects_decoding_data(
    deriv_root: Path,
    config: EEGConfig,
    subjects: Optional[List[str]] = None,
    subject_discovery_policy: Literal["intersection", "union", "config_only"] = "intersection",
    bids_root: Optional[Path] = None,
    task: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if task is None:
        task = config.get("project.task")
    
    if bids_root is None:
        bids_root = config.bids_root
    
    if subjects is None or subjects == ["all"]:
        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            bids_root=bids_root,
            task=task,
            discovery_sources=["features"],
            subject_discovery_policy=subject_discovery_policy,
            logger=logger,
        )
    else:
        logger.info(f"Using explicitly provided subjects: {subjects}")
    
    X_list, y_list, g_list = [], [], []
    trial_ids, subj_ids = [], []
    col_template = None
    n_found = 0
    
    for s in subjects:
        try:
            X, y, groups, meta = load_decoding_data(
                subject=s,
                deriv_root=deriv_root,
                config=config,
                bids_root=bids_root,
                task=task,
                logger=logger,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {s}: {e}")
            continue
        
        if col_template is None:
            col_template = list(X.columns)
        elif list(X.columns) != col_template:
            common = [c for c in col_template if c in X.columns]
            if not common:
                raise RuntimeError(f"No overlapping features for {s}")
            if len(common) < len(col_template):
                logger.warning(f"Using {len(common)} common features for {s}")
            X = X.loc[:, common]
            col_template = common
        
        n = len(X)
        X_list.append(X)
        y_list.append(y)
        g_list.extend([meta["subject_id"].iloc[0]] * n)
        trial_ids.extend(list(range(n)))
        subj_ids.extend([meta["subject_id"].iloc[0]] * n)
        n_found += 1
    
    if n_found == 0:
        raise RuntimeError(
            "No subjects with both features and targets were found. "
                "Run 03_feature_extraction.py first."
        )
    
    if col_template is None:
        raise RuntimeError("No feature columns detected.")
    
    X_list = [Xi.loc[:, col_template].copy() for Xi in X_list]
    X_all = pd.concat(X_list, axis=0, ignore_index=True)
    y_all = pd.concat(y_list, axis=0, ignore_index=True)
    groups = np.array(g_list)
    feature_names = list(X_all.columns)
    
    meta = pd.DataFrame({
        "subject_id": subj_ids,
        "trial_id": trial_ids,
    })
    
    logger.info(
        f"Aggregated features: n_trials={len(X_all)}, n_features={X_all.shape[1]}, n_subjects={n_found}"
    )
    
    return X_all, y_all, groups, meta


def load_epochs_with_targets(
    deriv_root: Path,
    config: EEGConfig,
    subjects: Optional[List[str]] = None,
    subject_discovery_policy: Literal["intersection", "union", "config_only"] = "intersection",
    task: str = "",
    bids_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[str, mne.Epochs, pd.Series]], List[str]]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if task == "":
        task = config.get("project.task")
    
    if bids_root is None:
        bids_root = config.bids_root
    
    if subjects is None or subjects == ["all"]:
        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            bids_root=bids_root,
            task=task,
            discovery_sources=["features"],
            subject_discovery_policy=subject_discovery_policy,
            logger=logger,
        )
    
    out: List[Tuple[str, mne.Epochs, pd.Series]] = []
    ch_sets: List[set] = []
    
    for s in subjects:
        sub = f"sub-{s}"
        epo_path = _find_clean_epochs_path(s, task, deriv_root=deriv_root, config=config)
        if epo_path is None or not Path(epo_path).exists():
            logger.warning(f"Clean epochs not found for {sub}; skipping.")
            continue
        
        epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
        
        epochs.set_montage(mne.channels.make_standard_montage("standard_1005"))
        
        if len(epochs.info.get("bads", [])) > 0:
            epochs.interpolate_bads(reset_bads=True)
        
        manifest_path = _get_trial_alignment_manifest_path(deriv_root, s)
        manifest = _load_trial_alignment_manifest(manifest_path, logger)
        
        if len(epochs) != len(manifest):
            raise ValueError(
                f"Epoch count mismatch for subject {sub}, task {task}: "
                f"epochs have {len(epochs)} trials but trial_alignment.tsv specifies {len(manifest)} trials. "
                f"This indicates the epochs file does not match the alignment used in feature extraction. "
                f"Re-run 03_feature_extraction.py to regenerate features with the current epochs."
            )
        
        _, aligned = load_epochs_for_analysis(s, task, align="strict", preload=False, deriv_root=deriv_root, config=config)
        if aligned is None or len(aligned) == 0:
            logger.warning(f"No aligned events/targets for {sub}; skipping.")
            continue
        
        _validate_event_columns(aligned, config, logger)
        
        constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
        tgt_col = _pick_target_column(aligned, constants=constants)
        if tgt_col is None:
            logger.warning(f"No suitable target column for {sub}; skipping.")
            continue
        
        y = pd.to_numeric(aligned[tgt_col], errors="coerce")
        
        if len(epochs) != len(y):
            logger.error(
                f"Epochs-target length mismatch for subject {sub}, task {task}: "
                f"epochs={len(epochs)}, y={len(y)}. "
                f"Cannot guarantee alignment. Skipping subject."
            )
            continue
        
        if len(epochs) == 0:
            logger.warning(f"No trials for {sub}; skipping.")
            continue
        
        out.append((sub, epochs, y))
        ch_sets.append(set([
            ch for ch in epochs.info["ch_names"]
            if epochs.get_channel_types(picks=[ch])[0] == "eeg"
        ]))
    
    if not out:
        raise RuntimeError("No epochs + targets could be loaded for any subject.")
    
    if not ch_sets:
        return out, []
    
    common_channels = sorted(set.intersection(*ch_sets)) if len(ch_sets) > 1 else sorted(ch_sets[0])
    
    return out, common_channels


###################################################################
# Epoch Selection and Processing Utilities
###################################################################

def apply_baseline(
    epochs: mne.Epochs,
    baseline_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    config=None
) -> mne.Epochs:
    from ..analysis.tfr import validate_baseline_indices
    times = np.asarray(epochs.times)
    validate_baseline_indices(times, baseline_window, logger=logger, config=config)
    baseline_start = float(baseline_window[0])
    baseline_end = float(baseline_window[1])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        return epochs.copy().apply_baseline((baseline_start, min(baseline_end, 0.0)))


def crop_epochs(
    epochs: mne.Epochs,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    logger: Optional[logging.Logger] = None
) -> mne.Epochs:
    if crop_tmin is None and crop_tmax is None:
        return epochs
    
    time_min = epochs.tmin if crop_tmin is None else float(crop_tmin)
    time_max = epochs.tmax if crop_tmax is None else float(crop_tmax)
    
    if time_max <= time_min:
        raise ValueError(f"Invalid crop window: tmin={time_min}, tmax={time_max}")
    
    epochs_copy = epochs.copy()
    if not getattr(epochs_copy, "preload", False):
        epochs_copy.load_data()
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        return epochs_copy.crop(tmin=time_min, tmax=time_max, include_tmax=include_tmax_in_crop)


def process_temperature_levels(epochs: mne.Epochs, temperature_column: str) -> Tuple[List, Dict, bool]:
    temperature_series = epochs.metadata[temperature_column]
    numeric_values = pd.to_numeric(temperature_series, errors="coerce")
    
    if numeric_values.notna().all():
        levels = np.sort(numeric_values.unique())
        labels = {
            value: str(int(value)) if float(value).is_integer() else str(value)
            for value in levels
        }
        return levels, labels, True
    
    levels = sorted(temperature_series.astype(str).unique())
    return levels, {}, False


def select_epochs_by_value(epochs: mne.Epochs, column: str, value) -> mne.Epochs:
    if epochs.metadata is None or column not in epochs.metadata.columns:
        raise ValueError(f"Column '{column}' not found in epochs.metadata")
    
    val_expr = value if isinstance(value, (int, float)) else f"'{value}'"
    selected = epochs[f"{column} == {val_expr}"]
    if len(selected) == 0:
        available_values = sorted(epochs.metadata[column].unique().tolist()) if column in epochs.metadata.columns else []
        raise ValueError(
            f"No epochs found for {column} == {value}. "
            f"Available values: {available_values}"
        )
    return selected


###################################################################
# Covariate Building Utilities
###################################################################

def build_covariate_matrix(
    events_df: Optional[pd.DataFrame],
    partial_covars: Optional[List[str]],
    config,
) -> Optional[pd.DataFrame]:
    if events_df is None or events_df.empty:
        return None
    
    covariate_names = list(partial_covars) if partial_covars else []
    if not covariate_names:
        covariate_names = extract_default_covariates(events_df, config)
    
    if not covariate_names:
        return None
    
    covariates_df = pd.DataFrame()
    for covariate_name in covariate_names:
        if covariate_name in events_df.columns:
            covariates_df[covariate_name] = pd.to_numeric(
                events_df[covariate_name], errors="coerce"
            )
    
    if covariates_df.empty:
        return None
    
    return covariates_df


def build_covariates_without_temp(
    covariates_df: Optional[pd.DataFrame],
    temp_col: Optional[str],
) -> Optional[pd.DataFrame]:
    if covariates_df is None:
        return None
    
    if not temp_col:
        return covariates_df.copy()
    
    covariates_without_temp = covariates_df.drop(columns=[temp_col], errors="ignore")
    if covariates_without_temp.shape[1] == 0:
        return None
    
    return covariates_without_temp


###################################################################
# Feature Loading Utilities
###################################################################

def load_subject_features(subjects: List[str], deriv_root: Path, logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """Loads feature files for multiple subjects.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject identifiers (without 'sub-' prefix)
    deriv_root : Path
        Root directory for derivatives
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping subject IDs to feature DataFrames
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not subjects:
        return {}
    
    subject_features = {}
    for subject in subjects:
        feature_path = deriv_features_path(deriv_root, subject) / "features_eeg_direct.tsv"
        if not feature_path.exists():
            logger.warning(f"Missing features for sub-{subject}: {feature_path}")
            continue
        subject_features[subject] = pd.read_csv(feature_path, sep="\t")
    return subject_features


def load_feature_bundle_for_subject(
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
]:
    """
    Load per-subject feature tables commonly used for visualization.
    Returns power, microstate, connectivity, aperiodic, PAC, PAC-trial, PAC-time, and ITPC dataframes (missing entries are None).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    features_dir = deriv_features_path(deriv_root, subject)
    power_path = features_dir / "features_eeg_direct.tsv"
    if not power_path.exists():
        logger.warning("Power features not found at %s", power_path)
        return None, None, None, None, None, None, None, None

    try:
        pow_df = read_tsv(power_path)
    except Exception as exc:
        logger.error("Failed to read power features from %s: %s", power_path, exc)
        return None, None, None, None, None, None, None, None

    def _safe_read(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return read_tsv(path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    ms_df = _safe_read(features_dir / "features_microstates.tsv")
    conn_df = _safe_read(features_dir / "features_connectivity.tsv")
    aper_df = _safe_read(features_dir / "features_aperiodic.tsv")
    pac_df = _safe_read(features_dir / "features_pac.tsv")
    pac_trials_df = _safe_read(features_dir / "features_pac_trials.tsv")
    pac_time_df = _safe_read(features_dir / "features_pac_time.tsv")
    itpc_df = _safe_read(features_dir / "features_itpc.tsv")

    return pow_df, ms_df, conn_df, aper_df, pac_df, pac_trials_df, pac_time_df, itpc_df


def load_feature_dfs_for_subjects(
    subjects: List[str],
    deriv_root: Path,
    input_filename_key: str,
    logger: logging.Logger,
    config,
) -> pd.DataFrame:
    """
    Load feature DataFrames for multiple subjects and concatenate them.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject identifiers (without 'sub-' prefix)
    deriv_root : Path
        Root directory for derivatives
    input_filename_key : str
        Config key for the feature filename (e.g., "group_aggregation.power_features_file")
    logger : logging.Logger
        Logger instance
    config
        Configuration object
        
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all subjects' features, with 'subject' column prepended
    """
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
        except Exception as e:
            logger.warning(f"Failed to read features for sub-{subject} at {file_path}: {e}")
            continue
    
    if not all_dfs:
        logger.warning(f"No features found for {input_filename_key} across any subject")
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def load_behavior_plot_features(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
) -> tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame], Optional[List[str]]]:
    """Load power features, targets, and metadata needed for behavior plots."""
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False,
        deriv_root=config.deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )

    _, pow_df, _, y, info = _load_features_and_targets(
        subject, task, config.deriv_root, config, epochs=epochs
    )

    if pow_df is None or y is None:
        logger.error("Features or targets not found for sub-%s", subject)
        return None, None, None, None

    power_bands = config.get("features.frequency_bands")
    return pow_df, y, info, power_bands


def load_behavior_stats_files(stats_dir: Path, logger: logging.Logger) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load rating and temperature correlation stats if available."""
    rating_stats = None
    temp_stats = None

    rating_path = stats_dir / "corr_stats_pow_roi_vs_rating.tsv"
    temp_path = stats_dir / "corr_stats_pow_roi_vs_temp.tsv"

    if rating_path.exists():
        try:
            rating_stats = read_tsv(rating_path)
        except Exception as exc:
            logger.warning("Failed to read rating stats: %s", exc)

    if temp_path.exists():
        try:
            temp_stats = read_tsv(temp_path)
        except Exception as exc:
            logger.warning("Failed to read temp stats: %s", exc)

    return rating_stats, temp_stats


def load_subject_data_for_summary(subjects: List[str], task: str, deriv_root: Path, config, logger: Optional[logging.Logger] = None) -> Tuple[Dict, Dict, Dict, Dict, bool]:
    """Loads subject data for overall band summary.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject identifiers (without 'sub-' prefix)
    task : str
        Task identifier
    deriv_root : Path
        Root directory for derivatives
    config : Any
        Config object
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    Tuple[Dict, Dict, Dict, Dict, bool]
        (rating_x, rating_y, temp_x, temp_y, has_temperature)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not subjects:
        return {}, {}, {}, {}, False
    
    from ..io.general import _find_clean_epochs_path
    
    rating_x = {}
    rating_y = {}
    temp_x = {}
    temp_y = {}
    has_temperature = False
    
    for subject in subjects:
        try:
            _temporal_df, power_df, _conn_df, ratings, _info = _load_features_and_targets(
                subject, task, deriv_root, config
            )
            ratings = pd.to_numeric(ratings, errors="coerce")
            
            epochs_path = _find_clean_epochs_path(subject, task, deriv_root=deriv_root, config=config)
            if not epochs_path:
                continue
            
            epochs, aligned = load_epochs_for_analysis(
                subject, task, align="strict", preload=False,
                deriv_root=deriv_root, bids_root=config.bids_root, config=config
            )
            if epochs is None:
                continue
        except (FileNotFoundError, ValueError) as e:
            logger.debug(f"Skipping subject {subject} due to expected error: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error loading data for subject {subject}: {e}", exc_info=True)
            raise
        
        temperature = None
        if aligned is not None:
            temp_columns = config.get("event_columns.temperature")
            temp_col = _pick_first_column(aligned, temp_columns)
            if temp_col:
                temperature = pd.to_numeric(aligned[temp_col], errors="coerce")
                has_temperature = True
        
        power_bands = config.get("power.bands_to_use")
        for band in power_bands:
            power_cols = [col for col in power_df.columns if col.startswith(f"pow_{band}_")]
            if not power_cols:
                continue
            
            band_power = power_df[power_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
            rating_x.setdefault(band, []).append(band_power)
            rating_y.setdefault(band, []).append(ratings.to_numpy())
            
            if temperature is not None:
                temp_x.setdefault(band, []).append(band_power)
                temp_y.setdefault(band, []).append(temperature.to_numpy())
    
    return rating_x, rating_y, temp_x, temp_y, has_temperature


###################################################################
# Data Transformation Utilities
###################################################################

def flatten_lower_triangles(connectivity_trials: np.ndarray, labels: Optional[np.ndarray], prefix: str) -> Tuple[pd.DataFrame, List[str]]:
    """Flattens lower triangle of connectivity matrices.
    
    Parameters
    ----------
    connectivity_trials : np.ndarray
        3D array with shape (trials, nodes, nodes)
    labels : Optional[np.ndarray]
        Node labels array, optional
    prefix : str
        Prefix for column names
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (flattened DataFrame, column names list)
    """
    if connectivity_trials.ndim != 3:
        raise ValueError("Connectivity array must be 3D (trials, nodes, nodes)")
    
    n_trials, n_nodes, _ = connectivity_trials.shape
    lower_tri_i, lower_tri_j = np.tril_indices(n_nodes, k=-1)
    flattened_data = connectivity_trials[:, lower_tri_i, lower_tri_j]

    if labels is not None and len(labels) == n_nodes:
        pair_names = [f"{labels[i]}__{labels[j]}" for i, j in zip(lower_tri_i, lower_tri_j)]
    else:
        pair_names = [f"n{i}_n{j}" for i, j in zip(lower_tri_i, lower_tri_j)]
    
    column_names = [f"{prefix}_{pair}" for pair in pair_names]
    return pd.DataFrame(flattened_data), column_names


def align_feature_blocks(blocks: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Aligns feature DataFrames to same length.
    
    Parameters
    ----------
    blocks : List[pd.DataFrame]
        List of feature DataFrames
        
    Returns
    -------
    List[pd.DataFrame]
        List of aligned DataFrames (all same length)
    """
    if not blocks:
        return []
    
    valid_blocks = [block for block in blocks if block is not None and not block.empty]
    if not valid_blocks:
        return []
    
    min_trials = min(len(block) for block in valid_blocks)
    aligned_blocks = [block.iloc[:min_trials, :] for block in valid_blocks]
    return aligned_blocks


###################################################################
# Data Filtering & Validation
###################################################################

def filter_finite_targets(
    indices: np.ndarray,
    targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    target_values = targets[indices]
    finite_mask = np.isfinite(target_values)
    filtered_indices = indices[finite_mask]
    filtered_targets = target_values[finite_mask]
    return filtered_indices, filtered_targets

def validate_trial_alignment_manifest(
    aligned_events: pd.DataFrame,
    features_dir: Path,
    logger: logging.Logger,
) -> None:
    """Validate trial alignment using manifest file.
    
    Parameters
    ----------
    aligned_events : pd.DataFrame
        Aligned events DataFrame
    features_dir : Path
        Features directory containing trial_alignment.tsv
    logger : logging.Logger
        Logger instance
        
    Raises
    ------
    ValueError
        If manifest not found or trial counts don't match
    """
    manifest_path = features_dir / "trial_alignment.tsv"
    if not manifest_path.exists():
        raise ValueError(f"Trial alignment manifest not found: {manifest_path}")
    
    manifest = pd.read_csv(manifest_path, sep="\t")
    if len(manifest) != len(aligned_events):
        raise ValueError(
            f"Trial count mismatch: manifest has {len(manifest)} trials, "
            f"aligned_events has {len(aligned_events)} trials"
        )
    logger.info(f"Trial alignment validated: {len(manifest)} trials")


def register_feature_block(
    name: str,
    block: Optional[Union[pd.DataFrame, pd.Series]],
    registry: Dict[str, pd.DataFrame],
    lengths: Dict[str, int],
) -> None:
    """Register a feature block in the registry and record its length.
    
    Parameters
    ----------
    name : str
        Feature block name
    block : Optional[Union[pd.DataFrame, pd.Series]]
        Feature block to register
    registry : Dict[str, pd.DataFrame]
        Dictionary to store registered blocks
    lengths : Dict[str, int]
        Dictionary to store block lengths
    """
    if block is None:
        lengths[name] = 0
        return
    
    if isinstance(block, pd.Series):
        block_df = block.to_frame()
        block_length = len(block)
    else:
        block_df = block
        block_length = len(block_df) if not block_df.empty else 0
    
    lengths[name] = block_length
    if block_length > 0:
        registry[name] = block_df.reset_index(drop=True)


def validate_feature_block_lengths(
    lengths: Dict[str, int],
    logger: logging.Logger,
    critical_features: Optional[List[str]] = None,
) -> None:
    """Validate that feature blocks have consistent lengths.
    
    Parameters
    ----------
    lengths : Dict[str, int]
        Dictionary mapping feature block names to their lengths
    logger : logging.Logger
        Logger instance
    critical_features : Optional[List[str]]
        List of critical feature names that must not be empty.
        Defaults to ["power", "baseline", "target"]
        
    Raises
    ------
    ValueError
        If lengths are inconsistent or critical features are empty
    """
    if critical_features is None:
        critical_features = ["power", "baseline", "target"]
    
    unique_lengths = set(lengths.values())
    nonzero_lengths = {length for length in unique_lengths if length > 0}
    
    if len(nonzero_lengths) > 1:
        mismatch = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            "Feature blocks have mismatched trial counts and cannot be safely aligned: "
            f"{mismatch}. Inspect the per-feature drop logs (e.g., features/dropped_trials.tsv) "
            "to ensure each extractor operates on the same trial manifest."
        )
    
    empty_blocks = [name for name, length in lengths.items() if length == 0]
    nonempty_blocks = [name for name, length in lengths.items() if length > 0]
    
    if not empty_blocks or not nonempty_blocks:
        return
    
    empty_critical = [name for name in empty_blocks if name in critical_features]
    
    if empty_critical:
        error_msg = (
            f"Critical feature blocks are empty while others are not: empty={empty_blocks}, "
            f"non-empty={nonempty_blocks}. Critical empty blocks: {empty_critical}. "
            f"This indicates extraction failures and prevents valid analysis. "
            f"Fix feature extraction before proceeding."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.warning(
        f"Some non-critical feature blocks are empty while others are not: empty={empty_blocks}, "
        f"non-empty={nonempty_blocks}. Analysis will proceed but may be incomplete."
    )


def validate_trial_alignment(
    events: pd.DataFrame,
    kept_indices: np.ndarray,
    meta_trial_ids: np.ndarray,
    subject_label: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if "trial_number" not in events.columns:
        return
    
    ev_trial_numbers = pd.to_numeric(events["trial_number"], errors="coerce").to_numpy()
    ev_trials_selected = ev_trial_numbers[kept_indices]
    meta_trial_numbers = meta_trial_ids.astype(float) + 1.0
    
    if len(ev_trials_selected) != len(meta_trial_numbers):
        return
    
    mismatches = ~np.isclose(ev_trials_selected, meta_trial_numbers, rtol=1e-5, atol=1e-3)
    if not np.any(mismatches):
        return
    
    n_mismatch = int(np.sum(mismatches))
    first_mismatch_idx = np.where(mismatches)[0][0] if np.any(mismatches) else -1
    logger.error(
        f"{subject_label}: Trial identity mismatch after truncation: {n_mismatch}/{len(meta_trial_numbers)} "
        f"trials do not match between events (trial_number) and metadata (trial_id). "
        f"This may indicate misalignment. First mismatch at index {first_mismatch_idx}"
    )


###################################################################
# Epoch Data Extraction
###################################################################

def extract_epoch_data_block(
    indices: np.ndarray,
    trial_records: List[Tuple[str, int]],
    aligned_epochs: Dict[str, mne.Epochs]
) -> np.ndarray:
    X_list = []
    for i in indices:
        sub_i, ti = trial_records[int(i)]
        try:
            X_i = aligned_epochs[sub_i].get_data(picks="eeg", reject_by_annotation=None)[ti]
        except TypeError:
            X_i = aligned_epochs[sub_i].get_data(picks="eeg")[ti]
        X_list.append(X_i)
    return np.stack(X_list, axis=0)

def prepare_trial_records_from_epochs(
    tuples: List[Tuple[str, mne.Epochs, pd.Series]]
) -> Tuple[List[Tuple[str, int]], np.ndarray, np.ndarray, Dict[str, mne.Epochs], Dict[str, pd.Series]]:
    trial_records = []
    y_all_list = []
    groups_list = []
    subj_to_epochs = {}
    subj_to_y = {}
    
    for sub, epochs, y in tuples:
        n = min(len(epochs), len(y))
        if n == 0:
            continue
        subj_to_epochs[sub] = epochs
        subj_to_y[sub] = pd.to_numeric(y.iloc[:n], errors="coerce")
        for ti in range(n):
            trial_records.append((sub, ti))
            y_all_list.append(float(subj_to_y[sub].iloc[ti]))
            groups_list.append(sub)
    
    if len(trial_records) == 0:
        raise RuntimeError("No trial data available.")
    
    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    
    return trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y

def extract_epoch_data(epochs: Any, picks: np.ndarray) -> np.ndarray:
    try:
        return epochs.get_data(picks=picks)
    except TypeError:
        return epochs.get_data()[:, picks, :]


###################################################################
# Metadata Processing
###################################################################

def load_kept_indices(subject_label: str, deriv_root: Path, n_events: int, logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    dropped_path = deriv_root / subject_label / "eeg" / "features" / "dropped_trials.tsv"
    
    if not dropped_path.exists():
        return None
    
    dropped_df = pd.read_csv(dropped_path, sep="\t")
    if "original_index" not in dropped_df.columns:
        return None
    
    dropped_indices_raw = pd.to_numeric(dropped_df["original_index"], errors="coerce").dropna()
    if len(dropped_indices_raw) == 0:
        return None
    
    dropped_indices = set(dropped_indices_raw.astype(int).tolist())
    kept_indices = np.array([i for i in range(n_events) if i not in dropped_indices])
    logger.info(f"{subject_label}: {len(dropped_indices)} trials dropped, {len(kept_indices)} kept")
    return kept_indices

def process_subject_metadata(
    subject_label: str,
    meta_indices: np.ndarray,
    events: pd.DataFrame,
    kept_indices: np.ndarray,
    meta_trial_ids: np.ndarray,
    config: dict,
    temps_out: np.ndarray,
    trials_out: np.ndarray,
    blocks_out: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    n_subject_trials = len(meta_indices)
    
    if len(kept_indices) < n_subject_trials:
        logger.warning(
            f"{subject_label}: kept_indices ({len(kept_indices)}) < feature trials ({n_subject_trials}); "
            f"using first {len(kept_indices)}"
        )
        kept_indices = kept_indices[:n_subject_trials]
    elif len(kept_indices) > n_subject_trials:
        logger.warning(
            f"{subject_label}: More kept event rows ({len(kept_indices)}) than feature trials "
            f"({n_subject_trials}); truncating"
        )
        kept_indices = kept_indices[:n_subject_trials]
    
    validate_trial_alignment(events, kept_indices, meta_trial_ids, subject_label, logger)
    
    _, temp_col, _ = resolve_columns(events, config=config)
    if temp_col is not None:
        temps = pd.to_numeric(events[temp_col], errors="coerce").to_numpy()
        temps_out[meta_indices] = temps[kept_indices]
    
    if "trial_number" in events.columns:
        trials = pd.to_numeric(events["trial_number"], errors="coerce").to_numpy()
        trials_out[meta_indices] = trials[kept_indices]
    else:
        trials_out[meta_indices] = meta_trial_ids.astype(float) + 1.0
    
    subjects_with_blocks = 0
    total_trials_with_blocks = 0
    
    if "run_id" in events.columns:
        blocks = pd.to_numeric(events["run_id"], errors="coerce").to_numpy()
        blocks_out[meta_indices] = blocks[kept_indices]
        
        unique_blocks = np.unique(blocks_out[meta_indices][np.isfinite(blocks_out[meta_indices])])
        logger.info(
            f"{subject_label}: Assigned blocks with {len(unique_blocks)} unique values: "
            f"{unique_blocks.tolist()}"
        )
        
        valid_blocks = np.isfinite(blocks_out[meta_indices])
        if valid_blocks.sum() == n_subject_trials:
            subjects_with_blocks = 1
            total_trials_with_blocks = n_subject_trials
            logger.debug(f"Complete block info found for {subject_label}")
        elif valid_blocks.any():
            logger.warning(f"Partial block info for {subject_label}: {valid_blocks.sum()}/{n_subject_trials} trials")
        else:
            logger.warning(f"No valid block values for {subject_label}")
    
    return subjects_with_blocks, total_trials_with_blocks


###################################################################
# Column Extraction
###################################################################

def extract_roi_columns(
    roi: str,
    channels: List[str],
    band: str,
    band_columns: set,
) -> Optional[List[str]]:
    if not roi or not channels or not band:
        return None
    
    roi_columns = [
        f"pow_{band}_{ch}" for ch in channels if f"pow_{band}_{ch}" in band_columns
    ]
    return roi_columns if roi_columns else None

def extract_rating_array_for_tf(
    aligned_events: pd.DataFrame,
    config,
    logger,
) -> Optional[np.ndarray]:
    from ..io.general import _pick_first_column
    rating_col = _pick_first_column(aligned_events, config.get("event_columns.rating"))
    if rating_col is None:
        logger.error("No rating column found for TF correlation computation")
        return None

    y = pd.to_numeric(aligned_events[rating_col], errors="coerce")
    if y.isna().all():
        logger.error("All behavioral ratings are NaN; skipping TF correlation computation")
        return None
    
    return y.to_numpy(dtype=float)

def extract_measure_prefixes(column_names: List[str]) -> List[str]:
    return sorted({"_".join(c.split("_")[:2]) for c in column_names})

def extract_node_names_from_prefix(
    prefix: str,
    prefix_columns: List[str],
    min_nodes_for_heatmap: Optional[int] = None,
) -> Optional[Tuple[List[str], Dict[str, int]]]:
    pair_names = [col.split(prefix + "_", 1)[-1] for col in prefix_columns]
    node_names = sorted({name for pair in pair_names for name in pair.split("__")})
    
    if min_nodes_for_heatmap is None:
        # Default to 2 if not provided, using a safe default as this is a utility function
        min_nodes_for_heatmap = 2
    
    if len(node_names) < min_nodes_for_heatmap:
        return None
    
    node_to_index = {name: index for index, name in enumerate(node_names)}
    return node_names, node_to_index


###################################################################
# Data Structure Building
###################################################################

def build_per_subject_indices(
    groups: np.ndarray,
) -> Dict[str, np.ndarray]:
    unique_groups = np.unique(groups)
    per_subject_indices = {}
    for group in unique_groups:
        per_subject_indices[str(group)] = np.where(groups == group)[0]
    return per_subject_indices

def build_summary_map_for_prefix(
    prefix: str,
    prefix_columns: List[str],
    roi_map: Dict[str, List[str]],
) -> Dict[Tuple[str, str], List[str]]:
    from ..analysis.tfr import build_atlas_rois_from_nodes, build_summary_map_from_roi_nodes
    
    pair_names = [c.split(prefix + "_", 1)[-1] for c in prefix_columns]
    nodes = sorted({nm for pair in pair_names for nm in pair.split("__")})
    
    atlas_roi_map = build_atlas_rois_from_nodes(nodes, hemisphere_split=True)
    if atlas_roi_map:
        return build_summary_map_from_roi_nodes(atlas_roi_map, prefix, prefix_columns)
    
    if roi_map:
        return build_summary_map_from_roi_nodes(roi_map, prefix, prefix_columns)
    
    return {}


###################################################################
# File Loading
###################################################################

def load_channel_correlations(subject: str, band: str, deriv_root: Path, 
                               correlation_type: str) -> Optional[pd.DataFrame]:
    from ..io.general import deriv_stats_path
    
    if not subject or not band or correlation_type not in ("rating", "temp"):
        return None
    
    if correlation_type == "rating":
        file_path = deriv_stats_path(deriv_root, subject) / f"corr_stats_pow_{band}_vs_rating.tsv"
    else:
        file_path = deriv_stats_path(deriv_root, subject) / f"corr_stats_pow_{band}_vs_temp.tsv"
    
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path, sep="\t")
    if df.empty or "channel" not in df.columns or "r" not in df.columns:
        return None
    
    return df

def load_connectivity_files(subjects: List[str], deriv_root: Path) -> Dict[str, List[pd.DataFrame]]:
    from ..io.general import deriv_stats_path
    
    if not subjects:
        return {}
    
    connectivity_by_measure = {}
    for subject in subjects:
        subject_stats = deriv_stats_path(deriv_root, subject)
        if not subject_stats.exists():
            continue
        
        for conn_file in subject_stats.glob("corr_stats_conn_roi_summary_*_vs_rating.tsv"):
            df = pd.read_csv(conn_file, sep="\t")
            if df.empty or "measure_band" not in df.columns:
                continue
            measure_band = str(df["measure_band"].iloc[0])
            connectivity_by_measure.setdefault(measure_band, []).append(df)
    
    return connectivity_by_measure


###################################################################
# Data Extraction Utilities
###################################################################

def extract_channel_importance_from_coefficients(coef_matrix: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    from ..io.decoding import parse_pow_feature
    
    channel_band_to_indices = {}
    for idx, feat in enumerate(feature_names):
        parsed = parse_pow_feature(feat)
        if parsed:
            band, channel = parsed
            key = (channel, band)
            if key not in channel_band_to_indices:
                channel_band_to_indices[key] = []
            channel_band_to_indices[key].append(idx)
    
    channel_to_all_indices = {}
    for (channel, band), indices in channel_band_to_indices.items():
        if channel not in channel_to_all_indices:
            channel_to_all_indices[channel] = []
        channel_to_all_indices[channel].extend(indices)
    
    n_folds = coef_matrix.shape[0]
    channel_importance_data = []
    
    for channel, indices in channel_to_all_indices.items():
        channel_coefs = coef_matrix[:, indices]
        channel_mean_abs = np.nanmean(np.abs(channel_coefs), axis=1)
        
        for fold_idx in range(n_folds):
            if np.isfinite(channel_mean_abs[fold_idx]):
                channel_importance_data.append({
                    'channel': channel,
                    'importance': float(channel_mean_abs[fold_idx]),
                    'fold': fold_idx,
                })
    
    return pd.DataFrame(channel_importance_data)


def extract_band_channel_vectors(df: pd.DataFrame, band_names: List[str]) -> Dict[str, Dict[str, float]]:
    band_vectors = {}
    for band in band_names:
        band_str = str(band)
        cols = [c for c in df.columns if str(c).startswith(f"pow_{band_str}_")]
        if not cols:
            continue
        series = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        channel_means = {
            c.replace(f"pow_{band_str}_", ""): float(v)
            for c, v in series.items() if np.isfinite(v)
        }
        if channel_means:
            band_vectors[band_str] = channel_means
    return band_vectors


def validate_aligned_events_length(aligned_events: Optional[pd.DataFrame], epochs, logger: Optional[logging.Logger] = None) -> bool:
    if aligned_events is None:
        if logger:
            logger.error("Alignment failed for plotting function: aligned_events is None")
        return False
    
    if len(aligned_events) != len(epochs):
        if logger:
            logger.error(f"Alignment failed: events ({len(aligned_events)}) != epochs ({len(epochs)})")
        return False
    
    return True


def prepare_partial_correlation_data(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    Z_df: pd.DataFrame,
    pooling_strategy: str
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.DataFrame]]:
    from ..analysis.stats import apply_pooling_strategy
    
    xi = pd.Series(np.asarray(x_arr))
    yi = pd.Series(np.asarray(y_arr))
    Zi = Z_df.copy()
    
    n = min(len(xi), len(yi), len(Zi))
    xi = xi.iloc[:n]
    yi = yi.iloc[:n]
    Zi = Zi.iloc[:n].copy()
    
    mask = xi.notna() & yi.notna()
    xi = xi[mask]
    yi = yi[mask]
    Zi = Zi.loc[mask]
    
    if xi.empty or yi.empty:
        return None, None, None
    
    xi, yi = apply_pooling_strategy(xi, yi, pooling_strategy)
    if xi.empty or yi.empty:
        return None, None, None
    
    return xi.reset_index(drop=True), yi.reset_index(drop=True), Zi.reset_index(drop=True)


def extract_common_dataframe_columns(partial_Z: List[pd.DataFrame]) -> List[str]:
    if not partial_Z:
        return []
    
    common_cols = set(partial_Z[0].columns)
    for df in partial_Z[1:]:
        common_cols &= set(df.columns)
    
    return sorted(common_cols)


def prepare_group_partial_residuals_data(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    Z_lists: List[Optional[pd.DataFrame]],
    has_Z_flags: List[bool],
    subj_order: List[str],
    pooling_strategy: str,
    subject_fixed_effects: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    if not any(has_Z_flags):
        return None, None, None
    
    partial_x: List[pd.Series] = []
    partial_y: List[pd.Series] = []
    partial_Z: List[pd.DataFrame] = []
    partial_subj_ids: List[str] = []
    
    for idx, (has_cov, Z_df, x_arr, y_arr) in enumerate(zip(has_Z_flags, Z_lists, x_lists, y_lists)):
        if not has_cov or Z_df is None:
            continue
        
        xi, yi, Zi = prepare_partial_correlation_data(x_arr, y_arr, Z_df, pooling_strategy)
        if xi is None or yi is None:
            continue
        
        partial_x.append(xi)
        partial_y.append(yi)
        partial_Z.append(Zi)
        subj_id = subj_order[idx] if idx < len(subj_order) else str(idx)
        partial_subj_ids.extend([subj_id] * len(xi))
    
    if not partial_Z:
        return None, None, None
    
    common_cols = extract_common_dataframe_columns(partial_Z)
    if common_cols:
        partial_Z = [df[common_cols] for df in partial_Z]
    
    Z_all_vis = pd.concat(partial_Z, ignore_index=True)
    x_all_partial = pd.concat(partial_x, ignore_index=True)
    y_all_partial = pd.concat(partial_y, ignore_index=True)
    
    if subject_fixed_effects:
        Z_all_vis = _add_subject_dummies_if_needed(Z_all_vis, partial_subj_ids)
    
    return Z_all_vis, x_all_partial, y_all_partial


def _add_subject_dummies_if_needed(Z_df: pd.DataFrame, subj_ids: List[str]) -> pd.DataFrame:
    if not subj_ids or len(set(subj_ids)) <= 1:
        return Z_df
    
    Z_with_dummies = Z_df.copy()
    unique_subjects = sorted(set(subj_ids))
    for subj in unique_subjects[1:]:
        Z_with_dummies[f"subj_{subj}"] = (np.array(subj_ids) == subj).astype(int)
    
    return Z_with_dummies


def prepare_group_band_roi_data(
    x_list: List[np.ndarray],
    y_list: List[np.ndarray],
    Z_list: List[Optional[pd.DataFrame]],
    has_Z_flag: List[bool],
    subj_ord: List[str],
    pooling_strategy: str,
    subject_fixed_effects: bool,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[np.ndarray]]:
    from ..analysis.stats import prepare_group_data
    
    x_all, y_all, vis_subj_ids = prepare_group_data(
        x_list, y_list, subj_ord, pooling_strategy
    )
    if x_all.empty:
        return None, None, None, None, None, None
    
    Z_all_vis, x_all_partial, y_all_partial = prepare_group_partial_residuals_data(
        x_list, y_list, Z_list, has_Z_flag, subj_ord,
        pooling_strategy, subject_fixed_effects
    )
    
    if subject_fixed_effects and Z_all_vis is not None:
        Z_all_vis = _add_subject_dummies_if_needed(Z_all_vis, vis_subj_ids.tolist())
    
    return x_all, y_all, Z_all_vis, x_all_partial, y_all_partial, vis_subj_ids


def extract_aligned_column_vector(tfr, events_df: Optional[pd.DataFrame], column_name: str, n: int) -> Optional[pd.Series]:
    if getattr(tfr, "metadata", None) is not None and column_name in tfr.metadata.columns:
        return pd.to_numeric(tfr.metadata.iloc[:n][column_name], errors="coerce")
    if events_df is not None and column_name in events_df.columns:
        return pd.to_numeric(events_df.iloc[:n][column_name], errors="coerce")
    return None


def extract_pain_vector(tfr, events_df: Optional[pd.DataFrame], pain_col: Optional[str], n: int) -> Optional[pd.Series]:
    pain_vec = extract_aligned_column_vector(tfr, events_df, pain_col, n)
    if pain_vec is None:
        return None
    return pd.to_numeric(pain_vec, errors="coerce").fillna(0).astype(int)


def extract_pain_vector_array(tfr, events_df: Optional[pd.DataFrame], pain_col: Optional[str], n: int) -> Optional[np.ndarray]:
    pain_vec = extract_pain_vector(tfr, events_df, pain_col, n)
    if pain_vec is None:
        return None
    return pain_vec.values


def extract_temperature_series(tfr, events_df: Optional[pd.DataFrame], temp_col: Optional[str], n: int) -> Optional[pd.Series]:
    if temp_col is None:
        return None
    return extract_aligned_column_vector(tfr, events_df, temp_col, n)


def compute_aligned_data_length(tfr, events_df: Optional[pd.DataFrame]) -> int:
    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df) if events_df is not None else n_epochs
    return min(n_epochs, n_meta)


def create_temperature_masks(
    temp_series: pd.Series, 
    temperature_rounding_decimals: Optional[int] = None, 
    min_temperatures_required: Optional[int] = None,
    config: Optional[EEGConfig] = None,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    if temp_series is None:
        return None, None, None, None
    
    if temperature_rounding_decimals is None or min_temperatures_required is None:
        config = ensure_config(config)
        temperature_rounding_decimals = temperature_rounding_decimals or int(get_config_value(config, "plotting.tfr.temperature_rounding_decimals", 1))
        min_temperatures_required = min_temperatures_required or int(get_config_value(config, "plotting.validation.min_temperatures_required", 2))
    
    s_round = pd.to_numeric(temp_series, errors="coerce").round(temperature_rounding_decimals)
    temps = sorted(map(float, s_round.dropna().unique()))
    if len(temps) < min_temperatures_required:
        return None, None, None, None
    t_min = float(min(temps))
    t_max = float(max(temps))
    mask_min = np.asarray(s_round == round(t_min, temperature_rounding_decimals), dtype=bool)
    mask_max = np.asarray(s_round == round(t_max, temperature_rounding_decimals), dtype=bool)
    return t_min, t_max, mask_min, mask_max


def get_temperature_range(
    temp_series: pd.Series, 
    temperature_rounding_decimals: Optional[int] = None, 
    min_temperatures_required: Optional[int] = None,
    config: Optional[EEGConfig] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if temp_series is None:
        return None, None
    
    if temperature_rounding_decimals is None or min_temperatures_required is None:
        config = ensure_config(config)
        temperature_rounding_decimals = temperature_rounding_decimals or int(get_config_value(config, "plotting.tfr.temperature_rounding_decimals", 1))
        min_temperatures_required = min_temperatures_required or int(get_config_value(config, "plotting.validation.min_temperatures_required", 2))
    
    s_round = pd.to_numeric(temp_series, errors="coerce").round(temperature_rounding_decimals)
    temps = sorted(map(float, s_round.dropna().unique()))
    if len(temps) < min_temperatures_required:
        return None, None
    t_min = float(min(temps))
    t_max = float(max(temps))
    return t_min, t_max


def create_temperature_masks_from_range(
    temp_series: pd.Series, 
    t_min: float, 
    t_max: float, 
    temperature_rounding_decimals: Optional[int] = None,
    config: Optional[EEGConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if temp_series is None or t_min is None or t_max is None:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    
    if temperature_rounding_decimals is None:
        config = ensure_config(config)
        temperature_rounding_decimals = int(get_config_value(config, "plotting.tfr.temperature_rounding_decimals", 1))
    
    s_round = pd.to_numeric(temp_series, errors="coerce").round(temperature_rounding_decimals)
    mask_min = np.asarray(s_round == round(t_min, temperature_rounding_decimals), dtype=bool)
    mask_max = np.asarray(s_round == round(t_max, temperature_rounding_decimals), dtype=bool)
    return mask_min, mask_max


def extract_time_frequency_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if 'frequency' not in df.columns or 'time' not in df.columns:
        return np.array([]), np.array([])
    freqs = np.unique(np.round(df["frequency"].to_numpy(dtype=float), 6))
    times = np.unique(np.round(df["time"].to_numpy(dtype=float), 6))
    return freqs, times


def extract_importance_column(importance_df: pd.DataFrame, top_n: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if 'mean_abs_shap' in importance_df.columns:
        df_sorted = importance_df.sort_values('mean_abs_shap', ascending=False).head(top_n)
        return df_sorted['mean_abs_shap'].values, 'Mean |SHAP value|'
    
    if 'importance' in importance_df.columns:
        df_sorted = importance_df.sort_values('importance', ascending=False).head(top_n)
        return df_sorted['importance'].values, 'Importance (ΔR²)'
    
    return None, None


def validate_data_not_empty(
    X_all: Optional[pd.DataFrame],
    y_all: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    meta: Optional[pd.DataFrame],
) -> None:
    """Validate that data arrays are not None or empty.
    
    Parameters
    ----------
    X_all : Optional[pd.DataFrame]
        Feature matrix
    y_all : Optional[np.ndarray]
        Target vector
    groups : Optional[np.ndarray]
        Group labels
    meta : Optional[pd.DataFrame]
        Metadata DataFrame
        
    Raises
    ------
    ValueError
        If any input is None or empty
    """
    if X_all is None or X_all.empty:
        raise ValueError("X_all is None or empty after loading")
    if y_all is None or len(y_all) == 0:
        raise ValueError("y_all is None or empty after loading")
    if groups is None or len(groups) == 0:
        raise ValueError("groups is None or empty after loading")
    if meta is None or meta.empty:
        raise ValueError("meta is None or empty after loading")


def validate_data_lengths(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    """Validate that all data arrays have matching lengths.
    
    Parameters
    ----------
    X_all : pd.DataFrame
        Feature matrix
    y_all : np.ndarray
        Target vector
    groups : np.ndarray
        Group labels
    meta : pd.DataFrame
        Metadata DataFrame
        
    Raises
    ------
    ValueError
        If lengths don't match
    """
    if len(X_all) != len(y_all) or len(X_all) != len(groups) or len(X_all) != len(meta):
        raise ValueError(
            f"Length mismatch: X_all={len(X_all)}, y_all={len(y_all)}, "
            f"groups={len(groups)}, meta={len(meta)}"
        )


def validate_trial_ids(meta: pd.DataFrame, logger: Optional[logging.Logger] = None) -> None:
    """Validate that trial_id column exists and contains unique values.
    
    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame
    logger : Optional[logging.Logger]
        Logger instance
        
    Raises
    ------
    ValueError
        If trial_id column is missing or contains duplicates
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    if "trial_id" not in meta.columns:
        error_msg = (
            "Trial ID column not found in meta. Cannot validate that features and targets "
            "correspond to the same trials. This is required for valid decoding results."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    trial_ids = meta["trial_id"].values
    unique_count = len(np.unique(trial_ids))
    
    if unique_count != len(trial_ids):
        error_msg = (
            f"Trial identity validation failed: duplicate trial_id values found in meta. "
            f"Found {len(trial_ids)} total but only {unique_count} unique. "
            f"This indicates features and targets may not correspond to the same trials."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Trial identity validation passed: {len(trial_ids)} unique trial IDs verified")


def validate_sufficient_subjects(
    groups: np.ndarray, 
    min_subjects: Optional[int] = None,
    config: Optional[EEGConfig] = None,
) -> None:
    """Validate that there are sufficient subjects for analysis.
    
    Parameters
    ----------
    groups : np.ndarray
        Group labels (subject IDs)
    min_subjects : Optional[int]
        Minimum number of subjects required (default: from config)
    config : Optional[EEGConfig]
        Configuration dictionary
        
    Raises
    ------
    RuntimeError
        If insufficient subjects
    """
    if min_subjects is None:
        config = ensure_config(config)
        min_subjects = int(get_config_value(config, "analysis.min_subjects_for_group", 2))
    
    unique_subjects = len(np.unique(groups))
    if unique_subjects < min_subjects:
        raise RuntimeError(f"Need at least {min_subjects} subjects for analysis. Found {unique_subjects}.")


def build_epoch_query_string(column: str, level: Any, is_numeric: bool, labels: Optional[Dict] = None) -> Tuple[str, str]:
    if is_numeric:
        label = labels.get(level, str(level)) if labels else str(level)
        return f"{column} == {level}", label
    
    escaped_level = str(level).replace('"', '\\"')
    query = f'{column} == "{escaped_level}"'
    return query, str(level)


def prepare_topomap_correlation_data(band_data: Dict, info: mne.Info) -> Tuple[np.ndarray, np.ndarray]:
    n_info_chs = len(info['ch_names'])
    topo_data = np.zeros(n_info_chs)
    topo_mask = np.zeros(n_info_chs, dtype=bool)
    
    for j, info_ch in enumerate(info['ch_names']):
        if info_ch in band_data['channels']:
            ch_idx = band_data['channels'].index(info_ch)
            if np.isfinite(band_data['correlations'][ch_idx]):
                topo_data[j] = band_data['correlations'][ch_idx]
            topo_mask[j] = band_data['significant_mask'][ch_idx]
    
    return topo_data, topo_mask


__all__ = [
    "load_decoding_data",
    "load_multiple_subjects_decoding_data",
    "load_epochs_with_targets",
    "load_epochs_for_analysis",
    "DecodingDataResult",
    "get_available_subjects",
    "parse_subject_args",
    "_collect_subject_ids_with_features",
    "_validate_event_columns",
    "load_epochs_with_aligned_events",
    "pick_event_columns",
    "resolve_columns",
    "align_events_to_epochs",
    "align_events_to_epochs_strict",
    "align_events_with_policy",
    "trim_behavioral_to_events_strict",
    "validate_alignment",
    "align_or_raise",
    "get_aligned_events",
    "apply_baseline",
    "crop_epochs",
    "process_temperature_levels",
    "select_epochs_by_value",
    "build_covariate_matrix",
    "build_covariates_without_temp",
    # Feature loading utilities
    "load_subject_features",
    "load_subject_data_for_summary",
    # Data transformation utilities
    "flatten_lower_triangles",
    "align_feature_blocks",
    # Data filtering and validation
    "filter_finite_targets",
    "validate_trial_alignment",
    "validate_trial_alignment_manifest",
    # Epoch data extraction
    "extract_epoch_data_block",
    "prepare_trial_records_from_epochs",
    "extract_epoch_data",
    # Metadata processing
    "load_kept_indices",
    "process_subject_metadata",
    # Column extraction
    "extract_roi_columns",
    "extract_rating_array_for_tf",
    "extract_measure_prefixes",
    "extract_node_names_from_prefix",
    # Data structure building
    "build_per_subject_indices",
    "build_summary_map_for_prefix",
    # File loading
    "load_channel_correlations",
    "load_connectivity_files",
    # Data extraction utilities
    "extract_channel_importance_from_coefficients",
    "extract_band_channel_vectors",
    "validate_aligned_events_length",
    "prepare_partial_correlation_data",
    "extract_common_dataframe_columns",
    "prepare_group_partial_residuals_data",
    "prepare_group_band_roi_data",
    # Data extraction utilities (Priority 2)
    "extract_aligned_column_vector",
    "extract_pain_vector",
    "extract_pain_vector_array",
    "extract_temperature_series",
    "compute_aligned_data_length",
    "create_temperature_masks",
    "extract_time_frequency_grid",
    "extract_importance_column",
    "build_epoch_query_string",
    "create_temperature_masks_from_range",
    "get_temperature_range",
    # Data validation utilities
    "validate_data_not_empty",
    "validate_data_lengths",
    "validate_trial_ids",
    "validate_sufficient_subjects",
    "validate_feature_block_lengths",
    "register_feature_block",
]
