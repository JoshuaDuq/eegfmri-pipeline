from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import mne

from .io_utils import (
    _find_clean_epochs_path,
    _load_events_df,
    _pick_target_column,
    deriv_features_path,
)
from .config_loader import load_settings, ConfigDict

EEGConfig = ConfigDict


###################################################################
# Event / Epoch Alignment
###################################################################

def align_events_to_epochs(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    strict_mode: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if events_df is None or len(events_df) == 0:
        logger.debug("No events DataFrame provided")
        return None

    if len(epochs) == 0:
        logger.debug("No epochs provided")
        return pd.DataFrame()

    logger.info(f"Attempting alignment: {len(events_df)} events to {len(epochs)} epochs (strict={strict_mode})")

    if hasattr(epochs, "selection") and epochs.selection is not None:
        sel = epochs.selection
        if len(events_df) > int(np.max(sel)) and len(sel) == len(epochs):
            aligned = events_df.iloc[sel].reset_index(drop=True)
            logger.info(f"Successfully aligned using epochs.selection ({len(sel)} epochs)")
            return aligned
        logger.warning(f"epochs.selection invalid: max={np.max(sel)}, events_len={len(events_df)}")

    if "sample" in events_df.columns and hasattr(epochs, "events") and epochs.events is not None:
        epoch_samples = epochs.events[:, 0]
        events_indexed = events_df.set_index("sample")
        aligned = events_indexed.reindex(epoch_samples)

        if len(aligned) == len(epochs) and not aligned.isna().all(axis=1).any():
            aligned_reset = aligned.reset_index()
            logger.info(f"Successfully aligned using sample column ({len(aligned)} epochs)")
            return aligned_reset
        logger.warning(f"Sample-based alignment failed: {aligned.isna().all(axis=1).sum()} NaN rows")

    logger.critical(
        f"CRITICAL: Unable to align events to epochs reliably. "
        f"Events: {len(events_df)} rows, Epochs: {len(epochs)} epochs. "
        f"Explicit alignment keys required: epochs.selection or 'sample' column in events."
    )

    raise ValueError(
        f"Cannot guarantee events-to-epochs alignment for reliable analysis. "
        f"Events DataFrame ({len(events_df)} rows) cannot be reliably aligned to "
        f"epochs ({len(epochs)} epochs). Explicit alignment is required:\n"
        f"1. Ensure epochs.selection is properly set during epoching, OR\n"
        f"2. Include 'sample' column in events with matching sample indices, OR\n"
        f"3. Use trial_alignment.manifest.tsv created by 02_feature_extraction.py and align via load_epochs_for_analysis."
    )


def trim_behavioral_to_events_strict(
    behavioral_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger(__name__)

    if len(behavioral_df) == len(events_df):
        return behavioral_df.reset_index(drop=True)

    if len(behavioral_df) > len(events_df):
        trimmed = behavioral_df.iloc[:len(events_df)].reset_index(drop=True)
        logger.warning(
            f"Behavioral data longer than events. "
            f"Trimming from {len(behavioral_df)} to {len(events_df)} rows."
        )
        return trimmed

    logger.critical(
        f"CRITICAL: Behavioral data shorter than events. "
        f"Behavioral: {len(behavioral_df)}, Events: {len(events_df)}. "
        f"This indicates missing behavioral trials."
    )

    raise ValueError(
        f"Behavioral data ({len(behavioral_df)} rows) is shorter than events "
        f"({len(events_df)} rows). This indicates missing behavioral trials "
        f"that cannot be safely recovered. Check data collection/preprocessing."
    )


def align_events_to_epochs_strict(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    return align_events_to_epochs(events_df, epochs, strict_mode=True, logger=logger)


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

    allow_trim = bool(config.get("alignment.allow_misaligned_trim", False))
    min_samples = int(config.get("alignment.min_alignment_samples", 5))
    strict_mode = bool(config.get("analysis.strict_mode", True))

    if events_df is None or len(events_df) == 0:
        logger.debug("No events DataFrame provided")
        return None

    if len(epochs) == 0:
        logger.debug("No epochs provided")
        return pd.DataFrame()

    n_events = len(events_df)
    n_epochs = len(epochs)

    aligned = align_events_to_epochs(events_df, epochs, strict_mode=False, logger=logger)

    if aligned is None:
        logger.error(f"Alignment failed: could not align {n_events} events to {n_epochs} epochs")
        if strict_mode:
            raise ValueError("Alignment failed and strict_mode=True")
        return None

    if len(aligned) != n_epochs:
        diff = abs(len(aligned) - n_epochs)
        if allow_trim and diff >= min_samples:
            n_keep = min(len(aligned), n_epochs)
            logger.warning(
                f"Alignment length mismatch (events={len(aligned)}, epochs={n_epochs}, diff={diff}). "
                f"Trimming to {n_keep} samples (allow_misaligned_trim=True, min_alignment_samples={min_samples})"
            )
            if len(aligned) > n_epochs:
                aligned = aligned.iloc[:n_epochs].reset_index(drop=True)
            else:
                logger.warning(f"Cannot trim: aligned has {len(aligned)} rows, epochs has {n_epochs}")
        elif strict_mode:
            raise ValueError(
                f"Alignment length mismatch (events={len(aligned)}, epochs={n_epochs}). "
                f"allow_misaligned_trim={allow_trim}, strict_mode=True"
            )
        else:
            logger.warning(
                f"Alignment length mismatch (events={len(aligned)}, epochs={n_epochs}, diff={diff}). "
                f"allow_misaligned_trim={allow_trim}, continuing with mismatch"
            )

    return aligned


def validate_alignment(
    aligned_events: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> bool:
    if logger is None:
        logger = logging.getLogger(__name__)

    if aligned_events is None:
        if strict:
            raise ValueError("aligned_events is None; cannot validate alignment")
        return False

    if len(aligned_events) != len(epochs):
        msg = f"Length mismatch: events={len(aligned_events)}, epochs={len(epochs)}"
        if strict:
            raise ValueError(msg)
        logger.error(msg)
        return False

    nan_fraction = aligned_events.isna().all(axis=1).mean()
    if nan_fraction > 0.1:
        msg = f"High NaN fraction in aligned events: {nan_fraction:.1%}"
        if strict:
            raise ValueError(msg)
        logger.error(msg)
        return False

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
    validate_alignment(aligned, epochs, logger=logger, strict=True)
    return aligned


def get_aligned_events(
    epochs: mne.Epochs,
    subject: str,
    task: str,
    *,
    strict: bool = True,
    allow_trim: bool = False,
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

    strict_mode = strict and not allow_trim

    try:
        aligned_events = align_events_to_epochs(events_df, epochs, strict_mode=strict_mode, logger=logger)
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

    validate_alignment(aligned_events, epochs, logger, strict=strict)

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
            f"This file is required and must be created by 02_feature_extraction.py. "
            f"Run 02_feature_extraction.py first to generate features with proper alignment."
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
    from .io_utils import _find_clean_epochs_path
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
            sub = sub_dir.parts[-3]
            subjects.append(sub.replace("sub-", ""))
    return subjects


def _collect_subject_ids_with_features(deriv_root: Path) -> List[str]:
    return _collect_subjects_from_features(deriv_root)


def _pick_first_column(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    if df is None:
        return None
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def _canonical_covariate_name(name: Optional[str], config=None) -> Optional[str]:
    if name is None:
        return None
    n = str(name).lower()
    if config is None:
        try:
            config = load_settings()
        except Exception:
            config = None
    temp_aliases = {"stimulus_temp", "stimulus_temperature", "temp", "temperature"}
    trial_aliases = {"trial", "trial_number", "trial_index", "run", "block"}
    if config is not None:
        temp_aliases.update(str(c).lower() for c in config.get("event_columns.temperature", []))
    if n in temp_aliases:
        return "temperature"
    if n in trial_aliases:
        return "trial"
    return n


def _build_covariate_matrices(
    df_events: Optional[pd.DataFrame],
    partial_covars: Optional[List[str]],
    temp_col: Optional[str],
    config=None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if df_events is None:
        return None, None

    if config is None:
        try:
            config = load_settings()
        except Exception:
            config = None

    covars: List[str] = []
    name_map: Dict[str, str] = {}
    temp_candidates: List[str] = config.get("event_columns.temperature", []) if config else []

    if partial_covars:
        for cov in partial_covars:
            if cov in df_events.columns:
                covars.append(cov)
                name_map[cov] = _canonical_covariate_name(cov, config=config) or cov
                continue
            canon = _canonical_covariate_name(cov, config=config)
            if canon == "temperature":
                tcol = _pick_first_column(df_events, temp_candidates)
                if tcol:
                    covars.append(tcol)
                    name_map[tcol] = canon
    else:
        tcol = _pick_first_column(df_events, temp_candidates)
        if tcol:
            covars.append(tcol)
            name_map[tcol] = "temperature"
        for cand in ["trial", "trial_number", "trial_index", "run", "block"]:
            if cand in df_events.columns:
                covars.append(cand)
                name_map[cand] = _canonical_covariate_name(cand, config=config) or cand
                break

    if not covars:
        return None, None

    Z = pd.DataFrame()
    for cov in covars:
        if cov in df_events.columns:
            Z[name_map.get(cov, cov)] = pd.to_numeric(df_events[cov], errors="coerce")

    if Z.empty:
        return None, None

    temp_canon = _canonical_covariate_name(temp_col, config=config) if temp_col else None
    Z_temp = Z.drop(columns=[temp_canon], errors="ignore") if temp_canon else Z.copy()
    if Z_temp.shape[1] == 0:
        Z_temp = None

    return Z, Z_temp


def _load_features_and_targets(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
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
    elif policy == "intersection":
        resolved = sorted(list(set(subjects_from_files) & set(subjects_from_config)))
        logger.info(
            f"Using intersection: {len(resolved)} subjects (file={len(subjects_from_files)}, config={len(subjects_from_config)})"
        )
    elif policy == "union":
        resolved = sorted(list(set(subjects_from_files) | set(subjects_from_config)))
        logger.info(
            f"Using union: {len(resolved)} subjects (file={len(subjects_from_files)}, config={len(subjects_from_config)})"
        )
    else:
        raise ValueError(f"Unknown policy: {policy}. Must be 'intersection', 'union', or 'config_only'")
    
    return resolved


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

    if align not in ("strict", "warn", "none"):
        raise ValueError(f"align must be one of 'strict', 'warn', 'none', got '{align}'")

    if config is None:
        raise ValueError("config is required for load_epochs_for_analysis")

    allow_trim = bool(config.get("alignment.allow_misaligned_trim", False))
    min_alignment_samples = int(config.get("alignment.min_alignment_samples", 5))
    strict_mode = align == "strict"

    epochs_path = _find_clean_epochs_path(subject, task, deriv_root=deriv_root, config=config, constants=constants)
    if epochs_path is None or not epochs_path.exists():
        logger.error(f"Could not find cleaned epochs file for sub-{subject}, task-{task}")
        return None, None

    logger.info(f"Loading epochs: {epochs_path}")
    epochs = mne.read_epochs(epochs_path, preload=preload, verbose=False)

    events_df = _load_events_df(subject, task, bids_root=bids_root, config=config, constants=constants)
    if events_df is None:
        if align == "strict":
            raise ValueError(f"Events TSV not found for sub-{subject}, task-{task}. Required when align='strict'")
        logger.warning("Events TSV not found; metadata will not be set.")
        return epochs, None

    logger.info(f"Loaded events: {len(events_df)} rows")
    logger.debug(f"Alignment parameters: allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}")

    _validate_event_columns(events_df, config, logger)

    try:
        aligned_events = align_events_to_epochs(events_df, epochs, strict_mode=strict_mode, logger=logger)
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

    n_events = len(aligned_events)
    n_epochs = len(epochs)

    if n_events != n_epochs:
        diff = abs(n_events - n_epochs)
        if allow_trim and diff >= min_alignment_samples:
            logger.warning(
                f"Alignment length mismatch for sub-{subject}, task-{task}: "
                f"events={n_events}, epochs={n_epochs}, diff={diff}. "
                f"Trimming enabled (min_alignment_samples={min_alignment_samples})."
            )
            if n_events > n_epochs:
                aligned_events = aligned_events.iloc[:n_epochs].reset_index(drop=True)
                n_events = len(aligned_events)
        elif strict_mode:
            raise ValueError(
                f"Alignment length mismatch for sub-{subject}, task-{task}: "
                f"events={n_events}, epochs={n_epochs}, diff={diff}. "
                f"allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}, "
                f"strict_mode=True. Cannot proceed without alignment."
            )
        else:
            logger.warning(
                f"Alignment length mismatch for sub-{subject}, task-{task}: "
                f"events={n_events}, epochs={n_epochs}, diff={diff}. "
                f"allow_misaligned_trim={allow_trim}, continuing with mismatch."
            )

    if n_events != n_epochs and strict_mode:
        raise ValueError(
            f"Failed to align events to epochs for sub-{subject}, task-{task}: "
            f"final length mismatch (events={n_events}, epochs={n_epochs}). "
            f"Cannot guarantee alignment when strict_mode=True."
        )

    validation_result = validate_alignment(aligned_events, epochs, logger, strict=(align == "strict"))
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
    rating_cols = config.get("event_columns.rating", [])
    temp_cols = config.get("event_columns.temperature", [])
    pain_cols = config.get("event_columns.pain_binary", [])

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
        from .config_loader import load_settings
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
        task = config.task
    
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
        task = config.task
    
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
    
    return subjects


###################################################################
# Main Data Loading Functions
###################################################################

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
        task = config.task
    
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
            f"This indicates a misalignment between feature extraction and trial manifest."
        )
    
    if len(y_df) != expected_n_trials:
        raise ValueError(
            f"Target count mismatch for subject {sub}, task {task}: "
            f"targets have {len(y_df)} rows but trial_alignment.tsv specifies {expected_n_trials} trials. "
            f"This indicates a misalignment between target extraction and trial manifest."
        )
    
    tgt_col = _pick_target_column(y_df, config=config)
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
        task = config.task
    
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
            "Run 02_feature_extraction.py first."
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
        task = config.task
    
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
                f"Re-run 02_feature_extraction.py to regenerate features with the current epochs."
            )
        
        _, aligned = load_epochs_for_analysis(s, task, align="strict", preload=False, deriv_root=deriv_root, config=config)
        if aligned is None or len(aligned) == 0:
            logger.warning(f"No aligned events/targets for {sub}; skipping.")
            continue
        
        _validate_event_columns(aligned, config, logger)
        
        tgt_col = _pick_target_column(aligned, config=config)
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
    
    return out, []


###################################################################
# Epoch Selection and Processing Utilities
###################################################################

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
# High-Level Epoch Loading with Manifest Alignment
###################################################################

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
    "process_temperature_levels",
    "select_epochs_by_value",
]
