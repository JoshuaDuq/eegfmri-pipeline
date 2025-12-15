from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass
import logging
import warnings
import numpy as np
import pandas as pd
import mne

from eeg_pipeline.io.paths import (
    _find_clean_epochs_path,
    _load_events_df,
    deriv_features_path,
    find_connectivity_features_path,
)
from eeg_pipeline.io.columns import pick_target_column
from eeg_pipeline.io.tsv import read_tsv, read_table
from ..config.loader import load_settings, ConfigDict, get_config_value, ensure_config, get_frequency_band_names

EEGConfig = ConfigDict

from .epochs_loading import (
    load_epochs_for_analysis as _load_epochs_for_analysis,
    load_epochs_with_aligned_events as _load_epochs_with_aligned_events,
    pick_event_columns as _pick_event_columns_v2,
    resolve_columns as _resolve_columns_v2,
    _validate_event_columns as _validate_event_columns_v2,
)
from .subjects import (
    get_available_subjects as _get_available_subjects_v2,
    parse_subject_args as _parse_subject_args_v2,
)
from .behavior import (
    load_behavior_plot_features as _load_behavior_plot_features_v2,
    load_stats_file_with_fallbacks as _load_stats_file_with_fallbacks_v2,
    load_behavior_stats_files as _load_behavior_stats_files_v2,
    load_subject_data_for_summary as _load_subject_data_for_summary_v2,
)


###################################################################
# Imports from submodule refactor
###################################################################

from .alignment import (
    align_events_to_epochs,
    align_events_to_epochs_strict,
    align_events_with_policy,
    get_aligned_events,
    validate_alignment,
)


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
    return deriv_features_path(deriv_root, subject) / "trial_alignment.tsv"


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




from .discovery import (
    get_available_subjects,
    _collect_subjects_from_bids,
    _collect_subjects_from_derivatives_epochs,
    _collect_subjects_from_features,
    _collect_subject_ids_with_features,
)
from .covariates import (
    extract_temperature_data,
    extract_default_covariates,
    _canonical_covariate_name,
    _resolve_covariate_columns,
    _build_covariate_matrices,
    _pick_first_column,
)
from .stats import (
    load_precomputed_correlations,
    get_precomputed_stats_for_roi_band,
    load_subject_scatter_data,
)



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
    _validate_event_columns_v2(events_df, config, logger)


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
            from eeg_pipeline.io.logging import log_and_raise_error
            error_msg = (
                f"Alignment length mismatch for sub-{subject}, task-{task}: "
                f"events={n_events}, epochs={n_epochs}, diff={diff}. "
                f"Cannot trim when events < epochs."
            )
            log_and_raise_error(logger, error_msg)
    else:
        from eeg_pipeline.io.logging import log_and_raise_error
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
    return _load_epochs_for_analysis(
        subject,
        task,
        align=align,
        preload=preload,
        deriv_root=deriv_root,
        bids_root=bids_root,
        logger=logger,
        config=config,
        constants=constants,
        use_cache=use_cache,
    )


def load_epochs_with_aligned_events(
    subject: str,
    task: str,
    config,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    preload: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    return _load_epochs_with_aligned_events(
        subject,
        task,
        config,
        deriv_root=deriv_root,
        bids_root=bids_root,
        preload=preload,
        logger=logger,
    )


def pick_event_columns(df: pd.DataFrame, config) -> Dict[str, Optional[str]]:
    return _pick_event_columns_v2(df, config)


def resolve_columns(
    df: pd.DataFrame,
    config: Optional[EEGConfig] = None,
    deriv_root: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return _resolve_columns_v2(df, config=config, deriv_root=deriv_root)




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
    return _get_available_subjects_v2(
        config=config,
        constants=constants,
        deriv_root=deriv_root,
        bids_root=bids_root,
        task=task,
        discovery_sources=discovery_sources,
        subject_discovery_policy=subject_discovery_policy,
        logger=logger,
    )


def parse_subject_args(
    args,
    config,
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    return _parse_subject_args_v2(
        args,
        config,
        task=task,
        deriv_root=deriv_root,
        logger=logger,
    )


# -----------------------------------------------------------------
# Re-exports (preferred import path)
# -----------------------------------------------------------------
from .decoding import (  # noqa: E402
    load_decoding_data as load_decoding_data,
    load_multiple_subjects_decoding_data as load_multiple_subjects_decoding_data,
    load_plateau_matrix as load_plateau_matrix,
    load_epoch_windows as load_epoch_windows,
)


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
        
        target_columns = list(config.get("event_columns.rating", []) or [])
        tgt_col = pick_target_column(aligned, target_columns=target_columns)
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


# -----------------------------------------------------------------
# Re-exports (preferred import path)
# -----------------------------------------------------------------
from .epochs import (  # noqa: E402
    apply_baseline as apply_baseline,
    crop_epochs as crop_epochs,
    process_temperature_levels as process_temperature_levels,
    select_epochs_by_value as select_epochs_by_value,
)

from .features_io import (  # noqa: E402
    load_subject_features as load_subject_features,
    load_feature_bundle_for_subject as load_feature_bundle_for_subject,
    FeatureBundle as FeatureBundle,
    load_feature_bundle as load_feature_bundle,
    load_feature_dfs_for_subjects as load_feature_dfs_for_subjects,
)


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




def load_behavior_plot_features(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
) -> tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame], Optional[List[str]]]:
    return _load_behavior_plot_features_v2(subject, task, config, logger)


def load_stats_file_with_fallbacks(
    stats_dir: Path,
    patterns: List[str],
) -> Optional[pd.DataFrame]:
    return _load_stats_file_with_fallbacks_v2(stats_dir, patterns)


def load_behavior_stats_files(stats_dir: Path, logger: logging.Logger) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    return _load_behavior_stats_files_v2(stats_dir, logger)


def load_subject_data_for_summary(subjects: List[str], task: str, deriv_root: Path, config, logger: Optional[logging.Logger] = None) -> Tuple[Dict, Dict, Dict, Dict, bool]:
    return _load_subject_data_for_summary_v2(
        subjects,
        task,
        deriv_root,
        config,
        logger=logger,
    )


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
    
    roi_columns = []
    for ch in channels:
        candidates = [
            f"power_plateau_{band}_ch_{ch}_logratio",
            f"power_plateau_{band}_ch_{ch}_log10raw",
            f"pow_{band}_{ch}",
        ]
        col = next((c for c in candidates if c in band_columns), None)
        if col is not None:
            roi_columns.append(col)
    return roi_columns if roi_columns else None

def extract_rating_array_for_tf(
    aligned_events: pd.DataFrame,
    config,
    logger,
) -> Optional[np.ndarray]:
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
    from eeg_pipeline.io.paths import deriv_stats_path
    
    if not subject or not band or correlation_type not in ("rating", "temp"):
        return None
    
    stats_dir = deriv_stats_path(deriv_root, subject)
    if correlation_type == "rating":
        candidates = [
            stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv",
            stats_dir / f"corr_stats_power_{band}_vs_rating.tsv",
        ]
    else:
        candidates = [
            stats_dir / f"corr_stats_pow_{band}_vs_temp.tsv",
            stats_dir / f"corr_stats_power_{band}_vs_temp.tsv",
        ]

    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        return None
    
    df = pd.read_csv(file_path, sep="\t")
    if df.empty or "channel" not in df.columns or "r" not in df.columns:
        return None
    
    return df

def load_connectivity_files(subjects: List[str], deriv_root: Path) -> Dict[str, List[pd.DataFrame]]:
    from eeg_pipeline.io.paths import deriv_stats_path
    
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
    from eeg_pipeline.io.decoding import parse_pow_feature
    
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
        cols = [
            c for c in df.columns
            if str(c).startswith(f"power_plateau_{band_str}_ch_")
        ]
        legacy = False
        if not cols:
            cols = [c for c in df.columns if str(c).startswith(f"pow_{band_str}_")]
            legacy = True
        if not cols:
            continue
        series = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        channel_means = {}
        for c, v in series.items():
            if not np.isfinite(v):
                continue
            name = str(c)
            if legacy:
                ch = name.replace(f"pow_{band_str}_", "")
            else:
                prefix = f"power_plateau_{band_str}_ch_"
                rest = name[len(prefix):]
                ch = rest.rsplit("_", 1)[0] if "_" in rest else rest
            channel_means[ch] = float(v)
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
        temperature_rounding_decimals = temperature_rounding_decimals or int(get_config_value(config, "plotting.tfr.temperature.rounding_decimals", 1))
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
        temperature_rounding_decimals = temperature_rounding_decimals or int(get_config_value(config, "plotting.tfr.temperature.rounding_decimals", 1))
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
        temperature_rounding_decimals = int(get_config_value(config, "plotting.tfr.temperature.rounding_decimals", 1))
    
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
    "load_stats_file_with_fallbacks",
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
    # Decoding data loaders
    "load_plateau_matrix",
    "load_epoch_windows",
]
