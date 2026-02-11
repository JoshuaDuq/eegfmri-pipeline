from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..config.loader import ConfigDict
from eeg_pipeline.infra.paths import find_clean_epochs_path, resolve_deriv_root


EEGConfig = ConfigDict


def _normalize_subject_id(subject: Any) -> Optional[str]:
    """Normalize a subject label to an ID without the ``sub-`` prefix."""
    if subject is None:
        return None

    subject_id = str(subject).strip()
    if not subject_id or subject_id.lower() in {"none", "null"}:
        return None

    if subject_id.startswith("sub-"):
        subject_id = subject_id[4:]

    return subject_id or None


def _subject_match_key(subject_id: str) -> str:
    """Build a stable comparison key for subject matching across sources."""
    if subject_id.isdigit():
        try:
            return str(int(subject_id))
        except ValueError:
            return subject_id
    return subject_id


def _normalize_subject_list(subjects: Any) -> List[str]:
    """Normalize config/user subject values into a clean, deduplicated list."""
    if subjects is None:
        return []

    if isinstance(subjects, (str, int, float)):
        subjects = [subjects]

    normalized: List[str] = []
    seen_keys: set[str] = set()

    for raw in subjects:
        subject_id = _normalize_subject_id(raw)
        if subject_id is None:
            continue
        key = _subject_match_key(subject_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized.append(subject_id)

    return normalized


def _subject_key_to_preferred_id(subjects: List[str]) -> Dict[str, str]:
    """Map normalized subject comparison keys to a preferred display ID."""
    mapping: Dict[str, str] = {}
    for raw in subjects:
        subject_id = _normalize_subject_id(raw)
        if subject_id is None:
            continue
        key = _subject_match_key(subject_id)
        if key not in mapping:
            mapping[key] = subject_id
    return mapping


def _collect_subjects_from_bids(bids_root: Path) -> List[str]:
    """Collect all subjects from a BIDS directory."""
    if not bids_root.exists():
        return []
    
    subjects = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if sub_dir.is_dir():
            subject_id = _normalize_subject_id(sub_dir.name)
            if subject_id is not None:
                subjects.append(subject_id)
    return sorted(_subject_key_to_preferred_id(subjects).values())


def _collect_subjects_from_source_data(source_root: Path) -> List[str]:
    """Collect all subjects from a source data directory."""
    if not source_root.exists():
        return []
    
    subjects = []
    for sub_dir in sorted(source_root.glob("*")):
        if not sub_dir.is_dir():
            continue
        
        name = sub_dir.name
        subject_id = _normalize_subject_id(name)
        if subject_id is not None:
            subjects.append(subject_id)
    return sorted(_subject_key_to_preferred_id(subjects).values())


def _collect_subjects_from_derivatives_epochs(
    deriv_root: Path,
    task: str,
    config: Optional[ConfigDict] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Collect subjects that have clean epochs available."""
    if not deriv_root.exists():
        return []
    
    subjects = set()
    search_locations = [
        deriv_root.glob("sub-*"),
    ]
    
    preprocessed_eeg_dir = deriv_root / "preprocessed" / "eeg"
    if preprocessed_eeg_dir.exists():
        search_locations.append(preprocessed_eeg_dir.glob("sub-*"))
    
    for location in search_locations:
        for sub_dir in sorted(location):
            if not sub_dir.is_dir():
                continue
            
            sub_id = sub_dir.name[4:]
            epo_path = find_clean_epochs_path(
                sub_id, task, deriv_root=deriv_root, config=config, constants=constants
            )
            if epo_path is not None and epo_path.exists():
                subjects.add(sub_id)
    
    return sorted(subjects)


def _collect_subjects_from_features(deriv_root: Path) -> List[str]:
    """Collect subjects that have extracted features."""
    if not deriv_root.exists():
        return []
    
    subjects = set()
    search_locations = [
        deriv_root.glob("sub-*/eeg/features"),
    ]
    
    preprocessed_eeg_dir = deriv_root / "preprocessed" / "eeg"
    if preprocessed_eeg_dir.exists():
        search_locations.append(preprocessed_eeg_dir.glob("sub-*/features"))
    
    for location in search_locations:
        for features_dir in sorted(location):
            if not _has_feature_files(features_dir):
                continue
            
            sub_id = _extract_subject_id_from_features_path(features_dir)
            if sub_id:
                subjects.add(sub_id)
    
    return sorted(subjects)


def _has_feature_files(directory: Path) -> bool:
    """Check if directory contains feature files (including in subdirectories)."""
    if not directory.is_dir():
        return False
    return any(
        f.suffix in {".tsv", ".parquet"}
        for f in directory.rglob("features_*")
    )


def _extract_subject_id_from_features_path(features_path: Path) -> Optional[str]:
    """Extract subject ID from features directory path.

    Supports:
    - deriv_root/sub-XXX/eeg/features  (parts[-3] = sub-XXX)
    - deriv_root/preprocessed/eeg/sub-XXX/features  (parts[-2] = sub-XXX)
    """
    try:
        for part in reversed(features_path.parts):
            if part.startswith("sub-"):
                return part.replace("sub-", "", 1)
    except (IndexError, AttributeError):
        pass
    return None


def _resolve_source_root(config: EEGConfig, bids_root: Optional[Path]) -> Optional[Path]:
    """Resolve source data root from config or infer from bids_root."""
    source_data_path = config.get("paths.source_data")
    if source_data_path:
        return Path(source_data_path)
    
    if bids_root:
        return bids_root.parent / "source_data"
    
    return None


def _discover_subjects_from_sources(
    discovery_sources: List[str],
    deriv_root: Path,
    bids_root: Path,
    task: str,
    config: EEGConfig,
    constants: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> List[Tuple[str, List[str]]]:
    """Discover subjects from specified sources."""
    discovered_by_source = []
    
    if "bids" in discovery_sources:
        bids_subjects = _collect_subjects_from_bids(bids_root)
        discovered_by_source.append(("bids", bids_subjects))
        logger.debug(f"Discovered {len(bids_subjects)} subjects from BIDS")
    
    if "derivatives_epochs" in discovery_sources:
        epoch_subjects = _collect_subjects_from_derivatives_epochs(
            deriv_root, task, config=config, constants=constants
        )
        discovered_by_source.append(("derivatives_epochs", epoch_subjects))
        logger.debug(f"Discovered {len(epoch_subjects)} subjects from derivatives (clean epochs)")
    
    if "features" in discovery_sources:
        feature_subjects = _collect_subjects_from_features(deriv_root)
        discovered_by_source.append(("features", feature_subjects))
        logger.debug(f"Discovered {len(feature_subjects)} subjects from derivatives (features)")
    
    if "source_data" in discovery_sources:
        source_root = _resolve_source_root(config, bids_root)
        if source_root:
            source_subjects = _collect_subjects_from_source_data(source_root)
            discovered_by_source.append(("source_data", source_subjects))
            logger.debug(f"Discovered {len(source_subjects)} subjects from source data")
    
    return discovered_by_source


def _apply_config_only_policy(
    subjects_from_config: List[str],
    logger: logging.Logger,
) -> List[str]:
    """Apply config_only policy: use only subjects from config."""
    logger.info(f"Using config subjects only: {len(subjects_from_config)} subjects")
    return subjects_from_config


def _apply_intersection_policy(
    discovered_by_source: List[Tuple[str, List[str]]],
    subjects_from_config: List[str],
    logger: logging.Logger,
) -> List[str]:
    """Apply intersection policy: subjects present in all sources and config."""
    all_discovered = [subject for _, subjects in discovered_by_source for subject in subjects]
    discovered_by_key = _subject_key_to_preferred_id(all_discovered)
    
    if subjects_from_config:
        config_keys = {_subject_match_key(s) for s in _normalize_subject_list(subjects_from_config)}
        final_subjects = sorted(
            discovered_by_key[key]
            for key in discovered_by_key
            if key in config_keys
        )
        logger.info(
            f"Using intersection: {len(final_subjects)} subjects "
            f"(discovered={len(discovered_by_key)}, config={len(config_keys)})"
        )
        return final_subjects
    
    if len(discovered_by_source) > 1:
        subject_key_sets = [
            {_subject_match_key(s) for s in _normalize_subject_list(subjects)}
            for _, subjects in discovered_by_source
        ]
        final_keys = set.intersection(*subject_key_sets) if subject_key_sets else set()
        final_subjects = sorted(
            discovered_by_key[key]
            for key in final_keys
            if key in discovered_by_key
        )
        logger.info(
            f"Using intersection of discovery sources: {len(final_subjects)} subjects "
            f"(from {len(discovered_by_source)} sources)"
        )
        return final_subjects
    
    final_subjects = sorted(discovered_by_key.values())
    source_name = discovered_by_source[0][0] if discovered_by_source else "unknown"
    logger.info(f"Using discovered subjects: {len(final_subjects)} subjects (from {source_name})")
    return final_subjects


def _apply_union_policy(
    discovered_by_source: List[Tuple[str, List[str]]],
    subjects_from_config: List[str],
    logger: logging.Logger,
) -> List[str]:
    """Apply union policy: all subjects from any source or config."""
    all_discovered = [subject for _, subjects in discovered_by_source for subject in subjects]
    merged_by_key = _subject_key_to_preferred_id(all_discovered)

    for subject_id in _normalize_subject_list(subjects_from_config):
        key = _subject_match_key(subject_id)
        if key not in merged_by_key:
            merged_by_key[key] = subject_id

    final_subjects = sorted(merged_by_key.values())
    logger.info(
        f"Using union: {len(final_subjects)} subjects "
        f"(discovered={len(_subject_key_to_preferred_id(all_discovered))}, config={len(_normalize_subject_list(subjects_from_config))})"
    )
    return final_subjects


def _resolve_subjects_by_policy(
    discovered_by_source: List[Tuple[str, List[str]]],
    subjects_from_config: List[str],
    policy: Literal["intersection", "union", "config_only"],
    logger: logging.Logger,
) -> List[str]:
    """Resolve final subject list based on discovery policy."""
    if not discovered_by_source and policy != "config_only":
        logger.warning("No subjects discovered from any source")
        return []
    
    if policy == "config_only":
        return _apply_config_only_policy(subjects_from_config, logger)
    elif policy == "intersection":
        return _apply_intersection_policy(discovered_by_source, subjects_from_config, logger)
    elif policy == "union":
        return _apply_union_policy(discovered_by_source, subjects_from_config, logger)
    else:
        raise ValueError(
            f"Unknown policy: {policy}. Must be 'intersection', 'union', or 'config_only'"
        )


def get_available_subjects(
    config: EEGConfig,
    constants: Optional[Dict[str, Any]] = None,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    task: Optional[str] = None,
    discovery_sources: Optional[List[Literal["bids", "derivatives_epochs", "features", "source_data"]]] = None,
    subject_discovery_policy: Literal["intersection", "union", "config_only"] = "intersection",
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Discover and resolve available subjects based on config and discovery sources."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if deriv_root is None:
        deriv_root = resolve_deriv_root(config=config)
    
    if bids_root is None:
        bids_root = config.bids_root
    
    if task is None:
        task = config.get("project.task")
    
    if discovery_sources is None:
        discovery_sources = ["derivatives_epochs", "features"]
    
    subjects_from_config = _normalize_subject_list(config.get("project.subject_list"))
    
    discovered_by_source = _discover_subjects_from_sources(
        discovery_sources=discovery_sources,
        deriv_root=deriv_root,
        bids_root=bids_root,
        task=task,
        config=config,
        constants=constants,
        logger=logger,
    )
    
    return _resolve_subjects_by_policy(
        discovered_by_source=discovered_by_source,
        subjects_from_config=subjects_from_config,
        policy=subject_discovery_policy,
        logger=logger,
    )


def _determine_discovery_sources(args: Any) -> List[str]:
    """Determine discovery sources based on command arguments."""
    if hasattr(args, "source") and args.source:
        if args.source == "all":
            return ["bids", "derivatives_epochs", "features", "source_data"]
        elif args.source == "epochs":
            return ["derivatives_epochs"]
        else:
            return [args.source]
    
    # Preprocessing needs to discover from BIDS (epochs don't exist yet)
    if hasattr(args, "command") and args.command == "preprocessing":
        return ["bids"]
    
    if hasattr(args, "mode"):
        if args.mode in {"raw-to-bids", "fmri-raw-to-bids"}:
            return ["source_data"]
        elif args.mode in {"combine", "merge-behavior", "merge-psychopy", "visualize"}:
            return ["bids"]
    
    return ["derivatives_epochs"]


def _parse_group_argument(group_str: str) -> List[str]:
    """Parse comma or semicolon-separated group string into subject list."""
    normalized = group_str.replace(";", ",").replace(" ", ",")
    return [s.strip() for s in normalized.split(",") if s.strip()]


def _is_all_subjects_indicator(group_str: str) -> bool:
    """Check if group string indicates all subjects."""
    return group_str.lower() in {"all", "*", "@all"}


def _extract_subjects_from_args(args: Any) -> Optional[List[str]]:
    """Extract subjects from args attributes."""
    if hasattr(args, "group") and args.group is not None:
        group_str = args.group.strip()
        if _is_all_subjects_indicator(group_str):
            return None
        return _parse_group_argument(group_str)
    
    if hasattr(args, "all_subjects") and args.all_subjects:
        return None
    
    if hasattr(args, "subject") and args.subject:
        return list(dict.fromkeys(args.subject))
    
    if hasattr(args, "subjects") and args.subjects:
        return list(dict.fromkeys(args.subjects))
    
    return None


def parse_subject_args(
    args: Any,
    config: EEGConfig,
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Parse subject arguments from command-line args."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if deriv_root is None:
        deriv_root = resolve_deriv_root(config=config)
    
    if task is None:
        task = config.get("project.task")
    
    discovery_sources = _determine_discovery_sources(args)
    subjects = _extract_subjects_from_args(args)
    
    if subjects is None:
        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            task=task,
            discovery_sources=discovery_sources,
            logger=logger,
        )
    
    if not subjects:
        subjects = config.get("project.subject_list") or []
        if not subjects:
            subjects = get_available_subjects(
                config=config,
                deriv_root=deriv_root,
                task=task,
                discovery_sources=discovery_sources,
                logger=logger,
            )
    
    return subjects


def get_epoch_metadata(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Optional[ConfigDict] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Get metadata (tmin, tmax) for subject's epochs."""
    import mne
    
    epo_path = find_clean_epochs_path(
        subject, task, deriv_root=deriv_root, config=config, constants=constants
    )
    if epo_path is None or not epo_path.exists():
        return {}
    
    try:
        epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
        return {
            "tmin": float(epochs.tmin),
            "tmax": float(epochs.tmax),
        }
    except (OSError, ValueError, RuntimeError) as e:
        logging.getLogger(__name__).debug(
            f"Failed to read epoch metadata from {epo_path}: {e}"
        )
        return {}


__all__ = [
    "get_available_subjects",
    "parse_subject_args",
    "get_epoch_metadata",
    "_collect_subjects_from_bids",
    "_collect_subjects_from_derivatives_epochs",
    "_collect_subjects_from_features",
]
