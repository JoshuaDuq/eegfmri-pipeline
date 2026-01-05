from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ..config.loader import ConfigDict
from eeg_pipeline.infra.paths import find_clean_epochs_path, resolve_deriv_root


EEGConfig = ConfigDict


###################################################################
# Subject Discovery Helpers
###################################################################

def _collect_subjects_from_bids(bids_root: Path) -> List[str]:
    """Collect all subjects from a BIDS directory."""
    if not bids_root.exists():
        return []
    subjects = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if sub_dir.is_dir():
            subjects.append(sub_dir.name[4:])
    return subjects


def _collect_subjects_from_source_data(source_root: Path) -> List[str]:
    """Collect all subjects from a source data directory."""
    if not source_root.exists():
        return []
    subjects = []
    # Support both 'sub-0001' and '0001' naming in source data
    for sub_dir in sorted(source_root.glob("*")):
        if sub_dir.is_dir():
            name = sub_dir.name
            if name.startswith("sub-"):
                subjects.append(name[4:])
            else:
                subjects.append(name)
    return subjects


def _collect_subjects_from_derivatives_epochs(
    deriv_root: Path, 
    task: str, 
    config: Optional[ConfigDict] = None, 
    constants: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Collect subjects that have clean epochs available."""
    if not deriv_root.exists():
        return []
    
    subjects = set()
    
    # Search in deriv_root/sub-* and deriv_root/preprocessed/sub-*
    search_patterns = [
        deriv_root.glob("sub-*"),
        (deriv_root / "preprocessed").glob("sub-*") if (deriv_root / "preprocessed").exists() else [],
    ]
    
    for pattern in search_patterns:
        for sub_dir in sorted(pattern):
            if not sub_dir.is_dir():
                continue
            sub_id = sub_dir.name[4:]
            epo_path = find_clean_epochs_path(sub_id, task, deriv_root=deriv_root, config=config, constants=constants)
            if epo_path is not None and epo_path.exists():
                subjects.add(sub_id)
    
    return sorted(list(subjects))


def _collect_subjects_from_features(deriv_root: Path) -> List[str]:
    """Collect subjects that have extracted features."""
    if not deriv_root.exists():
        return []
    
    subjects = set()
    
    # Search in deriv_root/sub-*/eeg/features and deriv_root/preprocessed/sub-*/eeg/features
    search_patterns = [
        deriv_root.glob("sub-*/eeg/features"),
        (deriv_root / "preprocessed").glob("sub-*/eeg/features") if (deriv_root / "preprocessed").exists() else [],
    ]
    
    for pattern in search_patterns:
        for sub_dir in sorted(pattern):
            # Relaxed check: any feature file (tsv or parquet)
            has_features = any(
                f.suffix in {".tsv", ".parquet"} and f.name.startswith("features_")
                for f in sub_dir.iterdir()
            )
            if has_features:
                # Path parts are e.g. [..., 'sub-0001', 'eeg', 'features']
                # so -3 is the subject ID
                try:
                    sub_id = sub_dir.parts[-3].replace("sub-", "")
                    subjects.add(sub_id)
                except (IndexError, AttributeError):
                    continue
    
    return sorted(list(subjects))


###################################################################
# Primary Subject Discovery Functions
###################################################################

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

    subjects_from_config = getattr(config, "subjects", None) or []

    discovered_subjects = []

    if "bids" in discovery_sources:
        bids_subjects = _collect_subjects_from_bids(bids_root)
        discovered_subjects.append(("bids", bids_subjects))
        logger.debug(f"Discovered {len(bids_subjects)} subjects from BIDS")

    if "derivatives_epochs" in discovery_sources:
        epoch_subjects = _collect_subjects_from_derivatives_epochs(
            deriv_root, task, config=config, constants=constants
        )
        discovered_subjects.append(("derivatives_epochs", epoch_subjects))
        logger.debug(f"Discovered {len(epoch_subjects)} subjects from derivatives (clean epochs)")

    if "features" in discovery_sources:
        feature_subjects = _collect_subjects_from_features(deriv_root)
        discovered_subjects.append(("features", feature_subjects))
        logger.debug(f"Discovered {len(feature_subjects)} subjects from derivatives (features)")

    if "source_data" in discovery_sources:
        src_root = None
        if hasattr(config, "source_root"):
            src_root = config.source_root
        else:
            p_val = config.get("paths.source_data")
            if p_val:
                src_root = Path(p_val)
            elif bids_root:
                src_root = bids_root.parent / "source_data"
        
        if src_root:
            source_subjects = _collect_subjects_from_source_data(src_root)
            discovered_subjects.append(("source_data", source_subjects))
            logger.debug(f"Discovered {len(source_subjects)} subjects from source data")

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
        raise ValueError(
            f"Unknown policy: {subject_discovery_policy}. Must be 'intersection', 'union', or 'config_only'"
        )

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
        deriv_root = resolve_deriv_root(config=config)

    if task is None:
        task = config.get("project.task")

    subjects: Optional[List[str]] = None

    # Determine discovery sources based on command and arguments
    sources = ["derivatives_epochs"]
    if hasattr(args, "source") and args.source:
        if args.source == "all":
            sources = ["bids", "derivatives_epochs", "features", "source_data"]
        elif args.source == "epochs":
            sources = ["derivatives_epochs"]
        else:
            sources = [args.source]
    elif hasattr(args, "mode"):
        if args.mode == "raw-to-bids":
            sources = ["source_data"]
        elif args.mode in {"combine", "merge-behavior", "visualize"}:
            sources = ["features", "bids"]
    
    if hasattr(args, "group") and args.group is not None:
        g = args.group.strip()
        if g.lower() in {"all", "*", "@all"}:
            subjects = get_available_subjects(
                config=config,
                deriv_root=deriv_root,
                task=task,
                discovery_sources=sources,
                logger=logger,
            )
        else:
            subjects = [
                s.strip() for s in g.replace(";", ",").replace(" ", ",").split(",") if s.strip()
            ]
    elif hasattr(args, "all_subjects") and args.all_subjects:
        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            task=task,
            discovery_sources=sources,
            logger=logger,
        )
    elif hasattr(args, "subject") and args.subject:
        subjects = list(dict.fromkeys(args.subject))
    elif hasattr(args, "subjects") and args.subjects:
        subjects = list(dict.fromkeys(args.subjects))

    if not subjects:
        # Fallback to config if available
        subjects = getattr(config, "subjects", None) or []
        
        # If still no subjects, perform discovery
        if not subjects:
            subjects = get_available_subjects(
                config=config,
                deriv_root=deriv_root,
                task=task,
                discovery_sources=sources,
                logger=logger,
            )

    return subjects


__all__ = [
    "get_available_subjects",
    "parse_subject_args",
    "get_epoch_metadata",
    "_collect_subjects_from_bids",
    "_collect_subjects_from_derivatives_epochs", 
    "_collect_subjects_from_features",
]


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
        # Read header only
        epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
        return {
            "tmin": float(epochs.tmin),
            "tmax": float(epochs.tmax),
        }
    except Exception:
        return {}
