from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ..config.loader import ConfigDict
from eeg_pipeline.infra.paths import find_clean_epochs_path, resolve_deriv_root
from .discovery import (
    _collect_subjects_from_bids,
    _collect_subjects_from_derivatives_epochs,
    _collect_subjects_from_features,
)


EEGConfig = ConfigDict


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
        deriv_root = resolve_deriv_root(config=config)

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
        epoch_subjects = _collect_subjects_from_derivatives_epochs(
            deriv_root, task, config=config, constants=constants
        )
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

    if hasattr(args, "group") and args.group is not None:
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
                s.strip() for s in g.replace(";", ",").replace(" ", ",").split(",") if s.strip()
            ]
            subjects = []
            for s in candidates:
                if find_clean_epochs_path(s, task, deriv_root=deriv_root, config=config) is not None:
                    subjects.append(s)
                else:
                    logger.warning(f"--group subject '{s}' has no cleaned epochs; skipping")
    elif hasattr(args, "all_subjects") and args.all_subjects:
        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            task=task,
            discovery_sources=["derivatives_epochs"],
            logger=logger,
        )
    elif hasattr(args, "subject") and args.subject:
        subjects = list(dict.fromkeys(args.subject))
    elif hasattr(args, "subjects") and args.subjects:
        subjects = list(dict.fromkeys(args.subjects))

    if subjects is None:
        subjects = config.subjects if hasattr(config, "subjects") and config.subjects else []

    return subjects


__all__ = ["get_available_subjects", "parse_subject_args"]
