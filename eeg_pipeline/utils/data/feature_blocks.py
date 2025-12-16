from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import read_tsv


def validate_trial_alignment_manifest(
    aligned_events: pd.DataFrame,
    features_dir: Path,
    logger: logging.Logger,
) -> None:
    manifest_path = features_dir / "trial_alignment.tsv"
    if not manifest_path.exists():
        raise ValueError(f"Trial alignment manifest not found: {manifest_path}")

    manifest = read_tsv(manifest_path)
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
            "This indicates extraction failures and prevents valid analysis. "
            "Fix feature extraction before proceeding."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.warning(
        f"Some non-critical feature blocks are empty while others are not: empty={empty_blocks}, "
        f"non-empty={nonempty_blocks}. Analysis will proceed but may be incomplete."
    )


__all__ = [
    "validate_trial_alignment_manifest",
    "register_feature_block",
    "validate_feature_block_lengths",
]
