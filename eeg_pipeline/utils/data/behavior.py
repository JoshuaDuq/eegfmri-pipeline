from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from eeg_pipeline.infra.tsv import read_table


def load_stats_file(
    stats_dir: Path,
    patterns: List[str],
) -> Optional[pd.DataFrame]:
    """Load first available stats file matching any of the given patterns."""
    for pattern in patterns:
        filepath = stats_dir / pattern
        if filepath.exists():
            df = read_table(filepath)
            if df is not None and not df.empty:
                return df
    return None


def _build_stats_file_patterns(
    base_name: str,
    method_label: Optional[str],
) -> List[str]:
    """Build canonical correlation-stats file patterns."""
    method_suffix = f"_{method_label}" if method_label else ""

    def _both_ext(stem: str) -> List[str]:
        return [f"{stem}.parquet", f"{stem}.tsv"]

    return [
        *_both_ext(f"corr_stats_pow_roi_vs_{base_name}{method_suffix}"),
        *_both_ext(f"corr_stats_power_roi_vs_{base_name}{method_suffix}"),
    ]


def load_behavior_stats_files(
    stats_dir: Path,
    logger: logging.Logger,
    method_label: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load outcome and predictor correlation stats files."""
    outcome_patterns = _build_stats_file_patterns("outcome", method_label)
    predictor_patterns = _build_stats_file_patterns("predictor", method_label)

    outcome_stats = load_stats_file(stats_dir, outcome_patterns)
    predictor_stats = load_stats_file(stats_dir, predictor_patterns)

    return outcome_stats, predictor_stats


__all__ = [
    "load_stats_file",
    "load_behavior_stats_files",
]
