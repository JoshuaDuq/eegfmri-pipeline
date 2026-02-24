from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from eeg_pipeline.infra.tsv import read_table


def load_stats_file_with_fallbacks(
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
    """Build file patterns with and without method suffix for backwards compatibility."""
    method_suffix = f"_{method_label}" if method_label else ""

    def _both_ext(stem: str) -> List[str]:
        return [f"{stem}.parquet", f"{stem}.tsv"]

    patterns = [
        *_both_ext(f"corr_stats_pow_roi_vs_{base_name}{method_suffix}"),
        *_both_ext(f"corr_stats_power_roi_vs_{base_name}{method_suffix}"),
    ]
    
    if method_label:
        patterns.extend([
            *_both_ext(f"corr_stats_pow_roi_vs_{base_name}"),
            *_both_ext(f"corr_stats_power_roi_vs_{base_name}"),
        ])
    
    return patterns


def load_behavior_stats_files(
    stats_dir: Path,
    logger: logging.Logger,
    method_label: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load rating and predictor correlation stats files with fallback patterns."""
    rating_patterns = _build_stats_file_patterns("rating", method_label)
    temp_patterns = _build_stats_file_patterns("temp", method_label)
    
    rating_stats = load_stats_file_with_fallbacks(stats_dir, rating_patterns)
    temp_stats = load_stats_file_with_fallbacks(stats_dir, temp_patterns)
    
    return rating_stats, temp_stats


__all__ = [
    "load_stats_file_with_fallbacks",
    "load_behavior_stats_files",
]
