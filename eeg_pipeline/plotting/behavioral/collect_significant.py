"""
Collect Significant Plots

Utility to assemble all significant behavioral correlation plots
into a dedicated directory for quick review.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eeg_pipeline.utils.io.general import (
    ensure_dir,
    get_subject_logger,
    get_default_config as _get_default_config,
)
from eeg_pipeline.plotting.config import get_plot_config


def collect_significant_plots(
    subject: str,
    deriv_root: Path,
    all_results: List[Dict[str, Any]],
    config=None,
    alpha: float = 0.05,
) -> Path:
    """
    Copy all scatter plots where correlation was significant (p < alpha)
    into a dedicated 'significant' subdirectory for quick review.
    
    Parameters
    ----------
    subject : str
        Subject ID
    deriv_root : Path
        Derivatives root directory
    all_results : List[Dict]
        Combined results from all scatter functions, each with 'significant' key
    config : Any
        Configuration object
    alpha : float
        Significance threshold
        
    Returns
    -------
    Path
        Path to the significant plots directory
    """
    config = config or _get_default_config()
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    
    from eeg_pipeline.utils.io.general import deriv_plots_path
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    sig_dir = plots_dir / "significant"
    ensure_dir(sig_dir)
    
    significant_records = []
    for result_set in all_results:
        if isinstance(result_set, dict) and "significant" in result_set:
            significant_records.extend(result_set["significant"])
    
    if not significant_records:
        logger.info("No significant correlations found to collect")
        return sig_dir
    
    copied_count = 0
    formats = plot_cfg.formats if hasattr(plot_cfg, 'formats') else ["png", "svg"]
    
    for record in significant_records:
        src_path = Path(record.get("path", ""))
        if not src_path.name:
            continue
        
        for fmt in formats:
            src_file = src_path.with_suffix(f".{fmt}")
            if src_file.exists():
                feature = record.get("feature", "unknown")
                roi = record.get("roi", "unknown")
                target = record.get("target", "unknown")
                r_val = record.get("r", 0)
                p_val = record.get("p", 1)
                
                dst_name = f"{feature}_{roi}_{target}_r{r_val:.2f}_p{p_val:.3f}.{fmt}"
                dst_name = dst_name.replace(" ", "_").replace("/", "_")
                
                dst_file = sig_dir / dst_name
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {src_file}: {e}")
    
    summary_records = []
    for record in significant_records:
        summary_records.append({
            "feature": record.get("feature", ""),
            "roi": record.get("roi", ""),
            "target": record.get("target", ""),
            "r": record.get("r", float("nan")),
            "p": record.get("p", float("nan")),
            "n": record.get("n", 0),
        })
    
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df = summary_df.sort_values("p", ascending=True)
        summary_path = sig_dir / f"sub-{subject}_significant_correlations.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        logger.info(f"Saved significant correlations summary: {summary_path}")
    
    logger.info(f"Collected {len(significant_records)} significant plots ({copied_count} files) to {sig_dir}")
    return sig_dir


def _get_all_correlation_records(
    stats_dir: Path,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Load all correlation statistics from TSV files and return as records.
    This is used as a fallback when scatter functions don't return results.
    """
    records = []
    
    tsv_patterns = [
        "corr_stats_*.tsv",
    ]
    
    for pattern in tsv_patterns:
        for tsv_path in stats_dir.glob(pattern):
            try:
                df = pd.read_csv(tsv_path, sep="\t")
                for _, row in df.iterrows():
                    records.append(row.to_dict())
            except Exception as e:
                logger.warning(f"Failed to read {tsv_path}: {e}")
    
    return records


__all__ = [
    "collect_significant_plots",
]
