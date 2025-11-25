import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    ensure_dir,
    get_subject_logger,
    parse_analysis_type_from_filename,
    parse_target_from_filename,
    parse_measure_band_from_filename,
    build_file_updates_dict,
    build_meta_for_row,
    read_tsv,
    write_tsv,
    fdr_bh_reject,
)
from eeg_pipeline.utils.analysis.stats import (
    get_pvalue_series,
    extract_pvalue_from_dataframe,
    _safe_float,
    bh_adjust as _bh_adjust,
)


def _apply_fdr_updates_to_files(
    file_updates: Dict[Path, List[Tuple[int, float, bool, float]]],
    critical_p: float,
    stats_dir: Path,
    logger: Any,
) -> None:
    for file_path, update_items in file_updates.items():
        try:
            dataframe = read_tsv(file_path)
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        
        if dataframe is None or dataframe.empty:
            continue
        
        n_rows = len(dataframe)
        q_column = np.full(n_rows, np.nan, dtype=float)
        rejection_column = np.zeros(n_rows, dtype=bool)
        p_used_column = np.full(n_rows, np.nan, dtype=float)
        
        for row_index, q_value, is_rejected, p_used in update_items:
            row_idx = int(row_index)
            if 0 <= row_idx < n_rows:
                q_column[row_idx] = q_value
                rejection_column[row_idx] = is_rejected
                p_used_column[row_idx] = p_used
        
        dataframe["p_used_for_global_fdr"] = p_used_column
        dataframe["q_fdr_global"] = q_column
        dataframe["fdr_reject_global"] = rejection_column
        dataframe["fdr_crit_p_global"] = _safe_float(critical_p)
        
        try:
            write_tsv(dataframe, file_path)
            logger.info(f"Updated {file_path.name} with global FDR correction")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to update {file_path.name} with global FDR correction: {e}")


def apply_global_fdr(subject: str, alpha: float = None) -> None:
    """
    Apply global FDR correction across ALL analysis types for valid statistical inference.
    
    This function is CRITICAL for maintaining statistical validity. It applies Benjamini-Hochberg
    FDR correction across all tests from different analysis types (power ROI correlations,
    connectivity ROI summaries, connectivity edges, etc.) to control the false discovery rate
    across all tests.
    
    Per-analysis-type FDR correction (applied in _apply_fdr_correction_and_save) is insufficient
    when interpreting results across multiple analysis types, as it inflates the false positive rate.
    
    The global FDR correction adds columns to each stats file:
    - p_used_for_global_fdr: The p-value used in global FDR correction
    - q_fdr_global: The FDR-adjusted q-value from global correction
    - fdr_reject_global: Boolean indicating rejection at global FDR level
    - fdr_crit_p_global: The critical p-value threshold for global FDR
    
    Parameters
    ----------
    subject : str
        Subject identifier (without 'sub-' prefix)
    alpha : float, optional
        FDR significance level (default: from config behavior_analysis.statistics.fdr_alpha, or 0.05)
    """
    if not subject:
        return
    
    config = load_settings()
    if alpha is None:
        alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    logger.info(
        f"Applying GLOBAL FDR correction (alpha={alpha}) across all analysis types. "
        f"This is CRITICAL for valid statistical inference when interpreting results across "
        f"multiple analysis types (power ROI, connectivity ROI, edges, etc.)."
    )

    patterns = [
        "corr_stats_pow_roi_vs_rating.tsv",
        "corr_stats_pow_roi_vs_temp.tsv", 
        "corr_stats_conn_roi_summary_*_vs_rating.tsv",
        "corr_stats_conn_roi_summary_*_vs_temp.tsv",
        "corr_stats_edges_*_vs_rating.tsv",
        "corr_stats_edges_*_vs_temp.tsv",
        "corr_stats_tf_clusters_*.tsv",
        "corr_stats_temporal_*.tsv",
        "pain_nonpain_time_clusters_*.tsv",
    ]
    
    files = [f for pat in patterns for f in sorted(stats_dir.glob(pat))]
    if not files:
        error_msg = (
            f"No stats TSVs found for global FDR in {stats_dir}. "
            f"Global FDR correction was requested but cannot be applied."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    p_values = []
    file_references = []
    metadata_records = []

    for file_path in files:
        try:
            dataframe = read_tsv(file_path)
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if dataframe is None or dataframe.empty:
            continue

        filename = file_path.name
        analysis_type = parse_analysis_type_from_filename(filename)
        target = parse_target_from_filename(filename)
        measure_band = parse_measure_band_from_filename(analysis_type, filename)

        p_permutation_series, p_raw_series = get_pvalue_series(dataframe)
        p_combined_series = p_permutation_series.where(np.isfinite(p_permutation_series), p_raw_series)
        p_combined_values = p_combined_series.to_numpy()
        if not np.any(np.isfinite(p_combined_values)):
            continue

        for row_index, p_value in enumerate(p_combined_values):
            if not np.isfinite(p_value):
                continue
            p_value = float(p_value)
            _, p_source = extract_pvalue_from_dataframe(dataframe, row_index)
            p_values.append(p_value)
            file_references.append((file_path, row_index))
            metadata = build_meta_for_row(
                dataframe, row_index, filename, analysis_type, target, measure_band, p_source
            )
            metadata_records.append(metadata)

    if not p_values:
        error_msg = (
            "No valid p-values found for global FDR correction. "
            "This indicates that the underlying stats files contain no finite p-values."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    p_array = np.asarray(p_values, dtype=float)
    n_tests = len(p_array)
    logger.info(f"Applying global FDR correction to {n_tests} tests across {len(files)} files")
    
    q_array = _bh_adjust(p_array)
    rejections_array, critical_p = fdr_bh_reject(p_array, alpha=_safe_float(alpha))
    
    n_rejected = int(rejections_array.sum())
    logger.info(
        f"Global FDR correction results: {n_rejected}/{n_tests} tests significant "
        f"at global FDR alpha={alpha} (critical p={critical_p:.6f})"
    )

    file_updates = build_file_updates_dict(file_references, q_array, rejections_array, p_array)
    _apply_fdr_updates_to_files(file_updates, critical_p, stats_dir, logger)

    summary_rows = []
    for index, metadata in enumerate(metadata_records):
        row = dict(metadata)
        row["p_used_for_global_fdr"] = _safe_float(p_array[index])
        row["q_fdr_global"] = _safe_float(q_array[index])
        row["fdr_reject_global"] = bool(rejections_array[index])
        row["fdr_crit_p_global"] = _safe_float(critical_p)
        summary_rows.append(row)

    try:
        summary_dataframe = pd.DataFrame(summary_rows)
        write_tsv(summary_dataframe, stats_dir / "global_fdr_summary.tsv")
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to write global FDR summary: {e}")

