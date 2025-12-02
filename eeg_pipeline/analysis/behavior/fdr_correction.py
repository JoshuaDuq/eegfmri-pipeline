"""Global FDR correction across all analysis types."""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.io.general import (
    deriv_stats_path, ensure_dir, get_subject_logger,
    parse_analysis_type_from_filename, parse_target_from_filename,
    parse_measure_band_from_filename, build_file_updates_dict, build_meta_for_row,
    read_tsv, write_tsv, fdr_bh_reject,
)
from eeg_pipeline.utils.analysis.stats import (
    get_pvalue_series, extract_pvalue_from_dataframe, _safe_float, bh_adjust as _bh_adjust,
)


def _apply_fdr_updates_to_files(file_updates: Dict[Path, List[Tuple[int, float, bool, float]]],
                                critical_p: float, logger) -> None:
    for path, items in file_updates.items():
        try:
            df = read_tsv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        n = len(df)
        q_col, rej_col, p_col = np.full(n, np.nan), np.zeros(n, dtype=bool), np.full(n, np.nan)
        for idx, q, rej, p in items:
            if 0 <= idx < n:
                q_col[idx], rej_col[idx], p_col[idx] = q, rej, p
        df["p_used_for_global_fdr"] = p_col
        df["q_fdr_global"] = q_col
        df["fdr_reject_global"] = rej_col
        df["fdr_crit_p_global"] = _safe_float(critical_p)
        try:
            write_tsv(df, path)
        except Exception as e:
            logger.warning(f"Failed to update {path.name}: {e}")


def apply_global_fdr(subject: str, alpha: float = None) -> None:
    """Apply global FDR correction across all correlation stats files."""
    if not subject:
        return
    
    config = load_settings()
    alpha = alpha or config.get("behavior_analysis.statistics.fdr_alpha") or config.get("statistics.fdr_alpha", 0.05)
    logger = get_subject_logger("behavior_analysis", subject, config.get("logging.log_file_name", "behavior_analysis.log"), config=config)
    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    ensure_dir(stats_dir)
    
    # Simplified: just glob all correlation stats files
    files = sorted(stats_dir.glob("corr_stats_*.tsv")) + sorted(stats_dir.glob("pain_nonpain_*.tsv"))
    if not files:
        logger.error(f"No stats TSVs found for global FDR in {stats_dir}")
        raise RuntimeError("No stats TSVs found for global FDR correction")

    p_values, file_refs, meta_records = [], [], []
    
    for path in files:
        try:
            df = read_tsv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        fn = path.name
        atype = parse_analysis_type_from_filename(fn)
        target = parse_target_from_filename(fn)
        measure_band = parse_measure_band_from_filename(atype, fn)

        p_perm, p_raw = get_pvalue_series(df)
        p_combined = p_perm.where(np.isfinite(p_perm), p_raw).to_numpy()
        
        for i, p in enumerate(p_combined):
            if not np.isfinite(p):
                continue
            _, p_src = extract_pvalue_from_dataframe(df, i)
            p_values.append(float(p))
            file_refs.append((path, i))
            meta_records.append(build_meta_for_row(df, i, fn, atype, target, measure_band, p_src))

    if not p_values:
        logger.error("No valid p-values found for global FDR")
        raise RuntimeError("No valid p-values for global FDR")

    p_arr = np.asarray(p_values)
    n = len(p_arr)
    logger.info(f"Global FDR on {n} tests from {len(files)} files")
    
    q_arr = _bh_adjust(p_arr)
    rej_arr, crit_p = fdr_bh_reject(p_arr, alpha=_safe_float(alpha))
    
    n_rej = int(rej_arr.sum())
    logger.info(f"FDR: {n_rej}/{n} significant at alpha={alpha} (crit p={crit_p:.6f})")

    file_updates = build_file_updates_dict(file_refs, q_arr, rej_arr, p_arr)
    _apply_fdr_updates_to_files(file_updates, crit_p, logger)

    summary = []
    for i, meta in enumerate(meta_records):
        row = dict(meta)
        row.update({"p_used_for_global_fdr": _safe_float(p_arr[i]), "q_fdr_global": _safe_float(q_arr[i]),
                    "fdr_reject_global": bool(rej_arr[i]), "fdr_crit_p_global": _safe_float(crit_p)})
        summary.append(row)
    
    try:
        write_tsv(pd.DataFrame(summary), stats_dir / "global_fdr_summary.tsv")
    except Exception as e:
        logger.warning(f"Failed to write FDR summary: {e}")
