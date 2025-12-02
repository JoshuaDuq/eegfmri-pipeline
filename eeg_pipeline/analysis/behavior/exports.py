"""Export utilities for behavioral correlation results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings, get_frequency_band_names
from eeg_pipeline.utils.io.general import (
    deriv_stats_path, ensure_dir, get_subject_logger,
    validate_predictor_file, build_predictor_column_mapping,
    build_predictor_name, read_tsv, write_tsv,
)


ANALYSIS_PATTERNS = {
    "power_roi": "corr_stats_pow_roi_vs_{target}.tsv",
    "power_combined": "corr_stats_pow_combined_vs_{target}.tsv",
    "connectivity": "corr_stats_conn_*_vs_{target}_*.tsv",
    "precomputed": "corr_stats_precomputed_vs_{target}_*.tsv",
    "microstates": "corr_stats_microstates_vs_{target}_*.tsv",
    "temporal": "corr_stats_temporal_*.tsv",
    "aperiodic": "corr_stats_aper_*_vs_{target}.tsv",
}


def _process_file(path: Path, target: str, ptype: str, use_fdr: bool,
                  alpha: float, logger) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = read_tsv(path)
    if df is None or df.empty or not validate_predictor_file(df, ptype, target, logger):
        return None
    # Determine which p-value column to use for filtering
    p_col = "q" if use_fdr and "q" in df.columns else "p"
    if p_col not in df.columns:
        logger.warning(f"No p-value column in {path.name}, skipping")
        return None
    sig = df[df[p_col] <= alpha].copy()
    if len(sig) == 0:
        return None
    sig = sig.copy()
    sig["predictor_type"] = ptype
    sig["target"] = target
    sig["source_file"] = path.name
    sig["predictor"] = build_predictor_name(sig, ptype)
    cols = build_predictor_column_mapping(ptype)
    for c in ["fdr_reject", "fdr_crit_p", "q_fdr_global", "fdr_reject_global"]:
        if c in sig.columns:
            cols[c] = c
    avail = [c for c in cols if c in sig.columns]
    return sig[avail].rename(columns={k: v for k, v in cols.items() if k in avail})


def export_all_significant_predictors(subject: str, alpha: float = None,
                                       use_fdr: bool = True) -> Optional[pd.DataFrame]:
    """Export all significant predictors across analyses."""
    config = load_settings()
    alpha = alpha or config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    ensure_dir(stats_dir)
    logger = get_subject_logger("behavior_analysis", subject,
                                config.get("logging.log_file_name", "behavior_analysis.log"), config=config)

    predictors = []
    configs = [
        ("corr_stats_pow_roi_vs_{target}.tsv", "ROI_power"),
        ("corr_stats_pow_combined_vs_{target}.tsv", "Channel_power"),
        ("corr_stats_conn_graph_vs_{target}_*.tsv", "Connectivity"),
        ("corr_stats_precomputed_vs_{target}_*.tsv", "Precomputed"),
        ("corr_stats_microstates_vs_{target}_*.tsv", "Microstate"),
        ("corr_stats_aper_*_vs_{target}.tsv", "Aperiodic"),
    ]

    for target in ("rating", "temp", "temperature"):
        for pattern, ptype in configs:
            for f in stats_dir.glob(pattern.format(target=target)):
                df = _process_file(f, target, ptype, use_fdr, alpha, logger)
                if df is not None:
                    predictors.append(df)

    out = stats_dir / "all_significant_predictors.tsv"
    if predictors:
        combined = pd.concat(predictors, ignore_index=True)
        if "r" in combined.columns:
            combined["abs_r"] = combined["r"].abs()
            combined["effect"] = pd.cut(combined["abs_r"], [0, 0.1, 0.3, 0.5, 1],
                                        labels=["negligible", "small", "medium", "large"])
        if "p" in combined.columns:
            combined = combined.sort_values("p", ascending=True)
        write_tsv(combined, out)
        logger.info(f"Exported {len(combined)} predictors to {out}")
        return combined
    write_tsv(pd.DataFrame(), out)
    return pd.DataFrame()


def export_combined_power_corr_stats(subject: str) -> Dict[str, Path]:
    """Combine band-specific power stats into single files."""
    config = load_settings()
    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    ensure_dir(stats_dir)
    bands = get_frequency_band_names(config)
    out = {}

    for target in ("rating", "temp"):
        frames = []
        for band in bands:
            f = stats_dir / f"corr_stats_pow_{band}_vs_{target}.tsv"
            if not f.exists():
                continue
            df = read_tsv(f)
            if df is not None and not df.empty:
                df["band"] = df.get("band", pd.Series([band] * len(df))).fillna(band)
                frames.append(df)
        if frames:
            path = stats_dir / f"corr_stats_pow_combined_vs_{target}.tsv"
            write_tsv(pd.concat(frames, ignore_index=True), path)
            out[target] = path
    return out


def export_analysis_summary(subject: str) -> Dict[str, Any]:
    """Generate summary stats across all analyses."""
    config = load_settings()
    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    if not stats_dir.exists():
        return {}

    summary = {
        "subject": subject,
        "timestamp": datetime.now().isoformat(),
        "analyses": {},
        "totals": {"n_tests": 0, "n_sig_local": 0, "n_sig_global": 0}
    }

    for target in ("rating", "temp"):
        for atype, pattern in ANALYSIS_PATTERNS.items():
            for f in stats_dir.glob(pattern.format(target=target)):
                df = read_tsv(f)
                if df is None or df.empty:
                    continue
                n = len(df)
                n_loc = int((df.get("q", pd.Series()) < 0.05).sum())
                n_glob = int(df.get("fdr_reject_global", pd.Series()).sum())
                key = f"{atype}_{target}"
                if key not in summary["analyses"]:
                    summary["analyses"][key] = {"n_tests": 0, "n_sig_local": 0, "n_sig_global": 0}
                summary["analyses"][key]["n_tests"] += n
                summary["analyses"][key]["n_sig_local"] += n_loc
                summary["analyses"][key]["n_sig_global"] += n_glob
                summary["totals"]["n_tests"] += n
                summary["totals"]["n_sig_local"] += n_loc
                summary["totals"]["n_sig_global"] += n_glob

    with open(stats_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def export_top_predictors(subject: str, n_top: int = None) -> pd.DataFrame:
    """Export top N predictors by effect size."""
    config = load_settings()
    n_top = n_top or int(config.get("behavior_analysis.predictors.top_n", 20))
    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    path = stats_dir / "all_significant_predictors.tsv"

    if not path.exists():
        df = export_all_significant_predictors(subject)
    else:
        df = read_tsv(path)

    if df is None or df.empty:
        return pd.DataFrame()

    if "abs_r" not in df.columns and "r" in df.columns:
        df["abs_r"] = df["r"].abs()
    result = df.sort_values("abs_r", ascending=False).head(n_top)
    write_tsv(result, stats_dir / f"top_{n_top}_predictors.tsv")
    return result
