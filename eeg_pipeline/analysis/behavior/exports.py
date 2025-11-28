import logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    ensure_dir,
    get_subject_logger,
    validate_predictor_file,
    build_predictor_column_mapping,
    build_predictor_name,
    read_tsv,
    write_tsv,
)
from eeg_pipeline.utils.analysis.stats import (
    filter_significant_predictors,
)


def _add_fdr_columns_to_mapping(cols: Dict[str, str], df: pd.DataFrame) -> Dict[str, str]:
    if "fdr_reject" in df.columns:
        cols["fdr_reject"] = "fdr_significant"
    if "fdr_crit_p" in df.columns:
        cols["fdr_crit_p"] = "fdr_critical_p"
    return cols


def _process_predictor_file(
    file_path: Path,
    target: str,
    predictor_type: str,
    use_fdr: bool,
    alpha: float,
    logger,
) -> Optional[pd.DataFrame]:
    if not file_path.exists():
        return None

    dataframe = read_tsv(file_path)
    if not validate_predictor_file(dataframe, predictor_type, target, logger):
        return None

    significant_predictors = filter_significant_predictors(dataframe, use_fdr, alpha)
    if len(significant_predictors) == 0:
        return None

    significant_predictors["predictor_type"] = predictor_type
    significant_predictors["target"] = target
    significant_predictors["predictor"] = build_predictor_name(
        significant_predictors, predictor_type
    )
    
    column_mapping = build_predictor_column_mapping(predictor_type)
    column_mapping = _add_fdr_columns_to_mapping(column_mapping, significant_predictors)
    result_subset = significant_predictors[list(column_mapping.keys())].rename(
        columns=column_mapping
    )
    logger.info(
        f"Found {len(significant_predictors)} significant {predictor_type} "
        f"predictors for target '{target}'"
    )
    return result_subset


def export_all_significant_predictors(
    subject: str, alpha: float = None, use_fdr: bool = True
) -> None:
    if not subject:
        return
    
    config = load_settings()
    if alpha is None:
        alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Exporting all significant predictors for sub-{subject} (alpha={alpha})")

    all_predictors = []
    target_names = ("rating", "temp", "temperature")

    for target_name in target_names:
        roi_file_path = stats_dir / f"corr_stats_pow_roi_vs_{target_name}.tsv"
        roi_predictors = _process_predictor_file(
            roi_file_path, target_name, "ROI", use_fdr, alpha, logger
        )
        if roi_predictors is not None:
            all_predictors.append(roi_predictors)
        
        combined_file_path = stats_dir / f"corr_stats_pow_combined_vs_{target_name}.tsv"
        channel_predictors = _process_predictor_file(
            combined_file_path, target_name, "Channel", use_fdr, alpha, logger
        )
        if channel_predictors is not None:
            all_predictors.append(channel_predictors)

    output_file_path = stats_dir / "all_significant_predictors.csv"
    if all_predictors:
        combined_dataframe = pd.concat(all_predictors, ignore_index=True)
        combined_dataframe["abs_r"] = combined_dataframe["r"].abs()
        combined_dataframe = combined_dataframe.sort_values("p", ascending=True)
        combined_dataframe.to_csv(output_file_path, index=False)

        logger.info(
            f"Exported {len(combined_dataframe)} total significant predictors to: "
            f"{output_file_path}"
        )
        n_roi_predictors = len(
            combined_dataframe[combined_dataframe["type"] == "ROI"]
        )
        n_channel_predictors = len(
            combined_dataframe[combined_dataframe["type"] == "Channel"]
        )
        strongest_predictor = combined_dataframe.iloc[0]
        logger.info(
            f"Summary: {n_roi_predictors} ROI + {n_channel_predictors} channel predictors"
        )
        logger.info(
            f"Strongest predictor: {strongest_predictor['predictor']} "
            f"(r={strongest_predictor['r']:.3f})"
        )
    else:
        logger.warning("No significant predictors found")
        empty_dataframe = pd.DataFrame(
            columns=[
                "predictor",
                "region",
                "band",
                "r",
                "p",
                "n",
                "type",
                "target",
                "abs_r",
            ]
        )
        empty_dataframe.to_csv(output_file_path, index=False)


def export_combined_power_corr_stats(subject: str) -> None:
    config = load_settings()
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    for target in ("rating", "temp"):
        frames: List[pd.DataFrame] = []
        for band in bands:
            f = stats_dir / f"corr_stats_pow_{band}_vs_{target}.tsv"
            if not f.exists():
                continue
            df = read_tsv(f)
            if df is None or df.empty:
                continue
            if "band" not in df.columns:
                df["band"] = band
            else:
                df["band"] = df["band"].fillna(band)
            frames.append(df)

        if frames:
            cat = pd.concat(frames, ignore_index=True)
            out_path = stats_dir / f"corr_stats_pow_combined_vs_{target}.tsv"
            write_tsv(cat, out_path)

