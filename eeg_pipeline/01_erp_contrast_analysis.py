from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config_loader import load_settings, load_config
from eeg_pipeline.utils.data_loading import (
    get_available_subjects,
    load_epochs_for_analysis,
    parse_subject_args,
    process_temperature_levels,
    select_epochs_by_value,
)
from eeg_pipeline.utils.tfr_utils import validate_baseline_window_for_times
from eeg_pipeline.utils.io_utils import (
    save_fig, 
    build_footer,
    _find_clean_epochs_path,
    ensure_derivatives_dataset_description,
    ensure_dir,
    deriv_plots_path,
    setup_matplotlib,
    unwrap_figure as _unwrap_figure,
    sanitize_label as _sanitize_label,
)
from eeg_pipeline.plotting.plot_features import (
    erp_contrast_pain,
    erp_by_temperature,
    group_erp_contrast_pain,
    group_erp_by_temperature,
)

import matplotlib.pyplot as plt
import argparse


###################################################################
# Data Processing
###################################################################


def _validate_baseline_window(epochs: mne.Epochs, baseline_window: Tuple[float, float], logger: Optional[logging.Logger] = None, config=None) -> bool:
    
    baseline_start, baseline_end = float(baseline_window[0]), float(baseline_window[1])
    times = np.asarray(epochs.times)
    
    try:
        b_start, b_end, idx = validate_baseline_window_for_times(
            times,
            baseline_window,
            logger=logger,
            config=config
        )
        if logger:
            timespan = (float(times[idx][0]), float(times[idx][-1]))
            logger.info(f"Baseline validation: window [{b_start:.3f}, {b_end:.3f}] s maps to indices [{idx[0]}, {idx[-1]}] with timespan [{timespan[0]:.3f}, {timespan[1]:.3f}] s")
        return True
    except ValueError as e:
        if logger:
            logger.error(str(e))
        raise

def _apply_baseline(epochs: mne.Epochs, baseline_window: Tuple[float, float], logger: Optional[logging.Logger] = None, config=None) -> mne.Epochs:
    _validate_baseline_window(epochs, baseline_window, logger, config)
    baseline_start = float(baseline_window[0])
    baseline_end = float(baseline_window[1])
    return epochs.copy().apply_baseline((baseline_start, min(baseline_end, 0.0)))

def _crop_epochs(
    epochs: mne.Epochs,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    logger: Optional[logging.Logger] = None
) -> mne.Epochs:
    if crop_tmin is None and crop_tmax is None:
        return epochs
    
    time_min = epochs.tmin if crop_tmin is None else float(crop_tmin)
    time_max = epochs.tmax if crop_tmax is None else float(crop_tmax)
    
    if time_max <= time_min:
        raise ValueError(f"Invalid crop window: tmin={time_min}, tmax={time_max}")
    
    if logger:
        logger.info(
            f"Cropping epochs to [{time_min:.3f}, {time_max:.3f}] s "
            f"(include_tmax={include_tmax_in_crop})"
        )
    
    epochs_copy = epochs.copy()
    if not getattr(epochs_copy, "preload", False):
        epochs_copy.load_data()
    
    return epochs_copy.crop(tmin=time_min, tmax=time_max, include_tmax=include_tmax_in_crop)

def _build_epoch_query_and_label(
    column: str,
    level: any,
    is_numeric: bool,
    labels: Dict
) -> Tuple[str, str]:
    if is_numeric:
        return f"{column} == {level}", labels[level]
    
    escaped_level = str(level).replace('"', '\\"')
    query = f'{column} == "{escaped_level}"'
    return query, str(level)


###################################################################
# Figure Saving
###################################################################


###################################################################
# ERP Analysis Functions
###################################################################

# Functions are imported directly from eeg_pipeline.plotting.plot_features

def _write_group_pain_counts(
    all_epochs: List[mne.Epochs],
    subjects: List[str],
    output_dir: Path,
    config,
    counts_file_name: str,
    logger: Optional[logging.Logger] = None
) -> None:
    rows = []
    for subject, epochs in zip(subjects, all_epochs):
        n_pain = 0
        n_nonpain = 0
        
        if epochs.metadata is not None:
            pain_columns = config.get("event_columns.pain_binary", []) if config else []
            pain_column = next((col for col in pain_columns if col in epochs.metadata.columns), None) if pain_columns else None
            if pain_column is not None:
                pain_values = pd.to_numeric(epochs.metadata[pain_column], errors="coerce")
                n_pain = int((pain_values == 1).sum())
                n_nonpain = int((pain_values == 0).sum())
        
        rows.append({
            "subject": subject,
            "n_pain": n_pain,
            "n_nonpain": n_nonpain,
            "n_total": n_pain + n_nonpain
        })
    
    if not rows:
        return
    
    counts_df = pd.DataFrame(rows)
    totals = counts_df[["n_pain", "n_nonpain", "n_total"]].sum()
    total_row = {
        "subject": "TOTAL",
        **{key: int(value) for key, value in totals.to_dict().items()}
    }
    counts_df = pd.concat([counts_df, pd.DataFrame([total_row])], ignore_index=True)
    
    ensure_dir(output_dir)
    output_path = output_dir / counts_file_name
    counts_df.to_csv(output_path, sep="\t", index=False)
    if logger:
        logger.info(f"Saved counts: {output_path}")

###################################################################
# Main Processing Functions
###################################################################

def process_single_subject(
    subject: str,
    task: str,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    baseline_window: Tuple[float, float],
    erp_picks: str,
    pain_color: str,
    nonpain_color: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    include_tmax_in_crop: bool,
    plots_subdir: str,
    logger: Optional[logging.Logger] = None,
    config = None
) -> Dict[str, Any]:
    if config is None:
        config = load_settings()
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"=== Processing sub-{subject}, task-{task} ===")
    
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir=plots_subdir)
    ensure_dir(plots_dir)
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False, deriv_root=config.deriv_root, bids_root=config.bids_root, config=config, logger=logger
    )
    
    result = {
        "n_trials_pain": 0,
        "n_trials_nonpain": 0,
        "temperatures_detected": [],
        "output_paths": []
    }
    
    if epochs is None:
        return result
    
    if aligned_events is None:
        logger.warning("Events TSV not found; ERP contrasts will be skipped.")
        return result
    
    # Set epochs metadata from aligned events
    epochs.metadata = aligned_events
    
    if crop_tmin is not None or crop_tmax is not None:
        epochs = _crop_epochs(epochs, crop_tmin, crop_tmax, include_tmax_in_crop, logger)
    
    if aligned_events is not None and epochs.metadata is not None:
        pain_columns = config.get("event_columns.pain_binary", []) if config else []
        pain_column = next((col for col in pain_columns if col in epochs.metadata.columns), None) if pain_columns else None
        temperature_columns = config.get("event_columns.temperature", []) if config else []
        temperature_column = next((col for col in temperature_columns if col in epochs.metadata.columns), None) if temperature_columns else None
        
        if pain_column is not None:
            pain_values = pd.to_numeric(epochs.metadata[pain_column], errors="coerce")
            result["n_trials_pain"] = int((pain_values == 1).sum())
            result["n_trials_nonpain"] = int((pain_values == 0).sum())
        
        if temperature_column is not None:
            temp_values = epochs.metadata[temperature_column]
            result["temperatures_detected"] = sorted(temp_values.unique().tolist())
        
        erp_contrast_pain(epochs, plots_dir, config, baseline_window, erp_picks, pain_color, nonpain_color, erp_combine, erp_output_files, logger)
        erp_by_temperature(epochs, plots_dir, config, baseline_window, erp_picks, erp_combine, erp_output_files, logger)
        
        for output_name in erp_output_files.values():
            if "{label}" not in output_name:
                result["output_paths"].append(str(plots_dir / output_name))
    
    logger.info("Single subject processing completed.")
    return result


###################################################################
# Main Entry Point
###################################################################

def main(
    subjects: Optional[List[str]] = None,
    all_subjects: bool = False,
    task: Optional[str] = None,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
    group: Optional[str] = None,
) -> None:
    config = load_config()
    setup_matplotlib(config)
    ensure_derivatives_dataset_description(config=config)
    
    from pathlib import Path
    DERIV_ROOT = Path(config.get("paths.deriv_root"))
    DEFAULT_TASK = config.get("project.task", "thermalactive")
    if task is None:
        task = DEFAULT_TASK
    
    FIG_DPI = config.get("output.fig_dpi", 300)
    ERP_PICKS = config.get("foundational_analysis.erp.picks", "eeg")
    PAIN_COLUMNS = config.get("event_columns.pain_binary", [])
    TEMPERATURE_COLUMNS = config.get("event_columns.temperature", [])
    ERP_BASELINE_WINDOW = tuple(config.get("foundational_analysis.erp.baseline_window", [-0.2, 0.0]))
    
    FIG_PAD_INCH = float(config.get("output.pad_inches", 0.02))
    BBOX_INCHES = config.get("output.bbox_inches", "tight")
    PAIN_COLOR = config.get("foundational_analysis.erp.pain_color", "crimson")
    NONPAIN_COLOR = config.get("foundational_analysis.erp.nonpain_color", "navy")
    INCLUDE_TMAX_IN_CROP = bool(config.get("foundational_analysis.erp.include_tmax_in_crop", False))
    DEFAULT_CROP_TMIN = config.get("foundational_analysis.erp.default_crop_tmin", None)
    DEFAULT_CROP_TMAX = config.get("foundational_analysis.erp.default_crop_tmax", None)
    ERP_COMBINE = config.get("foundational_analysis.erp.combine", "gfp")
    PLOTS_SUBDIR = config.get("foundational_analysis.erp.plots_subdir", "01_erp_contrast_analysis")
    COUNTS_FILE_NAME = config.get("foundational_analysis.erp.counts_file_name", "counts_pain.tsv")
    ERP_OUT = dict(config.get("foundational_analysis.erp.output_files", {
        "pain_gfp": "erp_pain_binary_gfp.png",
        "pain_butterfly": "erp_pain_binary_butterfly.png",
        "temp_gfp": "erp_by_temperature_gfp.png",
        "temp_butterfly": "erp_by_temperature_butterfly.png",
        "temp_butterfly_template": "erp_by_temperature_butterfly_{label}.png",
    }))
    if crop_tmin is None:
        crop_tmin = DEFAULT_CROP_TMIN
    if crop_tmax is None:
        crop_tmax = DEFAULT_CROP_TMAX
    
    if not subjects:
        raise ValueError(
            "No subjects specified. Use --group all|A,B,C, or --subject (can repeat) "
            "or --all-subjects."
        )
    
    max_subjects = int(os.getenv("ERP_MAX_SUBJECTS", "0"))
    if max_subjects > 0 and len(subjects) > max_subjects:
        subjects = subjects[:max_subjects]
    
    logger = logging.getLogger(__name__)
    logger.info(
        f"=== Multi-subject foundational analysis: {len(subjects)} subjects, task-{task} ==="
    )
    logger.info(f"Subjects: {', '.join(subjects)}")
    
    all_epochs = []
    successful_subjects = []
    
    for subject in subjects:
        logger.info(f"--- Processing subject: {subject} ---")
        result = process_single_subject(
            subject, task, crop_tmin, crop_tmax, ERP_BASELINE_WINDOW, 
            ERP_PICKS, PAIN_COLOR, NONPAIN_COLOR, ERP_COMBINE, ERP_OUT,
            INCLUDE_TMAX_IN_CROP, PLOTS_SUBDIR, logger, config
        )
        
        if result["n_trials_pain"] + result["n_trials_nonpain"] > 0:
            from pathlib import Path
            deriv_root = Path(config.get("paths.deriv_root"))
            bids_root = Path(config.get("paths.bids_root"))
            epochs, aligned_events = load_epochs_for_analysis(
                subject, task, align="strict", preload=False, deriv_root=deriv_root, bids_root=bids_root, config=config, logger=logger
            )
            if epochs is not None and aligned_events is not None:
                epochs.metadata = aligned_events
                if crop_tmin is not None or crop_tmax is not None:
                    epochs = _crop_epochs(epochs, crop_tmin, crop_tmax, INCLUDE_TMAX_IN_CROP, logger)
                all_epochs.append(epochs)
                successful_subjects.append(subject)
            else:
                logger.warning(f"Failed to load epochs for subject {subject}, excluding from group analysis")
        else:
            logger.warning(f"Failed to process subject {subject}, excluding from group analysis")
    
    if len(all_epochs) >= 2:
        logger.info(f"=== Group analysis: {len(successful_subjects)} successful subjects ===")
        from pathlib import Path
        group_dir = Path(config.get("paths.deriv_root")) / "group" / "eeg" / "plots" / PLOTS_SUBDIR
        ensure_dir(group_dir)
        group_erp_contrast_pain(all_epochs, group_dir, config, ERP_BASELINE_WINDOW, ERP_PICKS, PAIN_COLOR, NONPAIN_COLOR, ERP_COMBINE, ERP_OUT, logger)
        group_erp_by_temperature(all_epochs, group_dir, config, ERP_BASELINE_WINDOW, ERP_PICKS, ERP_COMBINE, ERP_OUT, logger)
        _write_group_pain_counts(all_epochs, successful_subjects, group_dir, config, COUNTS_FILE_NAME, logger)
        logger.info(f"Group analysis completed. Results saved to: {group_dir}")
    else:
        logger.warning(
            f"Only {len(all_epochs)} subjects processed successfully."
        )
    
    logger.info("=== Analysis complete ===")
    if successful_subjects:
        logger.info(f"Successfully processed: {', '.join(successful_subjects)}")
    
    failed_subjects = set(subjects) - set(successful_subjects)
    if failed_subjects:
        logger.warning(f"Failed to process: {', '.join(failed_subjects)}")

###################################################################
# Command Line Interface
###################################################################

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    config = load_settings()
    
    parser = argparse.ArgumentParser(
        description="Foundational EEG ERP analysis supporting single and multiple subjects"
    )
    
    subject_group = parser.add_mutually_exclusive_group()
    subject_group.add_argument(
        "--group",
        type=str,
        help=(
            "Group of subjects to process: either 'all' or a comma/space-separated list "
            "of BIDS labels without 'sub-' (e.g., '0001,0002,0003')."
        )
    )
    subject_group.add_argument(
        "--subject",
        "-s",
        type=str,
        action="append",
        help=(
            "BIDS subject label(s) without 'sub-' prefix (e.g., 0000). "
            "Can be specified multiple times for multiple subjects."
        )
    )
    subject_group.add_argument(
        "--all-subjects",
        action="store_true",
        help="Process all available subjects with cleaned epochs files"
    )
    
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        help="BIDS task label (default from config)"
    )
    parser.add_argument(
        "--crop-tmin",
        type=float,
        default=None,
        help="ERP epoch crop start time (s, default from config)"
    )
    parser.add_argument(
        "--crop-tmax",
        type=float,
        default=None,
        help="ERP epoch crop end time (s, default from config)"
    )
    
    args = parser.parse_args()
    
    config = load_settings()
    
    subjects = parse_subject_args(args, config, task=args.task)
    if not subjects:
        print("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        raise SystemExit(2)
    
    main(
        subjects=subjects,
        all_subjects=False,
        task=args.task,
        crop_tmin=args.crop_tmin,
        crop_tmax=args.crop_tmax,
        group=None,
    )
