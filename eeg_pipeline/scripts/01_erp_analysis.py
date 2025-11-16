# Standard library
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import logging
import os
import warnings

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Third-party
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# Local - config and data
from eeg_pipeline.utils.config_loader import load_settings, load_config
from eeg_pipeline.utils.data_loading import (
    get_available_subjects,
    load_epochs_for_analysis,
    parse_subject_args,
    crop_epochs,
)
from eeg_pipeline.utils.io_utils import (
    deriv_plots_path,
    ensure_derivatives_dataset_description,
    ensure_dir,
    setup_matplotlib,
    extract_subject_id_from_path,
    find_pain_column_in_metadata,
    write_group_trial_counts,
)
from eeg_pipeline.utils.stats_utils import (
    count_trials_by_condition,
)

# Local - plotting
from eeg_pipeline.plotting.plot_features import (
    erp_by_temperature,
    erp_contrast_pain,
    group_erp_by_temperature,
    group_erp_contrast_pain,
)


###################################################################
# Configuration Extraction
###################################################################

def _extract_erp_config(config) -> Dict[str, Any]:
    return {
        "deriv_root": Path(config.get("paths.deriv_root")),
        "bids_root": Path(config.get("paths.bids_root")),
        "task": config.get("project.task", "thermalactive"),
        "fig_dpi": config.get("output.fig_dpi", 300),
        "erp_picks": config.get("foundational_analysis.erp.picks", "eeg"),
        "pain_columns": config.get("event_columns.pain_binary", []),
        "temperature_columns": config.get("event_columns.temperature", []),
        "baseline_window": tuple(config.get("foundational_analysis.erp.baseline_window", [-0.2, 0.0])),
        "fig_pad_inch": float(config.get("output.pad_inches", 0.02)),
        "bbox_inches": config.get("output.bbox_inches", "tight"),
        "pain_color": config.get("foundational_analysis.erp.pain_color", "crimson"),
        "nonpain_color": config.get("foundational_analysis.erp.nonpain_color", "navy"),
        "include_tmax_in_crop": bool(config.get("foundational_analysis.erp.include_tmax_in_crop", False)),
        "default_crop_tmin": config.get("foundational_analysis.erp.default_crop_tmin", None),
        "default_crop_tmax": config.get("foundational_analysis.erp.default_crop_tmax", None),
        "erp_combine": config.get("foundational_analysis.erp.combine", "gfp"),
        "plots_subdir": config.get("foundational_analysis.erp.plots_subdir", "01_erp_analysis"),
        "counts_file_name": config.get("foundational_analysis.erp.counts_file_name", "counts_pain.tsv"),
        "output_files": dict(config.get("foundational_analysis.erp.output_files", {
            "pain_gfp": "erp_pain_binary_gfp.png",
            "pain_butterfly": "erp_pain_binary_butterfly.png",
            "temp_gfp": "erp_by_temperature_gfp.png",
            "temp_butterfly": "erp_by_temperature_butterfly.png",
            "temp_butterfly_template": "erp_by_temperature_butterfly_{label}.png",
        })),
    }


###################################################################
# Helper Functions
###################################################################




###################################################################
# Main Processing Functions
###################################################################

def _prepare_epochs_for_erp(
    subject: str,
    task: str,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    logger.info(f"  Loading epochs...")
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False, deriv_root=config.deriv_root, bids_root=config.bids_root, config=config, logger=logger
    )
    
    if epochs is None:
        error_msg = f"Failed to load epochs for sub-{subject}, task-{task}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if aligned_events is None:
        error_msg = f"Events TSV not found for sub-{subject}, task-{task}; cannot perform ERP contrasts"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"  Loaded {len(epochs)} epochs")
    
    epochs.metadata = aligned_events
    
    if crop_tmin is not None or crop_tmax is not None:
        crop_tmin_str = f"{crop_tmin:.3f}" if crop_tmin is not None else "None"
        crop_tmax_str = f"{crop_tmax:.3f}" if crop_tmax is not None else "None"
        logger.info(f"  Cropping epochs to [{crop_tmin_str}, {crop_tmax_str}] s")
        epochs = crop_epochs(epochs, crop_tmin, crop_tmax, include_tmax_in_crop, logger)
    
    return epochs, aligned_events


def _extract_trial_counts(epochs: mne.Epochs, config) -> Dict[str, Any]:
    result = {
        "n_trials_pain": 0,
        "n_trials_nonpain": 0,
        "temperatures_detected": [],
    }
    
    if epochs.metadata is None:
        return result
    
    pain_column = find_pain_column_in_metadata(epochs, config)
    if pain_column:
        n_pain, n_nonpain = count_trials_by_condition(epochs, pain_column, logger=None)
        result["n_trials_pain"] = n_pain
        result["n_trials_nonpain"] = n_nonpain
    
    temperature_columns = config.get("event_columns.temperature", [])
    temperature_column = next(
        (col for col in temperature_columns if col in epochs.metadata.columns),
        None
    ) if temperature_columns else None
    
    if temperature_column:
        temp_values = epochs.metadata[temperature_column]
        result["temperatures_detected"] = sorted(temp_values.unique().tolist())
    
    return result




def _run_erp_analysis(
    epochs: mne.Epochs,
    aligned_events: pd.DataFrame,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    pain_color: str,
    nonpain_color: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    plots_dir: Path,
    config,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    trial_counts = _extract_trial_counts(epochs, config)
    logger.info(
        f"  Computing pain contrast (pain: {trial_counts['n_trials_pain']}, "
        f"non-pain: {trial_counts['n_trials_nonpain']})"
    )
    
    subject_id = extract_subject_id_from_path(plots_dir)
    
    erp_contrast_pain(
        epochs, plots_dir, config, baseline_window, erp_picks,
        pain_color, nonpain_color, erp_combine, erp_output_files,
        logger, subject=subject_id
    )
    
    logger.info("  Computing temperature analysis")
    erp_by_temperature(
        epochs, plots_dir, config, baseline_window, erp_picks,
        erp_combine, erp_output_files, logger, subject=subject_id
    )
    
    output_paths = [
        str(plots_dir / output_name)
        for output_name in erp_output_files.values()
        if "{label}" not in output_name
    ]
    
    return output_paths


def _load_and_prepare_epochs(
    subject: str,
    task: str,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    config,
    logger: logging.Logger,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    epochs, aligned_events = _prepare_epochs_for_erp(
        subject, task, crop_tmin, crop_tmax, include_tmax_in_crop, config, logger
    )
    return epochs, aligned_events


def _collect_results(
    epochs: Optional[mne.Epochs],
    aligned_events: Optional[pd.DataFrame],
    config,
    logger: logging.Logger,
) -> Dict[str, Any]:
    result = {
        "n_trials_pain": 0,
        "n_trials_nonpain": 0,
        "temperatures_detected": [],
        "output_paths": []
    }
    
    if epochs is None or aligned_events is None:
        return result
    
    trial_counts = _extract_trial_counts(epochs, config)
    result.update(trial_counts)
    
    return result


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
    config = None,
    no_plots: bool = False,
) -> Dict[str, Any]:
    if not subject or not isinstance(subject, str):
        raise ValueError(f"subject must be non-empty string, got: {subject}")
    if not task or not isinstance(task, str):
        raise ValueError(f"task must be non-empty string, got: {task}")
    
    if config is None:
        config = load_settings()
    if logger is None:
        logger = logging.getLogger(__name__)
    
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir=plots_subdir)
    ensure_dir(plots_dir)
    
    epochs, aligned_events = _load_and_prepare_epochs(
        subject, task, crop_tmin, crop_tmax, include_tmax_in_crop, config, logger
    )
    
    result = _collect_results(epochs, aligned_events, config, logger)
    
    if epochs is None or aligned_events is None:
        return result
    
    if not no_plots:
        output_paths = _run_erp_analysis(
            epochs, aligned_events, baseline_window, erp_picks, pain_color, nonpain_color,
            erp_combine, erp_output_files, plots_dir, config, logger
        )
        result["output_paths"] = output_paths
    else:
        logger.info("  Skipping plotting (--no-plots enabled)")
        result["output_paths"] = []
    
    logger.info(f"  Completed sub-{subject}")
    
    return result


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    
    mne.set_log_level('WARNING')
    
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', message='.*tight_layout.*')
    
    return logger


###################################################################
# Main Entry Point
###################################################################

def _load_epochs_for_group_analysis(
    subject: str,
    task: str,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    config_obj,
    logger: logging.Logger,
) -> Optional[mne.Epochs]:
    deriv_root = Path(config_obj.get("paths.deriv_root"))
    bids_root = Path(config_obj.get("paths.bids_root"))
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False,
        deriv_root=deriv_root, bids_root=bids_root,
        config=config_obj, logger=logger
    )
    
    if epochs is None or aligned_events is None:
        return None
    
    epochs.metadata = aligned_events
    if crop_tmin is not None or crop_tmax is not None:
        epochs = crop_epochs(epochs, crop_tmin, crop_tmax, include_tmax_in_crop, logger)
    
    return epochs


def _process_subjects_for_group_analysis(
    subjects: List[str],
    task: str,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    config: Dict[str, Any],
    config_obj,
    logger: logging.Logger,
    no_plots: bool = False,
) -> Tuple[List[mne.Epochs], List[str]]:
    all_epochs = []
    successful_subjects = []
    min_trials_per_condition = config_obj.get("analysis.min_trials_per_condition", 5)
    
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Processing sub-{subject}")
        result = process_single_subject(
            subject, task, crop_tmin, crop_tmax, config["baseline_window"],
            config["erp_picks"], config["pain_color"], config["nonpain_color"],
            config["erp_combine"], config["output_files"],
            config["include_tmax_in_crop"], config["plots_subdir"], logger, config_obj,
            no_plots=no_plots
        )
        
        has_sufficient_trials = (
            result["n_trials_pain"] >= min_trials_per_condition and
            result["n_trials_nonpain"] >= min_trials_per_condition
        )
        
        if not has_sufficient_trials:
            logger.warning(
                f"  Insufficient trials for group analysis: "
                f"pain={result['n_trials_pain']}, nonpain={result['n_trials_nonpain']} "
                f"(minimum {min_trials_per_condition} per condition required)"
            )
            continue
        
        epochs = _load_epochs_for_group_analysis(
            subject, task, crop_tmin, crop_tmax,
            config["include_tmax_in_crop"], config_obj, logger
        )
        
        if epochs is None:
            logger.warning("  Failed to load epochs for group analysis")
            continue
        
        all_epochs.append(epochs)
        successful_subjects.append(subject)
    
    return all_epochs, successful_subjects


def _run_group_analysis(
    all_epochs: List[mne.Epochs],
    successful_subjects: List[str],
    config: Dict[str, Any],
    config_obj,
    logger: logging.Logger,
    no_plots: bool = False,
) -> None:
    group_dir = config["deriv_root"] / "group" / "eeg" / "plots" / config["plots_subdir"]
    ensure_dir(group_dir)
    
    if not no_plots:
        logger.info("  Computing group pain contrast")
        group_erp_contrast_pain(
            all_epochs, group_dir, config_obj, config["baseline_window"],
            config["erp_picks"], config["pain_color"], config["nonpain_color"],
            config["erp_combine"], config["output_files"], logger
        )
        
        logger.info("  Computing group temperature analysis")
        group_erp_by_temperature(
            all_epochs, group_dir, config_obj, config["baseline_window"],
            config["erp_picks"], config["erp_combine"], config["output_files"], logger
        )
    else:
        logger.info("  Skipping group plotting (--no-plots enabled)")
    
    pain_counts_list = []
    for epochs in all_epochs:
        pain_column = find_pain_column_in_metadata(epochs, config_obj)
        n_pain, n_nonpain = count_trials_by_condition(epochs, pain_column, logger) if pain_column else (0, 0)
        pain_counts_list.append((n_pain, n_nonpain))
    
    write_group_trial_counts(
        successful_subjects, group_dir,
        config["counts_file_name"], pain_counts_list, logger
    )
    
    logger.info(f"  Group analysis saved to: {group_dir}")


def main(
    subjects: Optional[List[str]] = None,
    all_subjects: bool = False,
    task: Optional[str] = None,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
    group: Optional[str] = None,
    no_plots: bool = False,
) -> None:
    logger = setup_logging()
    
    config_obj = load_config()
    setup_matplotlib(config_obj)
    ensure_derivatives_dataset_description(config=config_obj)
    
    config = _extract_erp_config(config_obj)
    
    task = task or config["task"]
    crop_tmin = crop_tmin or config["default_crop_tmin"]
    crop_tmax = crop_tmax or config["default_crop_tmax"]
    
    if not subjects:
        raise ValueError(
            "No subjects specified. Use --group all|A,B,C, or --subject (can repeat) "
            "or --all-subjects."
        )
    
    max_subjects = int(os.getenv("ERP_MAX_SUBJECTS", "0"))
    if max_subjects > 0 and len(subjects) > max_subjects:
        subjects = subjects[:max_subjects]
    
    logger.info(f"Starting ERP contrast analysis: {len(subjects)} subject(s), task={task}")
    logger.info(f"Subjects: {', '.join(subjects)}")
    
    all_epochs, successful_subjects = _process_subjects_for_group_analysis(
        subjects, task, crop_tmin, crop_tmax, config, config_obj, logger, no_plots=no_plots
    )
    
    if len(all_epochs) >= 2:
        logger.info(f"Running group analysis ({len(successful_subjects)} subjects)")
        _run_group_analysis(all_epochs, successful_subjects, config, config_obj, logger, no_plots=no_plots)
    else:
        logger.warning(f"Insufficient subjects for group analysis ({len(all_epochs)} < 2)")
    
    logger.info(
        f"Analysis complete: {len(successful_subjects)}/{len(subjects)} subjects processed successfully"
    )

###################################################################
# Command Line Interface
###################################################################

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    config = load_settings()
    
    parser = argparse.ArgumentParser(
        description="EEG ERP analysis supporting single and multiple subjects"
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
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting operations"
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
        no_plots=args.no_plots,
    )
