"""Preprocessing CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
    create_progress_reporter,
    resolve_task,
)


def setup_preprocessing(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the preprocessing command parser."""
    parser = subparsers.add_parser(
        "preprocessing",
        help="EEG preprocessing: bad channels, ICA, epochs",
        description="Run EEG preprocessing pipeline: detect bad channels, fit ICA, create epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["full", "bad-channels", "ica", "epochs"],
        help="Preprocessing mode: full (all steps), bad-channels, ica, or epochs",
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    
    prep_group = parser.add_argument_group("Preprocessing options")
    prep_group.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for bad channel detection (default from config)",
    )
    prep_group.add_argument(
        "--use-icalabel",
        action="store_true",
        default=True,
        help="Use mne-icalabel for ICA component classification (default: True)",
    )
    prep_group.add_argument(
        "--no-icalabel",
        dest="use_icalabel",
        action="store_false",
        help="Disable mne-icalabel, use MNE-BIDS pipeline ICA detection",
    )
    prep_group.add_argument(
        "--use-pyprep",
        action="store_true",
        default=True,
        help="Use PyPREP for bad channel detection (default: True)",
    )
    prep_group.add_argument(
        "--no-pyprep",
        dest="use_pyprep",
        action="store_false",
        help="Disable PyPREP bad channel detection",
    )
    prep_group.add_argument(
        "--resample",
        type=int,
        help="Resampling frequency (Hz)",
    )
    prep_group.add_argument(
        "--l-freq",
        type=float,
        help="High-pass filter frequency (Hz)",
    )
    prep_group.add_argument(
        "--h-freq",
        type=float,
        help="Low-pass filter frequency (Hz)",
    )
    prep_group.add_argument(
        "--notch",
        type=int,
        help="Notch filter frequency (Hz)",
    )
    prep_group.add_argument(
        "--ica-method",
        choices=["fastica", "infomax", "picard"],
        help="ICA method",
    )
    prep_group.add_argument(
        "--ica-components",
        type=float,
        help="Number of ICA components (int) or variance fraction (float)",
    )
    prep_group.add_argument(
        "--prob-threshold",
        type=float,
        help="ICA label probability threshold",
    )
    prep_group.add_argument(
        "--ica-labels-to-keep",
        nargs="+",
        help="ICA component labels to keep (e.g., brain other)",
    )
    prep_group.add_argument(
        "--keep-mnebids-bads",
        action="store_true",
        default=False,
        help="Keep MNE-BIDS flagged bad ICA components",
    )
    prep_group.add_argument(
        "--line-freq",
        type=int,
        help="Line frequency for EEG data (Hz), typically 50 or 60",
    )
    # PyPREP advanced options
    prep_group.add_argument(
        "--ransac",
        action="store_true",
        default=False,
        help="Use RANSAC for bad channel detection",
    )
    prep_group.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of bad channel detection iterations",
    )
    prep_group.add_argument(
        "--average-reref",
        action="store_true",
        default=False,
        help="Average re-reference before bad channel detection",
    )
    prep_group.add_argument(
        "--file-extension",
        type=str,
        default=".vhdr",
        help="File extension for EEG data",
    )
    prep_group.add_argument(
        "--consider-previous-bads",
        action="store_true",
        default=False,
        help="Keep previously marked bad channels",
    )
    prep_group.add_argument(
        "--overwrite-channels-tsv",
        action="store_true",
        default=True,
        help="Overwrite channels.tsv file with detected bad channels",
    )
    prep_group.add_argument(
        "--no-overwrite-channels-tsv",
        dest="overwrite_channels_tsv",
        action="store_false",
        help="Do not overwrite channels.tsv file",
    )
    prep_group.add_argument(
        "--delete-breaks",
        action="store_true",
        default=False,
        help="Delete breaks in data during bad channel detection",
    )
    prep_group.add_argument(
        "--breaks-min-length",
        type=int,
        default=20,
        help="Minimum break duration in seconds",
    )
    prep_group.add_argument(
        "--t-start-after-previous",
        type=int,
        default=2,
        help="Time after previous event in seconds",
    )
    prep_group.add_argument(
        "--t-stop-before-next",
        type=int,
        default=2,
        help="Time before next event in seconds",
    )
    prep_group.add_argument(
        "--tmin",
        type=float,
        help="Epoch start time (s)",
    )
    prep_group.add_argument(
        "--tmax",
        type=float,
        help="Epoch end time (s)",
    )
    prep_group.add_argument(
        "--baseline",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Epoch baseline window (start end) in seconds, e.g., -0.2 0",
    )
    prep_group.add_argument(
        "--no-baseline",
        action="store_true",
        help="Disable epoch baseline correction",
    )
    prep_group.add_argument(
        "--reject",
        type=float,
        help="Peak-to-peak amplitude rejection threshold (µV)",
    )

    add_path_args(parser)
    
    return parser


def _validate_epoch_parameters(args: argparse.Namespace) -> None:
    """Validate epoch parameter constraints."""
    if args.tmin is not None and args.tmax is not None:
        if args.tmin >= args.tmax:
            raise ValueError(
                f"Epoch start time ({args.tmin}s) must be less than end time ({args.tmax}s)"
            )
    
    if args.baseline is not None:
        baseline_start, baseline_end = args.baseline
        if baseline_start >= baseline_end:
            raise ValueError(
                f"Baseline start ({baseline_start}s) must be less than end ({baseline_end}s)"
            )
        if args.tmin is not None and baseline_end > args.tmin:
            raise ValueError(
                f"Baseline end ({baseline_end}s) must not exceed epoch start ({args.tmin}s)"
            )


def _update_path_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with path overrides from arguments."""
    if args.bids_root:
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if args.deriv_root:
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root


def _update_preprocessing_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with preprocessing parameter overrides."""
    preprocessing_config = config.setdefault("preprocessing", {})
    
    if args.resample:
        preprocessing_config["resample_freq"] = args.resample
    if args.l_freq is not None:
        preprocessing_config["l_freq"] = args.l_freq
    if args.h_freq is not None:
        preprocessing_config["h_freq"] = args.h_freq
    if args.notch:
        preprocessing_config["notch_freq"] = args.notch
    if args.line_freq:
        preprocessing_config["line_freq"] = args.line_freq


def _update_ica_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with ICA parameter overrides."""
    ica_config = config.setdefault("ica", {})
    
    if args.ica_method:
        ica_config["method"] = args.ica_method
    if args.ica_components:
        ica_config["n_components"] = args.ica_components
    if args.prob_threshold:
        ica_config["probability_threshold"] = args.prob_threshold
    if args.ica_labels_to_keep:
        ica_config["labels_to_keep"] = args.ica_labels_to_keep


def _update_epochs_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with epoch parameter overrides."""
    epochs_config = config.setdefault("epochs", {})
    
    if args.tmin is not None:
        epochs_config["tmin"] = args.tmin
    if args.tmax is not None:
        epochs_config["tmax"] = args.tmax
    if args.baseline:
        epochs_config["baseline"] = tuple(args.baseline)
    if args.no_baseline:
        epochs_config["baseline"] = None
    if args.reject is not None:
        microvolts_to_volts = 1e-6
        epochs_config["reject"] = {"eeg": args.reject * microvolts_to_volts}


def _resolve_n_jobs(args: argparse.Namespace, config: Any) -> int:
    """Resolve number of parallel jobs from args or config."""
    if args.n_jobs is not None:
        return args.n_jobs
    return config.get("preprocessing.n_jobs", 1)


def run_preprocessing(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the preprocessing command."""
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
    
    _validate_epoch_parameters(args)
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)

    _update_path_config(args, config)
    _update_preprocessing_config(args, config)
    _update_ica_config(args, config)
    _update_epochs_config(args, config)
    
    pipeline = PreprocessingPipeline(config=config)
    n_jobs = _resolve_n_jobs(args, config)

    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=args.mode,
        use_pyprep=args.use_pyprep,
        use_icalabel=args.use_icalabel,
        n_jobs=n_jobs,
        progress=progress,
    )
