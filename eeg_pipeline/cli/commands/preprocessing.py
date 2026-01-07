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
        "--line-freq",
        type=int,
        help="Line frequency for EEG data (Hz), typically 50 or 60",
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


def run_preprocessing(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the preprocessing command."""
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)

    if args.bids_root:
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if args.deriv_root:
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    
    # Preprocessing overrides
    if args.resample:
        config.setdefault("preprocessing", {})["resample_freq"] = args.resample
    if args.l_freq is not None:
        config.setdefault("preprocessing", {})["l_freq"] = args.l_freq
    if args.h_freq is not None:
        config.setdefault("preprocessing", {})["h_freq"] = args.h_freq
    if args.notch:
        config.setdefault("preprocessing", {})["notch_freq"] = args.notch
    
    # ICA overrides
    if args.ica_method:
        config.setdefault("ica", {})["method"] = args.ica_method
    if args.ica_components:
        config.setdefault("ica", {})["n_components"] = args.ica_components
    if args.prob_threshold:
        config.setdefault("ica", {})["probability_threshold"] = args.prob_threshold
    if args.ica_labels_to_keep:
        config.setdefault("ica", {})["labels_to_keep"] = args.ica_labels_to_keep
    if args.line_freq:
        config.setdefault("preprocessing", {})["line_freq"] = args.line_freq
        
    # Epoch overrides
    if args.tmin is not None:
        config.setdefault("epochs", {})["tmin"] = args.tmin
    if args.tmax is not None:
        config.setdefault("epochs", {})["tmax"] = args.tmax
    if args.baseline:
        config.setdefault("epochs", {})["baseline"] = tuple(args.baseline)
    if args.no_baseline:
        config.setdefault("epochs", {})["baseline"] = None
    if args.reject is not None:
        # Convert µV to V for MNE (MNE uses V internally)
        config.setdefault("epochs", {})["reject"] = {"eeg": args.reject * 1e-6}
    
    pipeline = PreprocessingPipeline(config=config)
    
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = config.get("preprocessing.n_jobs", 1)

    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=args.mode,
        use_pyprep=args.use_pyprep,
        use_icalabel=args.use_icalabel,
        n_jobs=n_jobs,
        progress=progress,
    )
