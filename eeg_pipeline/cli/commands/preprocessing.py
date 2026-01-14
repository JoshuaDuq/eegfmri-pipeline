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
    parser.add_argument("mode", choices=["full", "bad-channels", "ica", "epochs"], help="Preprocessing mode: full (all steps), bad-channels, ica, or epochs")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)

    prep_group = parser.add_argument_group("Preprocessing options")

    # Core options
    prep_group.add_argument("--n-jobs", type=int, default=None, help="Number of parallel jobs for bad channel detection (default from config)")
    prep_group.add_argument("--montage", type=str, default=None, help="EEG montage name (e.g., easycap-M1)")
    prep_group.add_argument("--ch-types", type=str, default=None, help="Channel types to include (e.g., 'eeg')")
    prep_group.add_argument("--eeg-reference", type=str, default=None, help="EEG reference type (e.g., 'average')")
    prep_group.add_argument("--eog-channels", type=str, default=None, help="EOG channels (comma-separated, e.g., 'Fp1,Fp2')")
    prep_group.add_argument("--random-state", type=int, default=None, help="Random seed for reproducibility")
    prep_group.add_argument("--task-is-rest", action="store_true", default=False, help="Whether task is resting-state")
    prep_group.add_argument("--use-icalabel", action="store_true", default=True, help="Use mne-icalabel for ICA component classification (default: True)")
    prep_group.add_argument("--no-icalabel", dest="use_icalabel", action="store_false", help="Disable mne-icalabel, use MNE-BIDS pipeline ICA detection")
    prep_group.add_argument("--use-pyprep", action="store_true", default=True, help="Use PyPREP for bad channel detection (default: True)")
    prep_group.add_argument("--no-pyprep", dest="use_pyprep", action="store_false", help="Disable PyPREP bad channel detection")

    # Filtering
    prep_group.add_argument("--resample", type=int, help="Resampling frequency (Hz)")
    prep_group.add_argument("--l-freq", type=float, help="High-pass filter frequency (Hz)")
    prep_group.add_argument("--h-freq", type=float, help="Low-pass filter frequency (Hz)")
    prep_group.add_argument("--notch", type=int, help="Notch filter frequency (Hz)")
    prep_group.add_argument("--line-freq", type=int, help="Line frequency for EEG data (Hz), typically 50 or 60")
    prep_group.add_argument("--zapline-fline", type=float, help="Zapline filtering frequency (Hz)")

    # ICA
    prep_group.add_argument("--spatial-filter", choices=["ica", "ssp"], help="Spatial filter type (ica or ssp)")
    prep_group.add_argument("--ica-method", choices=["extended_infomax", "fastica", "infomax", "picard"], help="ICA algorithm")
    prep_group.add_argument("--ica-components", type=float, help="Number of ICA components (int) or variance fraction (float)")
    prep_group.add_argument("--ica-l-freq", type=float, help="ICA high-pass filter frequency (Hz)")
    prep_group.add_argument("--ica-reject", type=float, help="ICA rejection threshold (µV)")
    prep_group.add_argument("--prob-threshold", type=float, help="ICA label probability threshold")
    prep_group.add_argument("--ica-labels-to-keep", nargs="+", help="ICA component labels to keep (e.g., brain other)")
    prep_group.add_argument("--keep-mnebids-bads", action="store_true", default=False, help="Keep MNE-BIDS flagged bad ICA components")

    # PyPREP advanced options
    prep_group.add_argument("--ransac", action="store_true", default=False, help="Use RANSAC for bad channel detection")
    prep_group.add_argument("--repeats", type=int, default=3, help="Number of bad channel detection iterations")
    prep_group.add_argument("--average-reref", action="store_true", default=False, help="Average re-reference before bad channel detection")
    prep_group.add_argument("--file-extension", type=str, default=".vhdr", help="File extension for EEG data")
    prep_group.add_argument("--consider-previous-bads", action="store_true", default=False, help="Keep previously marked bad channels")
    prep_group.add_argument("--overwrite-channels-tsv", action="store_true", default=True, help="Overwrite channels.tsv file with detected bad channels")
    prep_group.add_argument("--no-overwrite-channels-tsv", dest="overwrite_channels_tsv", action="store_false", help="Do not overwrite channels.tsv file")
    prep_group.add_argument("--delete-breaks", action="store_true", default=False, help="Delete breaks in data during bad channel detection")
    prep_group.add_argument("--breaks-min-length", type=int, default=20, help="Minimum break duration in seconds")

    # Event timing
    prep_group.add_argument("--t-start-after-previous", type=int, default=2, help="Time after previous event in seconds")
    prep_group.add_argument("--t-stop-before-next", type=int, default=2, help="Time before next event in seconds")

    # Epochs
    prep_group.add_argument("--conditions", type=str, help="Epoching conditions (comma-separated)")
    prep_group.add_argument("--tmin", type=float, help="Epoch start time (s)")
    prep_group.add_argument("--tmax", type=float, help="Epoch end time (s)")
    prep_group.add_argument("--baseline", nargs=2, type=float, metavar=("START", "END"), help="Epoch baseline window (start end) in seconds, e.g., -0.2 0")
    prep_group.add_argument("--no-baseline", action="store_true", help="Disable epoch baseline correction")
    prep_group.add_argument("--reject", type=float, help="Peak-to-peak amplitude rejection threshold (µV)")
    prep_group.add_argument("--reject-method", choices=["none", "autoreject_local", "autoreject_global"], help="Epoch rejection method")
    prep_group.add_argument("--find-breaks", action="store_true", default=False, help="Find breaks in data")
    prep_group.add_argument("--run-source-estimation", action="store_true", default=False, help="Run source estimation")

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
    
    if args.montage:
        config.setdefault("eeg", {})["montage"] = args.montage
    if args.ch_types:
        config.setdefault("eeg", {})["ch_types"] = args.ch_types
    if args.eeg_reference:
        config.setdefault("eeg", {})["reference"] = args.eeg_reference
    if args.eog_channels:
        config.setdefault("eeg", {})["eog_channels"] = args.eog_channels
    if args.random_state is not None:
        preprocessing_config["random_state"] = args.random_state
    if args.task_is_rest:
        preprocessing_config["task_is_rest"] = True
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
    if args.zapline_fline is not None:
        preprocessing_config["zapline_fline"] = args.zapline_fline
    if args.find_breaks:
        preprocessing_config["find_breaks"] = True
    if args.run_source_estimation:
        preprocessing_config["run_source_estimation"] = True


def _update_ica_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with ICA parameter overrides."""
    ica_config = config.setdefault("ica", {})
    
    if args.spatial_filter:
        ica_config["spatial_filter"] = args.spatial_filter
    if args.ica_method:
        ica_config["method"] = args.ica_method
    if args.ica_components:
        ica_config["n_components"] = args.ica_components
    if args.ica_l_freq is not None:
        ica_config["l_freq"] = args.ica_l_freq
    if args.ica_reject is not None:
        ica_config["reject"] = args.ica_reject
    if args.prob_threshold:
        ica_config["probability_threshold"] = args.prob_threshold
    if args.ica_labels_to_keep:
        ica_config["labels_to_keep"] = args.ica_labels_to_keep


def _update_epochs_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with epoch parameter overrides."""
    epochs_config = config.setdefault("epochs", {})
    
    if args.conditions:
        epochs_config["conditions"] = args.conditions.split(",")
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
    if args.reject_method:
        epochs_config["reject_method"] = args.reject_method


def _resolve_n_jobs(args: argparse.Namespace, config: Any) -> int:
    """Resolve number of parallel jobs from args or config."""
    if args.n_jobs is not None:
        return args.n_jobs
    return config.get("preprocessing.n_jobs", 1)


def _update_pyprep_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with PyPREP parameter overrides."""
    pyprep_config = config.setdefault("pyprep", {})
    
    if args.ransac:
        pyprep_config["ransac"] = True
    if args.repeats != 3:
        pyprep_config["repeats"] = args.repeats
    if args.average_reref:
        pyprep_config["average_reref"] = True
    if args.file_extension != ".vhdr":
        pyprep_config["file_extension"] = args.file_extension
    if args.consider_previous_bads:
        pyprep_config["consider_previous_bads"] = True
    if not args.overwrite_channels_tsv:
        pyprep_config["overwrite_chans_tsv"] = False
    if args.delete_breaks:
        pyprep_config["delete_breaks"] = True
    if args.breaks_min_length != 20:
        pyprep_config["breaks_min_length"] = args.breaks_min_length
    if args.t_start_after_previous != 2:
        pyprep_config["t_start_after_previous"] = args.t_start_after_previous
    if args.t_stop_before_next != 2:
        pyprep_config["t_stop_before_next"] = args.t_stop_before_next


def _update_icalabel_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with ICALabel parameter overrides."""
    icalabel_config = config.setdefault("icalabel", {})
    
    if args.prob_threshold:
        icalabel_config["prob_threshold"] = args.prob_threshold
    if args.ica_labels_to_keep:
        icalabel_config["labels_to_keep"] = args.ica_labels_to_keep
    if args.keep_mnebids_bads:
        icalabel_config["keep_mnebids_bads"] = True


def run_preprocessing(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the preprocessing command."""
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
    
    _validate_epoch_parameters(args)
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)

    _update_path_config(args, config)
    _update_preprocessing_config(args, config)
    _update_pyprep_config(args, config)
    _update_ica_config(args, config)
    _update_icalabel_config(args, config)
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
