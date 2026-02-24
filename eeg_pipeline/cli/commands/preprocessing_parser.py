"""Parser construction for preprocessing CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
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

    # Core options
    prep_group.add_argument("--n-jobs", type=int, default=None, help="Number of parallel jobs for bad channel detection (default from config)")
    prep_group.add_argument("--montage", type=str, default=None, help="EEG montage name (e.g., easycap-M1)")
    prep_group.add_argument("--ch-types", type=str, default=None, help="Channel types to include (e.g., 'eeg')")
    prep_group.add_argument("--eeg-reference", type=str, default=None, help="EEG reference type (e.g., 'average')")
    prep_group.add_argument("--eog-channels", type=str, default=None, help="EOG channels (comma-separated, e.g., 'Fp1,Fp2')")
    prep_group.add_argument("--ecg-channels", type=str, default=None, help="ECG channels (comma-separated, e.g., 'ECG')")
    prep_group.add_argument("--random-state", type=int, default=None, help="Random seed for reproducibility")
    prep_group.add_argument("--task-is-rest", action="store_true", default=False, help="Whether task is resting-state")
    prep_group.add_argument("--use-icalabel", action="store_true", default=True, help="Use mne-icalabel for ICA component classification (default: True)")
    prep_group.add_argument("--no-icalabel", dest="use_icalabel", action="store_false", help="Disable mne-icalabel, use MNE-BIDS pipeline ICA detection")
    prep_group.add_argument("--use-pyprep", action="store_true", default=True, help="Use PyPREP for bad channel detection (default: True)")
    prep_group.add_argument("--no-pyprep", dest="use_pyprep", action="store_false", help="Disable PyPREP bad channel detection")

    # Clean events.tsv (post-rejection)
    prep_group.add_argument(
        "--write-clean-events",
        dest="write_clean_events",
        action="store_true",
        default=None,
        help="Write a clean events.tsv aligned to kept epochs (default from config)",
    )
    prep_group.add_argument(
        "--no-write-clean-events",
        dest="write_clean_events",
        action="store_false",
        help="Disable writing clean events.tsv aligned to kept epochs",
    )
    prep_group.add_argument(
        "--overwrite-clean-events",
        dest="overwrite_clean_events",
        action="store_true",
        default=None,
        help="Overwrite existing clean events.tsv (default from config)",
    )
    prep_group.add_argument(
        "--no-overwrite-clean-events",
        dest="overwrite_clean_events",
        action="store_false",
        help="Do not overwrite existing clean events.tsv",
    )
    prep_group.add_argument(
        "--clean-events-strict",
        dest="clean_events_strict",
        action="store_true",
        default=None,
        help="Fail preprocessing if clean events.tsv cannot be written (default from config)",
    )
    prep_group.add_argument(
        "--no-clean-events-strict",
        dest="clean_events_strict",
        action="store_false",
        help="Warn instead of failing when clean events.tsv cannot be written",
    )

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
    prep_group.add_argument(
        "--ransac",
        dest="ransac",
        action="store_true",
        default=None,
        help="Enable RANSAC for bad channel detection (default from config)",
    )
    prep_group.add_argument(
        "--no-ransac",
        dest="ransac",
        action="store_false",
        help="Disable RANSAC for bad channel detection",
    )
    prep_group.add_argument("--repeats", type=int, default=3, help="Number of bad channel detection iterations")
    prep_group.add_argument("--average-reref", action="store_true", default=False, help="Average re-reference before bad channel detection")
    prep_group.add_argument("--file-extension", type=str, default=".vhdr", help="File extension for EEG data")
    prep_group.add_argument("--rename-anot-dict", type=str, default=None, help="JSON dict for annotation renaming, e.g. '{\"BAD boundary\":\"BAD_boundary\"}'")
    prep_group.add_argument("--custom-bad-dict", type=str, default=None, help="JSON dict of custom bad channels per task/subject")
    prep_group.add_argument(
        "--consider-previous-bads",
        dest="consider_previous_bads",
        action="store_true",
        default=None,
        help="Keep previously marked bad channels (default from config)",
    )
    prep_group.add_argument(
        "--no-consider-previous-bads",
        dest="consider_previous_bads",
        action="store_false",
        help="Ignore/clear previously marked bad channels",
    )
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
    prep_group.add_argument(
        "--autoreject-n-interpolate",
        nargs="+",
        type=int,
        default=None,
        help="Autoreject interpolation candidates (e.g., 4 8 16)",
    )
    prep_group.add_argument(
        "--find-breaks",
        dest="find_breaks",
        action="store_true",
        default=None,
        help="Enable automatic break detection (default from config)",
    )
    prep_group.add_argument(
        "--no-find-breaks",
        dest="find_breaks",
        action="store_false",
        help="Disable automatic break detection",
    )
    prep_group.add_argument("--run-source-estimation", action="store_true", default=False, help="Run source estimation")
    prep_group.add_argument("--allow-misaligned-trim", action="store_true", default=False, help="Allow trimming when EEG/fMRI trial counts are slightly misaligned")
    prep_group.add_argument("--min-alignment-samples", type=int, default=None, help="Minimum aligned sample count required after trimming")
    prep_group.add_argument("--trim-to-first-volume", action="store_true", default=False, help="Trim EEG events to first fMRI volume start when aligning")
    prep_group.add_argument(
        "--fmri-onset-reference",
        choices=["as_is", "first_volume", "scanner_trigger"],
        default=None,
        help="Reference event for fMRI onset alignment",
    )
    prep_group.add_argument("--event-col-temperature", nargs="+", type=str, default=None, help="events.tsv candidate columns for temperature")
    prep_group.add_argument("--event-col-rating", nargs="+", type=str, default=None, help="events.tsv candidate columns for rating")
    prep_group.add_argument("--event-col-binary-outcome", nargs="+", type=str, default=None, help="events.tsv candidate columns for binary-outcome split")
    prep_group.add_argument(
        "--condition-preferred-prefixes",
        nargs="+",
        type=str,
        default=None,
        help="Preferred trigger prefixes for auto-detecting epoch conditions from events.tsv",
    )

    add_path_args(parser)

    return parser
