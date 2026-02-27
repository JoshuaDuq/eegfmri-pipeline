"""Config override helpers for preprocessing CLI command."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _validate_epoch_parameters(args: argparse.Namespace) -> None:
    """Validate epoch parameter constraints."""
    if args.tmin is not None and args.tmax is not None and args.tmin >= args.tmax:
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
    paths_config = config.setdefault("paths", {})
    if args.bids_root:
        paths_config["bids_root"] = args.bids_root
    if args.deriv_root:
        paths_config["deriv_root"] = args.deriv_root


def _update_preprocessing_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with preprocessing parameter overrides."""
    eeg_config = config.setdefault("eeg", {})
    preprocessing_config = config.setdefault("preprocessing", {})

    if args.montage:
        eeg_config["montage"] = args.montage
    if args.ch_types:
        eeg_config["ch_types"] = args.ch_types
    if args.eeg_reference:
        eeg_config["reference"] = args.eeg_reference
    if args.eog_channels:
        eog_channels_list = [ch.strip() for ch in args.eog_channels.split(",") if ch.strip()]
        if eog_channels_list:
            eeg_config["eog_channels"] = eog_channels_list
    if args.ecg_channels:
        ecg_channels_list = [ch.strip() for ch in args.ecg_channels.split(",") if ch.strip()]
        if ecg_channels_list:
            eeg_config["ecg_channels"] = ecg_channels_list
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
    if args.find_breaks is not None:
        preprocessing_config["find_breaks"] = bool(args.find_breaks)
    if args.write_clean_events is not None:
        preprocessing_config["write_clean_events"] = bool(args.write_clean_events)
    if args.overwrite_clean_events is not None:
        preprocessing_config["clean_events_overwrite"] = bool(args.overwrite_clean_events)
    if args.clean_events_strict is not None:
        preprocessing_config["clean_events_strict"] = bool(args.clean_events_strict)


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


def _update_epochs_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with epoch parameter overrides."""
    epochs_config = config.setdefault("epochs", {})

    if args.conditions:
        epochs_config["conditions"] = args.conditions.split(",")
    if args.tmin is not None:
        epochs_config["tmin"] = args.tmin
    if args.tmax is not None:
        epochs_config["tmax"] = args.tmax
    if args.no_baseline:
        epochs_config["baseline"] = None
    elif args.baseline:
        epochs_config["baseline"] = tuple(args.baseline)
    if args.reject is not None and args.reject > 0:
        epochs_config["reject"] = {"eeg": args.reject * 1e-6}
    elif args.reject is not None and args.reject <= 0:
        epochs_config["reject"] = None
    if args.reject_method:
        epochs_config["reject_method"] = args.reject_method
    if args.autoreject_n_interpolate:
        epochs_config["autoreject_n_interpolate"] = [int(v) for v in args.autoreject_n_interpolate]


def _update_alignment_event_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with alignment + event-column overrides."""
    alignment_cfg = config.setdefault("alignment", {})
    event_cols = config.setdefault("event_columns", {})
    preprocessing_cfg = config.setdefault("preprocessing", {})
    if args.allow_misaligned_trim:
        alignment_cfg["allow_misaligned_trim"] = True
    if args.min_alignment_samples is not None:
        alignment_cfg["min_alignment_samples"] = int(args.min_alignment_samples)
    if args.trim_to_first_volume:
        alignment_cfg["trim_to_first_volume"] = True
    if args.fmri_onset_reference:
        mapping = {
            "first_volume": "first_iti_start",
            "scanner_trigger": "first_stim_start",
        }
        alignment_cfg["fmri_onset_reference"] = mapping.get(
            args.fmri_onset_reference,
            args.fmri_onset_reference,
        )

    if args.event_col_predictor:
        event_cols["predictor"] = [str(v).strip() for v in args.event_col_predictor if str(v).strip()]
    if args.event_col_outcome:
        event_cols["outcome"] = [str(v).strip() for v in args.event_col_outcome if str(v).strip()]
    if args.event_col_binary_outcome:
        event_cols["binary_outcome"] = [str(v).strip() for v in args.event_col_binary_outcome if str(v).strip()]
    if args.event_col_required:
        event_cols["required"] = [str(v).strip() for v in args.event_col_required if str(v).strip()]
    if args.event_col_condition:
        event_cols["condition"] = [str(v).strip() for v in args.event_col_condition if str(v).strip()]
    if args.condition_preferred_prefixes:
        preprocessing_cfg["condition_preferred_prefixes"] = [
            str(v).strip() for v in args.condition_preferred_prefixes if str(v).strip()
        ]


def _resolve_n_jobs(args: argparse.Namespace, config: Any) -> int:
    """Resolve number of parallel jobs from args or config."""
    if args.n_jobs is not None:
        return args.n_jobs
    return config.get("preprocessing.n_jobs", 1)


def _update_pyprep_config(args: argparse.Namespace, config: Any) -> None:
    """Update config with PyPREP parameter overrides."""
    pyprep_config = config.setdefault("pyprep", {})

    if args.ransac is not None:
        pyprep_config["ransac"] = bool(args.ransac)
    if args.repeats != 3:
        pyprep_config["repeats"] = args.repeats
    if args.average_reref:
        pyprep_config["average_reref"] = True
    if args.file_extension != ".vhdr":
        pyprep_config["file_extension"] = args.file_extension
    if args.rename_anot_dict:
        try:
            parsed = json.loads(args.rename_anot_dict)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --rename-anot-dict: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--rename-anot-dict must decode to a JSON object")
        pyprep_config["rename_anot_dict"] = parsed
    if args.custom_bad_dict:
        try:
            parsed = json.loads(args.custom_bad_dict)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --custom-bad-dict: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--custom-bad-dict must decode to a JSON object")
        pyprep_config["custom_bad_dict"] = parsed
    if args.consider_previous_bads is not None:
        pyprep_config["consider_previous_bads"] = bool(args.consider_previous_bads)
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
