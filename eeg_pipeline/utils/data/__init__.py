from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "align_feature_dataframes",
    "combine_runs_for_subject",
    "compute_aligned_data_length",
    "ensure_dataset_description",
    "extract_run_number",
    "extract_predictor_data",
    "find_brainvision_vhdrs",
    "find_column",
    "get_aligned_events",
    "get_available_subjects",
    "get_run_index",
    "load_behavior_stats_files",
    "load_epochs_for_analysis",
    "load_feature_bundle",
    "load_stats_file",
    "load_subject_scatter_data",
    "normalize_string",
    "parse_subject_args",
    "parse_subject_id",
    "save_all_features",
    "save_dropped_trials_log",
    "set_channel_types",
    "set_montage",
    "_pick_first_column",
]

_EXPORTS = {
    "align_feature_dataframes": ("eeg_pipeline.utils.data.features", "align_feature_dataframes"),
    "combine_runs_for_subject": (
        "eeg_pipeline.utils.data.preprocessing",
        "combine_runs_for_subject",
    ),
    "compute_aligned_data_length": (
        "eeg_pipeline.utils.data.tfr_alignment",
        "compute_aligned_data_length",
    ),
    "ensure_dataset_description": (
        "eeg_pipeline.utils.data.preprocessing",
        "ensure_dataset_description",
    ),
    "extract_run_number": ("eeg_pipeline.utils.data.preprocessing", "extract_run_number"),
    "extract_predictor_data": ("eeg_pipeline.utils.data.covariates", "extract_predictor_data"),
    "find_brainvision_vhdrs": ("eeg_pipeline.utils.data.preprocessing", "find_brainvision_vhdrs"),
    "find_column": ("eeg_pipeline.utils.data.manipulation", "find_column"),
    "get_aligned_events": ("eeg_pipeline.utils.data.alignment", "get_aligned_events"),
    "get_available_subjects": ("eeg_pipeline.utils.data.subjects", "get_available_subjects"),
    "get_run_index": ("eeg_pipeline.utils.data.preprocessing", "get_run_index"),
    "load_behavior_stats_files": (
        "eeg_pipeline.utils.data.behavior",
        "load_behavior_stats_files",
    ),
    "load_epochs_for_analysis": ("eeg_pipeline.utils.data.epochs", "load_epochs_for_analysis"),
    "load_feature_bundle": ("eeg_pipeline.utils.data.feature_io", "load_feature_bundle"),
    "load_stats_file": ("eeg_pipeline.utils.data.behavior", "load_stats_file"),
    "load_subject_scatter_data": (
        "eeg_pipeline.utils.data.stats_io",
        "load_subject_scatter_data",
    ),
    "normalize_string": ("eeg_pipeline.utils.data.preprocessing", "normalize_string"),
    "parse_subject_args": ("eeg_pipeline.utils.data.subjects", "parse_subject_args"),
    "parse_subject_id": ("eeg_pipeline.utils.data.preprocessing", "parse_subject_id"),
    "save_all_features": ("eeg_pipeline.utils.data.feature_io", "save_all_features"),
    "save_dropped_trials_log": (
        "eeg_pipeline.utils.data.feature_io",
        "save_dropped_trials_log",
    ),
    "set_channel_types": ("eeg_pipeline.utils.data.preprocessing", "set_channel_types"),
    "set_montage": ("eeg_pipeline.utils.data.preprocessing", "set_montage"),
    "_pick_first_column": ("eeg_pipeline.utils.data.covariates", "_pick_first_column"),
}


def __getattr__(name: str) -> Any:
    try:
        module_path, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_path)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
