"""
Preprocessing Data Utilities
============================

Helper functions for preprocessing operations:
- File discovery (BrainVision files)
- Run index extraction
- Behavioral data helpers
- Run combination functions
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import logging
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.utils.analysis.artifact_qc import (
    band_power_metric,
    mean_abs_correlation_metric,
    pick_channels,
    window_mask,
)
from eeg_pipeline.utils.config.loader import get_config_value

logger = logging.getLogger(__name__)


###################################################################
# File Discovery
###################################################################


def find_brainvision_vhdrs(source_root: Path) -> List[Path]:
    vhdrs = sorted(source_root.glob("sub-*/eeg/brainvision_processed_1khz/**/*.vhdr"))
    return [p for p in vhdrs if p.is_file() and not p.name.startswith("._")]


def parse_subject_id(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    if not m:
        raise ValueError(f"Could not parse subject from path: {path}")
    return m.group(1)


###################################################################
# Run Index Extraction
###################################################################


def extract_run_number(path: Path) -> Optional[int]:
    match = re.search(r"run[-_]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def get_run_index(path: Path) -> Optional[int]:
    run_index = extract_run_number(path)
    if run_index is not None:
        return run_index

    all_runs = sorted(path.parent.glob("*.vhdr"))
    if len(all_runs) <= 1:
        return None

    inferred_run = all_runs.index(path) + 1
    logger.warning(
        "No explicit run found in filename '%s'. "
        "Inferring run=%d by alphabetical order among %d files. "
        "Prefer 'run-01' style filenames to guarantee correct run IDs.",
        path.name,
        inferred_run,
        len(all_runs),
    )
    return inferred_run


###################################################################
# Behavioral Data Helpers
###################################################################


def normalize_string(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_event_filters(filters: Optional[List[str]]) -> Optional[List[str]]:
    if filters in (None, [], [None]):
        return None
    normalized = [normalize_string(f) for f in filters if str(f).strip() != ""]
    return normalized if normalized else None


def find_behavior_csv_for_run(
    source_sub_dir: Path,
    run: Optional[int] = None,
    *,
    behavior_dir_name: str = "behavior",
    glob_pattern: str = "*.csv",
) -> Optional[Path]:
    behavior_dir = source_sub_dir / behavior_dir_name
    if not behavior_dir.exists():
        return None

    csvs: List[Path] = sorted(
        p for p in behavior_dir.glob(glob_pattern) if p.is_file() and not p.name.startswith("._")
    )
    if not csvs:
        return None
    if run is None:
        return csvs[0]

    candidates: List[Path] = []
    pat = re.compile(rf"run-?{run}(?:[^0-9]|$)", flags=re.IGNORECASE)
    for c in csvs:
        if pat.search(c.name):
            candidates.append(c)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def create_event_mask(
    normalized_trial_types: pd.Series,
    prefixes: Optional[List[str]],
    types: Optional[List[str]],
) -> pd.Series:
    mask = pd.Series(False, index=normalized_trial_types.index)
    if prefixes:
        for prefix in prefixes:
            mask = mask | normalized_trial_types.str.startswith(prefix)
    if types:
        mask = mask | normalized_trial_types.isin(types)
    return mask


###################################################################
# Run Combination Functions
###################################################################


def load_run_files(run_files: List[Path]) -> List[Tuple[int, pd.DataFrame, Path]]:
    frames = []
    for file_path in run_files:
        run_number = extract_run_number(file_path)
        if run_number is None:
            continue
        try:
            dataframe = read_tsv(file_path)
        except (pd.errors.ParserError, OSError) as e:
            logger.warning("Skipping run file due to read error: %s -> %s", file_path, e)
            continue
        if "onset" in dataframe.columns:
            dataframe = dataframe.sort_values("onset", kind="mergesort")
        frames.append((run_number, dataframe, file_path))
    return frames


def get_union_columns(frames: List[Tuple[int, pd.DataFrame, Path]]) -> List[str]:
    union_columns = []
    for _, dataframe, _ in frames:
        for column in dataframe.columns:
            if column not in union_columns:
                union_columns.append(column)
    return union_columns


def add_run_id_column(dataframe: pd.DataFrame, run_number: int) -> None:
    if "run_id" not in dataframe.columns and "run" not in dataframe.columns:
        dataframe.insert(0, "run_id", run_number)
    elif "run_id" in dataframe.columns and dataframe["run_id"].isna().any():
        dataframe["run_id"] = dataframe["run_id"].fillna(run_number)
    elif "run" in dataframe.columns and "run_id" not in dataframe.columns:
        if dataframe["run"].isna().any():
            dataframe["run"] = dataframe["run"].fillna(run_number)


def update_sample_indices(dataframe: pd.DataFrame, cumulative_offset: int) -> int:
    if "sample" not in dataframe.columns:
        return cumulative_offset

    sample_numeric = pd.to_numeric(dataframe["sample"], errors="coerce")
    if not sample_numeric.notna().any():
        return cumulative_offset

    if cumulative_offset > 0:
        dataframe["sample"] = sample_numeric + cumulative_offset

    max_sample = int((sample_numeric + cumulative_offset).max())
    return max_sample + 1


def get_sort_columns(combined_df: pd.DataFrame) -> List[str]:
    if "onset" in combined_df.columns:
        if "run_id" in combined_df.columns:
            return ["run_id", "onset"]
        if "run" in combined_df.columns:
            return ["run", "onset"]
        return ["onset"]

    if "run_id" in combined_df.columns:
        return ["run_id"]
    if "run" in combined_df.columns:
        return ["run"]
    return []


def combine_runs_for_subject(sub_eeg_dir: Path, task: str) -> Optional[Path]:
    run_files = sorted(
        p
        for p in sub_eeg_dir.glob(f"*_task-{task}_run-*_events.tsv")
        if not p.name.startswith("._")
    )
    if not run_files:
        return None

    frames = load_run_files(run_files)
    if not frames:
        return None

    frames.sort(key=lambda t: t[0])
    n_runs = len({r for r, _, _ in frames})
    union_columns = get_union_columns(frames)

    dataframes = []
    cumulative_sample_offset = 0

    for run_number, dataframe, _ in frames:
        for column in union_columns:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA
        dataframe = dataframe[union_columns]

        add_run_id_column(dataframe, run_number)
        cumulative_sample_offset = update_sample_indices(dataframe, cumulative_sample_offset)

        dataframes.append(dataframe)

    combined = pd.concat(dataframes, axis=0, ignore_index=True)
    sort_columns = get_sort_columns(combined)
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    sub_prefix = sub_eeg_dir.parent.name
    out_path = sub_eeg_dir / f"{sub_prefix}_task-{task}_events.tsv"

    try:
        combined.to_csv(out_path, sep="\t", index=False)
        logger.info(
            "Wrote combined events (%d run(s), %d rows): %s", n_runs, len(combined), out_path
        )
        return out_path
    except OSError as e:
        logger.error("Failed writing combined events for %s: %s", sub_prefix, e)
        return None


###################################################################
# Raw-to-BIDS Helpers
###################################################################




def filter_annotations(
    raw: mne.io.BaseRaw,
    event_prefixes: Optional[List[str]],
    keep_all: bool,
    zero_base: bool,
) -> None:
    if len(raw.annotations) == 0:
        return

    if keep_all:
        if not zero_base:
            return
        onsets = [float(o) for o in raw.annotations.onset]
        if not onsets:
            return
        base = min(onsets)
        if base == 0.0:
            return
        shifted = mne.Annotations(
            onset=[float(o) - base for o in raw.annotations.onset],
            duration=list(raw.annotations.duration),
            description=list(raw.annotations.description),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(shifted)
        return

    if event_prefixes is None:
        # Default: keep both task triggers and fMRI volume triggers for
        # simultaneous EEG-fMRI alignment/QC.
        normalized_prefixes = ["Trig_", "Volume", "Pulse Artifact"]
    else:
        normalized_prefixes = [normalize_string(p) for p in event_prefixes if str(p).strip() != ""]
        normalized_prefixes.append("Pulse Artifact")

    keep_indices = [
        idx
        for idx, description in enumerate(raw.annotations.description)
        if any(normalize_string(description).startswith(prefix) for prefix in normalized_prefixes)
    ]

    if not keep_indices:
        logger.warning(
            "No annotations matched provided prefixes. "
            "Prefixes=%s. Found %d annotations but will drop all, resulting in no events.tsv. "
            "Use --keep_all_annotations or adjust --event_prefix to keep the desired events.",
            normalized_prefixes,
            len(raw.annotations),
        )
        raw.set_annotations(mne.Annotations([], [], [], orig_time=raw.annotations.orig_time))
        return

    new_onsets = [raw.annotations.onset[idx] for idx in keep_indices]
    new_durations = [raw.annotations.duration[idx] for idx in keep_indices]
    new_descriptions = [raw.annotations.description[idx] for idx in keep_indices]

    if zero_base and new_onsets:
        base = float(min(float(o) for o in new_onsets))
        if base != 0.0:
            new_onsets = [float(onset) - base for onset in new_onsets]

    filtered_annotations = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(filtered_annotations)


def set_channel_types(raw: mne.io.BaseRaw) -> None:
    non_eeg_channel_types = {"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"}
    present_channel_types = {
        name: channel_type
        for name, channel_type in non_eeg_channel_types.items()
        if name in raw.ch_names
    }
    if present_channel_types:
        raw.set_channel_types(present_channel_types, on_unit_change="ignore")


def set_montage(raw: mne.io.BaseRaw, montage_name: str) -> None:
    montage = mne.channels.make_standard_montage(montage_name)
    if "FPz" in raw.ch_names and "Fpz" not in raw.ch_names:
        raw.rename_channels({"FPz": "Fpz"})
    raw.set_montage(montage, on_missing="warn")


def ensure_dataset_description(bids_root: Path, name: str = "EEG BIDS dataset") -> None:
    from mne_bids import make_dataset_description

    bids_root.mkdir(parents=True, exist_ok=True)
    dataset_description = bids_root / "dataset_description.json"
    if dataset_description.exists():
        return
    make_dataset_description(
        path=bids_root,
        name=name,
        dataset_type="raw",
        overwrite=False,
    )


###################################################################
# Clean (Post-Rejection) Events TSV
###################################################################


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _require_sequence(value: Any, *, path: str) -> Tuple[Any, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list.")
    return tuple(value)


def _require_float_pair(
    value: Any,
    *,
    path: str,
    default: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    raw = default if value is None else value
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{path} must be a 2-item list [start, end].")
    start = float(raw[0])
    end = float(raw[1])
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError(f"{path} values must be finite.")
    if not start < end:
        raise ValueError(f"{path} must satisfy start < end.")
    return start, end


@dataclass(frozen=True)
class ECGCouplingQCConfig:
    enabled: bool
    output_column: str
    channels: Tuple[str, ...]
    window: Tuple[float, float]


@dataclass(frozen=True)
class PeripheralLowGammaQCConfig:
    enabled: bool
    output_column: str
    channels: Tuple[str, ...]
    band: Tuple[float, float]
    window: Tuple[float, float]


@dataclass(frozen=True)
class CleanEventsQCConfig:
    enabled: bool
    ecg_coupling: ECGCouplingQCConfig
    peripheral_low_gamma: PeripheralLowGammaQCConfig

    @classmethod
    def from_config(cls, config: Any) -> "CleanEventsQCConfig":
        raw = _require_mapping(
            get_config_value(config, "preprocessing.clean_events_qc", {}),
            path="preprocessing.clean_events_qc",
        )
        default_active_window = tuple(
            float(v)
            for v in _require_float_pair(
                get_config_value(config, "time_windows.active", [3.0, 10.5]),
                path="time_windows.active",
            )
        )
        ecg_raw = _require_mapping(
            raw.get("ecg_coupling", {}),
            path="preprocessing.clean_events_qc.ecg_coupling",
        )
        peripheral_raw = _require_mapping(
            raw.get("peripheral_low_gamma", {}),
            path="preprocessing.clean_events_qc.peripheral_low_gamma",
        )
        # The ECG coupling metric correlates each EEG channel against a recorded ECG
        # lead. What it needs is the lead. Gated on the scanner declaration until now,
        # which switched it off for anyone recording ECG outside a bore -- issue #14,
        # fixed in config coherence and missed here. Left on with no channel named, it
        # fails at ``pick_channels`` after PyPREP, ICA and epoching have already run,
        # so the guard stays; only its question changes.
        ecg_coupling_enabled = bool(ecg_raw.get("enabled", True)) and bool(
            get_config_value(config, "eeg.ecg_channels", None)
        )

        cfg = cls(
            enabled=bool(raw.get("enabled", True)),
            ecg_coupling=ECGCouplingQCConfig(
                enabled=ecg_coupling_enabled,
                output_column=str(ecg_raw.get("output_column", "residual_ecg_coupling")).strip(),
                channels=tuple(
                    str(value).strip()
                    for value in _require_sequence(
                        ecg_raw.get("channels", ["ECG"]),
                        path="preprocessing.clean_events_qc.ecg_coupling.channels",
                    )
                    if str(value).strip()
                ),
                window=_require_float_pair(
                    ecg_raw.get("window"),
                    path="preprocessing.clean_events_qc.ecg_coupling.window",
                    default=default_active_window,
                ),
            ),
            peripheral_low_gamma=PeripheralLowGammaQCConfig(
                enabled=bool(peripheral_raw.get("enabled", True)),
                output_column=str(
                    peripheral_raw.get("output_column", "peripheral_low_gamma_power")
                ).strip(),
                channels=tuple(
                    str(value).strip()
                    for value in _require_sequence(
                        peripheral_raw.get(
                            "channels",
                            ["Fp1", "Fp2", "FT9", "FT10", "TP9", "TP10"],
                        ),
                        path="preprocessing.clean_events_qc.peripheral_low_gamma.channels",
                    )
                    if str(value).strip()
                ),
                band=_require_float_pair(
                    peripheral_raw.get("band"),
                    path="preprocessing.clean_events_qc.peripheral_low_gamma.band",
                    default=(30.0, 45.0),
                ),
                window=_require_float_pair(
                    peripheral_raw.get("window"),
                    path="preprocessing.clean_events_qc.peripheral_low_gamma.window",
                    default=default_active_window,
                ),
            ),
        )
        if not cfg.enabled:
            return cfg
        if cfg.ecg_coupling.enabled:
            if not cfg.ecg_coupling.output_column:
                raise ValueError(
                    "preprocessing.clean_events_qc.ecg_coupling.output_column must not be blank."
                )
            if not cfg.ecg_coupling.channels:
                raise ValueError(
                    "preprocessing.clean_events_qc.ecg_coupling.channels must not be empty when enabled."
                )
        if cfg.peripheral_low_gamma.enabled:
            if not cfg.peripheral_low_gamma.output_column:
                raise ValueError(
                    "preprocessing.clean_events_qc.peripheral_low_gamma.output_column must not be blank."
                )
            if not cfg.peripheral_low_gamma.channels:
                raise ValueError(
                    "preprocessing.clean_events_qc.peripheral_low_gamma.channels must not be empty when enabled."
                )
        if not cfg.ecg_coupling.enabled and not cfg.peripheral_low_gamma.enabled:
            # Asking for QC and naming no metric is a config mistake worth raising on —
            # unless the only metric requested was the cardiac one and this dataset has
            # no ECG lead named, in which case there is nothing left to compute and
            # nothing the user got wrong.
            if not get_config_value(config, "eeg.ecg_channels", None) and bool(
                ecg_raw.get("enabled", True)
            ):
                return replace(cfg, enabled=False)
            raise ValueError(
                "preprocessing.clean_events_qc.enabled=true requires at least one QC metric."
            )
        return cfg


def _ecg_coupling_metric(
    *,
    epochs: mne.BaseEpochs,
    channels: Sequence[str],
    mask: np.ndarray,
) -> np.ndarray:
    picks = pick_channels(
        epochs.info,
        channels,
        path="preprocessing.clean_events_qc.ecg_coupling.channels",
    )
    ecg_data = np.asarray(epochs.get_data(picks=picks), dtype=float)
    eeg_data = np.asarray(epochs.get_data(picks="eeg"), dtype=float)
    if eeg_data.ndim != 3 or eeg_data.shape[1] == 0:
        raise ValueError("ECG coupling QC requires EEG channels.")
    return mean_abs_correlation_metric(ecg_data[:, :, mask], eeg_data[:, :, mask])


def _peripheral_low_gamma_metric(
    *,
    epochs: mne.BaseEpochs,
    channels: Sequence[str],
    band: Tuple[float, float],
    mask: np.ndarray,
) -> np.ndarray:
    return band_power_metric(
        epochs=epochs,
        channels=channels,
        band=band,
        mask=mask,
        path="preprocessing.clean_events_qc.peripheral_low_gamma.channels",
    )


def _compute_clean_events_qc_table(
    *,
    epochs: mne.BaseEpochs,
    qc_cfg: CleanEventsQCConfig,
) -> pd.DataFrame:
    if not qc_cfg.enabled:
        return pd.DataFrame(index=np.arange(len(epochs), dtype=int))
    if not epochs.preload:
        epochs.load_data()

    times = np.asarray(epochs.times, dtype=float)
    out = pd.DataFrame(index=np.arange(len(epochs), dtype=int))

    if qc_cfg.ecg_coupling.enabled:
        ecg_mask = window_mask(
            times,
            qc_cfg.ecg_coupling.window,
            path="preprocessing.clean_events_qc.ecg_coupling.window",
        )
        out[qc_cfg.ecg_coupling.output_column] = _ecg_coupling_metric(
            epochs=epochs,
            channels=qc_cfg.ecg_coupling.channels,
            mask=ecg_mask,
        )

    if qc_cfg.peripheral_low_gamma.enabled:
        peripheral_mask = window_mask(
            times,
            qc_cfg.peripheral_low_gamma.window,
            path="preprocessing.clean_events_qc.peripheral_low_gamma.window",
        )
        out[qc_cfg.peripheral_low_gamma.output_column] = _peripheral_low_gamma_metric(
            epochs=epochs,
            channels=qc_cfg.peripheral_low_gamma.channels,
            band=qc_cfg.peripheral_low_gamma.band,
            mask=peripheral_mask,
        )

    if out.shape[0] != len(epochs):
        raise ValueError("Clean-events QC table length must match the number of epochs.")
    return out.reset_index(drop=True)


def _matches_condition(trial_type: str, condition: str) -> bool:
    """Apply MNE's condition matching to one trial type.

    MNE treats ``/`` as a tag separator, so the condition ``pain`` selects the trial type
    ``pain/high``. It is not a plain prefix match: ``pain`` must not select ``painless``,
    which would silently pull extra rows into the events table and misalign it against the
    epochs MNE actually built.
    """
    if trial_type == condition:
        return True
    return condition in trial_type.split("/")


def _build_epoch_event_mask(
    events_df: pd.DataFrame,
    conditions: List[str],
) -> tuple[pd.Series, str]:
    condition_column = _resolve_epoch_condition_column(events_df)
    trial_type_norm = events_df[condition_column].astype(str).map(normalize_string)
    cond_norm = [normalize_string(c) for c in conditions if str(c).strip() != ""]

    mask = pd.Series(False, index=events_df.index)
    for cond in cond_norm:
        mask = mask | trial_type_norm.map(lambda value, cond=cond: _matches_condition(value, cond))
    return mask, condition_column


def _kept_event_mask(epochs: mne.BaseEpochs, target_count: int) -> np.ndarray:
    """Return which of the condition-matching events survived to the clean epochs.

    ``Epochs.drop_log`` has one entry per event MNE was originally given — every
    annotation on the concatenated raw, not just the ones being epoched. Entries tagged
    ``IGNORED`` are the events whose trial type was not among the requested conditions, so
    dropping those leaves exactly the condition-matching events, in order, with an empty
    tuple marking each one that was kept.

    This is why the mapping is taken from ``drop_log`` rather than ``Epochs.selection``:
    ``selection`` indexes the full event array, which for this dataset is dominated by
    scanner-volume and pulse markers, so using it to index the condition-filtered events
    table lines the two up only by coincidence.
    """
    drop_log = epochs.drop_log
    considered = [entry for entry in drop_log if "IGNORED" not in entry]
    if len(considered) != target_count:
        raise ValueError(
            f"Epoch drop log describes {len(considered)} condition events but the events "
            f"table has {target_count}. The conditions used for epoching and the ones "
            "used here do not select the same events."
        )
    return np.array([len(entry) == 0 for entry in considered], dtype=bool)


def _autoreject_counts_for_clean_events(
    *,
    epochs_path: Path,
    config: Any,
    n_epochs: int,
) -> Optional[pd.DataFrame]:
    """Per-trial AutoReject repair counts to carry into the clean events table.

    Returns ``None`` when the log is not requested. When it is, a missing or
    disagreeing log is an error: a clean events table that silently omits which
    channels were reconstructed is worse than one that fails to be written.
    """
    from eeg_pipeline.preprocessing.autoreject_log import (
        autoreject_log_path_for_epochs,
        kept_epoch_counts,
        read_autoreject_log,
    )

    if not bool(get_config_value(config, "preprocessing.autoreject_log", False)):
        return None

    log_path = autoreject_log_path_for_epochs(epochs_path)
    if not log_path.exists():
        raise FileNotFoundError(
            f"preprocessing.autoreject_log is enabled but {log_path} is missing. "
            "Write the AutoReject log before writing clean events."
        )

    counts = kept_epoch_counts(read_autoreject_log(log_path))
    if len(counts) != n_epochs:
        raise ValueError(
            f"AutoReject log keeps {len(counts)} epochs but the clean epochs file holds "
            f"{n_epochs}. The log does not describe this derivative."
        )
    return counts


def _resolve_epoch_condition_column(events_df: pd.DataFrame) -> str:
    column_lookup = {str(col).strip().lower(): str(col) for col in events_df.columns}
    resolved = column_lookup.get("trial_type")
    if resolved is not None:
        return resolved

    raise ValueError(
        "events.tsv is missing BIDS trial_type, which is required for clean-events "
        f"epoch alignment. Available columns: {list(events_df.columns)}"
    )


def _sort_events_for_epoch_alignment(events_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    if "run_id" in events_df.columns:
        sort_cols.append("run_id")
    if "onset" in events_df.columns:
        sort_cols.append("onset")
    if "sample" in events_df.columns:
        sort_cols.append("sample")
    if not sort_cols:
        return events_df.reset_index(drop=True)
    return events_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _load_run_level_events_for_epochs(
    run_files: list[Path],
    subject_label: str,
    task: str,
) -> pd.DataFrame:
    if not run_files:
        raise ValueError("run_files must contain at least one run-level events file.")

    frames = load_run_files(run_files)
    if len(frames) != len(run_files):
        raise ValueError(
            f"Could not read all run-level events files for {subject_label}, task-{task}: "
            f"read={len(frames)}, expected={len(run_files)}"
        )

    frames.sort(key=lambda t: t[0])
    union_columns = get_union_columns(frames)

    parts: list[pd.DataFrame] = []
    for run_number, df, _path in frames:
        for col in union_columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[union_columns].copy()
        add_run_id_column(df, run_number)
        parts.append(df)

    out = pd.concat(parts, axis=0, ignore_index=True)
    return _sort_events_for_epoch_alignment(out)


def _load_subject_events_for_epochs(
    bids_sub_eeg_dir: Path, subject_label: str, task: str
) -> pd.DataFrame:
    """Load the events table that matches the epoch source files."""
    run_files = sorted(bids_sub_eeg_dir.glob(f"{subject_label}_task-{task}_run-*_events.tsv"))
    if run_files:
        return _load_run_level_events_for_epochs(run_files, subject_label, task)

    combined = bids_sub_eeg_dir / f"{subject_label}_task-{task}_events.tsv"
    if combined.exists():
        return _sort_events_for_epoch_alignment(read_tsv(combined))

    raise FileNotFoundError(
        f"No events.tsv found for {subject_label}, task-{task} under {bids_sub_eeg_dir}"
    )


def _derive_clean_events_path_from_epochs(epochs_path: Path) -> Path:
    name = epochs_path.name
    if name.endswith("_proc-clean_epo.fif"):
        return epochs_path.with_name(name.replace("_proc-clean_epo.fif", "_proc-clean_events.tsv"))
    if name.endswith("_proc-cleaned_epo.fif"):
        return epochs_path.with_name(
            name.replace("_proc-cleaned_epo.fif", "_proc-cleaned_events.tsv")
        )
    if name.endswith("_clean_epo.fif"):
        return epochs_path.with_name(name.replace("_clean_epo.fif", "_clean_events.tsv"))
    if name.endswith("_epo.fif"):
        return epochs_path.with_name(name.replace("_epo.fif", "_events.tsv"))
    return epochs_path.with_suffix(".tsv")


def presented_events_for_epochs(
    *,
    subject: str,
    task: str,
    bids_root: Path,
    epochs: mne.BaseEpochs,
    conditions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return the BIDS event rows that entered the epoch-rejection denominator.

    This uses the same condition semantics and run ordering as the clean-events writer,
    but retains dropped rows. It is therefore the denominator required for per-condition
    and per-run retention rates in the report.
    """
    subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    bids_sub_eeg_dir = Path(bids_root) / subject_label / "eeg"
    if not bids_sub_eeg_dir.exists():
        raise FileNotFoundError(f"Missing BIDS EEG directory: {bids_sub_eeg_dir}")

    resolved_conditions = conditions
    if resolved_conditions is None:
        resolved_conditions = list(getattr(epochs, "event_id", {}).keys())
    if not resolved_conditions:
        raise ValueError(
            f"No epoching conditions provided and epochs.event_id is empty for "
            f"{subject_label}, task-{task}"
        )

    events = _load_subject_events_for_epochs(bids_sub_eeg_dir, subject_label, task)
    mask, condition_column = _build_epoch_event_mask(events, resolved_conditions)
    presented = events.loc[mask].copy().reset_index(drop=True)
    if presented.empty:
        available = sorted(events[condition_column].dropna().astype(str).unique())
        raise ValueError(
            f"No events matched conditions={resolved_conditions} in {subject_label}, "
            f"task-{task}. Available {condition_column} values: {available}"
        )

    # Validates that the BIDS selection is the same population represented by the MNE
    # drop log. The mask itself is intentionally not applied: rejected rows are the
    # denominators this function exists to retain.
    _kept_event_mask(epochs, len(presented))
    presented.insert(0, "event_index", range(len(presented)))
    return presented


def write_clean_events_tsv_for_epochs(
    *,
    subject: str,
    task: str,
    bids_root: Path,
    epochs_path: Path,
    config: Any,
    conditions: Optional[List[str]] = None,
    overwrite: bool = True,
    after_rejection: bool = True,
    _logger: Optional[logging.Logger] = None,
) -> Path:
    """Write an epoch-aligned events.tsv for a set of epochs.

    Output is written next to the epochs file (derivatives), using the same naming stem
    (e.g., ``*_proc-clean_events.tsv``).

    ``after_rejection`` says whether these epochs have been through the rejection step.
    It is not a switch for how much detail to include: AutoReject is *fitted* in that
    step, so before it has run there is no per-trial repair record to attach, and
    demanding one asks for a measurement that does not exist yet. The provisional
    band-ICA comparisons run at ICA-fitting time against pre-rejection epochs and pass
    ``False`` for exactly that reason.

    The default stays ``True`` so the post-rejection callers keep failing loudly when the
    AutoReject log is genuinely missing — that log is the only record of which
    channel-in-trial samples are spline estimates rather than measurements, and a clean
    events table that silently omits it is worse than one that is not written.
    """
    from eeg_pipeline.analysis.utilities.bids_metadata import ensure_events_sidecar

    log = _logger or logger

    subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    bids_sub_eeg_dir = bids_root / subject_label / "eeg"
    if not bids_sub_eeg_dir.exists():
        raise FileNotFoundError(f"Missing BIDS EEG directory: {bids_sub_eeg_dir}")
    if not epochs_path.exists():
        raise FileNotFoundError(f"Missing clean epochs file: {epochs_path}")

    out_path = _derive_clean_events_path_from_epochs(epochs_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    epochs = mne.read_epochs(epochs_path, preload=False, verbose=False)
    qc_cfg = CleanEventsQCConfig.from_config(config)

    target = presented_events_for_epochs(
        subject=subject,
        task=task,
        bids_root=bids_root,
        epochs=epochs,
        conditions=conditions,
    )

    n_epochs = len(epochs)
    if n_epochs == 0:
        kept = target.iloc[0:0].copy().reset_index(drop=True)
        kept.insert(0, "trial_id", [])
        kept.insert(0, "epoch_index", [])
        kept.to_csv(out_path, sep="\t", index=False)
        ensure_events_sidecar(out_path, list(kept.columns))
        log.warning("All epochs were rejected; wrote empty clean events: %s", out_path)
        return out_path

    try:
        kept_mask = _kept_event_mask(epochs, len(target))
    except ValueError as exc:
        raise ValueError(
            f"Cannot map kept epochs to events for {subject_label}, task-{task}: {exc}"
        ) from exc
    kept = target.loc[kept_mask].copy().reset_index(drop=True)
    if len(kept) != n_epochs:
        raise ValueError(
            f"Cannot map kept epochs to events for {subject_label}, task-{task}: the drop "
            f"log marks {len(kept)} events as kept but the file holds {n_epochs} epochs."
        )

    kept.insert(0, "trial_id", range(1, len(kept) + 1))
    kept.insert(0, "epoch_index", range(len(kept)))
    qc_table = _compute_clean_events_qc_table(epochs=epochs, qc_cfg=qc_cfg)
    if len(qc_table) != len(kept):
        raise ValueError(
            f"QC table length mismatch for {subject_label}, task-{task}: {len(qc_table)} vs {len(kept)}."
        )
    kept = pd.concat([kept.reset_index(drop=True), qc_table], axis=1)

    autoreject_counts = (
        _autoreject_counts_for_clean_events(
            epochs_path=epochs_path,
            config=config,
            n_epochs=n_epochs,
        )
        if after_rejection
        else None
    )
    if autoreject_counts is not None:
        kept = pd.concat([kept.reset_index(drop=True), autoreject_counts], axis=1)

    kept.to_csv(out_path, sep="\t", index=False)
    ensure_events_sidecar(out_path, list(kept.columns))
    log.info("Wrote clean events (n=%d): %s", len(kept), out_path)
    return out_path


__all__ = [
    "find_brainvision_vhdrs",
    "parse_subject_id",
    "presented_events_for_epochs",
    "extract_run_number",
    "get_run_index",
    "normalize_string",
    "normalize_event_filters",
    "find_behavior_csv_for_run",
    "create_event_mask",
    "load_run_files",
    "get_union_columns",
    "add_run_id_column",
    "update_sample_indices",
    "get_sort_columns",
    "combine_runs_for_subject",
    "filter_annotations",
    "set_channel_types",
    "set_montage",
    "ensure_dataset_description",
    "write_clean_events_tsv_for_epochs",
]
