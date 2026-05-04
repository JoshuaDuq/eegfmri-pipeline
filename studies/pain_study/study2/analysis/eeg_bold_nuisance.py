"""Trial-wise nuisance and QC metrics for EEG-BOLD coupling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import compute_regressor

from eeg_pipeline.utils.analysis.artifact_qc import (
    band_power_metric,
    metric_rms,
    pick_channels,
    window_mask,
)
from eeg_pipeline.utils.config.loader import get_config_value
from fmri_pipeline.analysis.contrast_builder import discover_confounds
from fmri_pipeline.utils.bold_discovery import discover_fmriprep_preproc_bold, get_tr_from_bold


_EPS = 1.0e-12
_SUPPORTED_HRF_MODELS = {"spm", "glover"}


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
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


def _subject_raw(subject: str) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text[4:]
    return text


def _subject_bids(subject: str) -> str:
    return f"sub-{_subject_raw(subject)}"


def _robust_zscore_by_run(values: np.ndarray, run_numbers: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for run_num in np.unique(run_numbers):
        mask = run_numbers == run_num
        run_values = values[mask]
        if run_values.size == 0:
            continue
        median = float(np.median(run_values))
        mad = float(np.median(np.abs(run_values - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            out[mask] = 0.0
            continue
        out[mask] = (run_values - median) / scale
    return out


def _positive_artifact_burden(z_matrix: np.ndarray) -> np.ndarray:
    if z_matrix.ndim != 2:
        raise ValueError("Artifact burden expects a 2D z-score matrix.")
    positive = np.maximum(np.asarray(z_matrix, dtype=float), 0.0)
    return np.mean(positive, axis=1)


def _sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text).strip()).strip("_").lower()


@dataclass(frozen=True)
class FDNuisanceConfig:
    enabled: bool
    output_column: str
    source_column: str
    hrf_model: str
    censor_above: Optional[float]


@dataclass(frozen=True)
class DVARSNuisanceConfig:
    enabled: bool
    output_column: str
    source_column: str
    hrf_model: str
    censor_above: Optional[float]


@dataclass(frozen=True)
class GlobalAmplitudeConfig:
    enabled: bool
    window: Tuple[float, float]


@dataclass(frozen=True)
class ECGAmplitudeConfig:
    enabled: bool
    channels: Tuple[str, ...]
    window: Tuple[float, float]


@dataclass(frozen=True)
class PeripheralBandPowerConfig:
    enabled: bool
    channels: Tuple[str, ...]
    band: Tuple[float, float]
    window: Tuple[float, float]


@dataclass(frozen=True)
class EEGArtifactConfig:
    enabled: bool
    output_column: str
    required_components: Tuple[str, ...]
    component_z_threshold: Optional[float]
    composite_threshold: Optional[float]
    event_numeric_columns: Tuple[str, ...]
    global_amplitude: GlobalAmplitudeConfig
    ecg_amplitude: ECGAmplitudeConfig
    peripheral_band_power: PeripheralBandPowerConfig


@dataclass(frozen=True)
class CensoringConfig:
    require_all_model_terms_finite: bool


@dataclass(frozen=True)
class CouplingNuisanceConfig:
    fd: FDNuisanceConfig
    dvars: DVARSNuisanceConfig
    eeg_artifact: EEGArtifactConfig
    censoring: CensoringConfig

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        active_window: Tuple[float, float],
        epoch_window: Tuple[float, float],
        default_hrf_model: str,
    ) -> "CouplingNuisanceConfig":
        raw = _require_mapping(
            get_config_value(config, "eeg_bold_coupling.nuisance", {}),
            path="eeg_bold_coupling.nuisance",
        )

        fd_raw = _require_mapping(raw.get("fd", {}), path="eeg_bold_coupling.nuisance.fd")
        hrf_model = str(fd_raw.get("hrf_model", default_hrf_model)).strip().lower()
        if fd_raw.get("enabled", True) and hrf_model not in _SUPPORTED_HRF_MODELS:
            raise ValueError(
                "eeg_bold_coupling.nuisance.fd.hrf_model must be 'spm' or 'glover'."
            )
        fd = FDNuisanceConfig(
            enabled=bool(fd_raw.get("enabled", True)),
            output_column=str(fd_raw.get("output_column", "fd")).strip(),
            source_column=str(fd_raw.get("source_column", "framewise_displacement")).strip(),
            hrf_model=hrf_model,
            censor_above=(
                None
                if fd_raw.get("censor_above", None) in {None, ""}
                else float(fd_raw.get("censor_above"))
            ),
        )
        if fd.enabled and not fd.output_column:
            raise ValueError("eeg_bold_coupling.nuisance.fd.output_column must not be blank.")
        if fd.enabled and not fd.source_column:
            raise ValueError("eeg_bold_coupling.nuisance.fd.source_column must not be blank.")

        dvars_raw = _require_mapping(
            raw.get("dvars", {}),
            path="eeg_bold_coupling.nuisance.dvars",
        )
        dvars_hrf_model = str(
            dvars_raw.get("hrf_model", default_hrf_model)
        ).strip().lower()
        if dvars_raw.get("enabled", False) and dvars_hrf_model not in _SUPPORTED_HRF_MODELS:
            raise ValueError(
                "eeg_bold_coupling.nuisance.dvars.hrf_model must be 'spm' or 'glover'."
            )
        dvars = DVARSNuisanceConfig(
            enabled=bool(dvars_raw.get("enabled", False)),
            output_column=str(dvars_raw.get("output_column", "dvars")).strip(),
            source_column=str(dvars_raw.get("source_column", "std_dvars")).strip(),
            hrf_model=dvars_hrf_model,
            censor_above=(
                None
                if dvars_raw.get("censor_above", None) in {None, ""}
                else float(dvars_raw.get("censor_above"))
            ),
        )
        if dvars.enabled and not dvars.output_column:
            raise ValueError("eeg_bold_coupling.nuisance.dvars.output_column must not be blank.")
        if dvars.enabled and not dvars.source_column:
            raise ValueError("eeg_bold_coupling.nuisance.dvars.source_column must not be blank.")

        eeg_raw = _require_mapping(
            raw.get("eeg_artifact", {}),
            path="eeg_bold_coupling.nuisance.eeg_artifact",
        )
        global_raw = _require_mapping(
            eeg_raw.get("global_amplitude", {}),
            path="eeg_bold_coupling.nuisance.eeg_artifact.global_amplitude",
        )
        ecg_raw = _require_mapping(
            eeg_raw.get("ecg_amplitude", {}),
            path="eeg_bold_coupling.nuisance.eeg_artifact.ecg_amplitude",
        )
        peripheral_raw = _require_mapping(
            eeg_raw.get("peripheral_band_power", {}),
            path="eeg_bold_coupling.nuisance.eeg_artifact.peripheral_band_power",
        )
        eeg_artifact = EEGArtifactConfig(
            enabled=bool(eeg_raw.get("enabled", False)),
            output_column=str(eeg_raw.get("output_column", "eeg_artifact")).strip(),
            required_components=tuple(
                str(value).strip()
                for value in _require_sequence(
                    eeg_raw.get("required_components"),
                    path="eeg_bold_coupling.nuisance.eeg_artifact.required_components",
                )
                if str(value).strip()
            ),
            component_z_threshold=(
                None
                if eeg_raw.get("component_z_threshold", None) in {None, ""}
                else float(eeg_raw.get("component_z_threshold"))
            ),
            composite_threshold=(
                None
                if eeg_raw.get("composite_threshold", None) in {None, ""}
                else float(eeg_raw.get("composite_threshold"))
            ),
            event_numeric_columns=tuple(
                str(value).strip()
                for value in _require_sequence(
                    eeg_raw.get("event_numeric_columns"),
                    path="eeg_bold_coupling.nuisance.eeg_artifact.event_numeric_columns",
                )
                if str(value).strip()
            ),
            global_amplitude=GlobalAmplitudeConfig(
                enabled=bool(global_raw.get("enabled", True)),
                window=_require_float_pair(
                    global_raw.get("window"),
                    path="eeg_bold_coupling.nuisance.eeg_artifact.global_amplitude.window",
                    default=epoch_window,
                ),
            ),
            ecg_amplitude=ECGAmplitudeConfig(
                enabled=bool(ecg_raw.get("enabled", False)),
                channels=tuple(
                    str(value).strip()
                    for value in _require_sequence(
                        ecg_raw.get("channels"),
                        path="eeg_bold_coupling.nuisance.eeg_artifact.ecg_amplitude.channels",
                    )
                    if str(value).strip()
                ),
                window=_require_float_pair(
                    ecg_raw.get("window"),
                    path="eeg_bold_coupling.nuisance.eeg_artifact.ecg_amplitude.window",
                    default=epoch_window,
                ),
            ),
            peripheral_band_power=PeripheralBandPowerConfig(
                enabled=bool(peripheral_raw.get("enabled", False)),
                channels=tuple(
                    str(value).strip()
                    for value in _require_sequence(
                        peripheral_raw.get("channels"),
                        path="eeg_bold_coupling.nuisance.eeg_artifact.peripheral_band_power.channels",
                    )
                    if str(value).strip()
                ),
                band=_require_float_pair(
                    peripheral_raw.get("band"),
                    path="eeg_bold_coupling.nuisance.eeg_artifact.peripheral_band_power.band",
                    default=(30.0, 45.0),
                ),
                window=_require_float_pair(
                    peripheral_raw.get("window"),
                    path="eeg_bold_coupling.nuisance.eeg_artifact.peripheral_band_power.window",
                    default=active_window,
                ),
            ),
        )
        if eeg_artifact.enabled and not eeg_artifact.output_column:
            raise ValueError(
                "eeg_bold_coupling.nuisance.eeg_artifact.output_column must not be blank."
            )
        if eeg_artifact.enabled and not eeg_artifact.required_components:
            raise ValueError(
                "eeg_bold_coupling.nuisance.eeg_artifact.required_components must explicitly lock the composite definition."
            )
        if eeg_artifact.enabled and eeg_artifact.required_components == ("global_amplitude",):
            raise ValueError(
                "A global-amplitude-only EEG artifact composite is not a valid primary nuisance definition. "
                "Use artifact-specific components or disable eeg_artifact."
            )
        if eeg_artifact.enabled and eeg_artifact.component_z_threshold is None:
            raise ValueError(
                "eeg_bold_coupling.nuisance.eeg_artifact.component_z_threshold must be set "
                "when eeg_artifact is enabled."
            )
        if eeg_artifact.enabled and eeg_artifact.composite_threshold is None:
            raise ValueError(
                "eeg_bold_coupling.nuisance.eeg_artifact.composite_threshold must be set "
                "when eeg_artifact is enabled."
            )
        if eeg_artifact.ecg_amplitude.enabled and not eeg_artifact.ecg_amplitude.channels:
            raise ValueError(
                "eeg_bold_coupling.nuisance.eeg_artifact.ecg_amplitude.channels must not be empty when enabled."
            )
        if (
            eeg_artifact.peripheral_band_power.enabled
            and not eeg_artifact.peripheral_band_power.channels
        ):
            raise ValueError(
                "eeg_bold_coupling.nuisance.eeg_artifact.peripheral_band_power.channels must not be empty when enabled."
            )
        if eeg_artifact.enabled:
            has_component = (
                eeg_artifact.global_amplitude.enabled
                or eeg_artifact.ecg_amplitude.enabled
                or eeg_artifact.peripheral_band_power.enabled
                or bool(eeg_artifact.event_numeric_columns)
            )
            if not has_component:
                raise ValueError(
                    "eeg_bold_coupling.nuisance.eeg_artifact.enabled=true requires at least one component."
                )

        censor_raw = _require_mapping(
            raw.get("censoring", {}),
            path="eeg_bold_coupling.nuisance.censoring",
        )
        censoring = CensoringConfig(
            require_all_model_terms_finite=bool(
                censor_raw.get("require_all_model_terms_finite", True)
            ),
        )
        return cls(
            fd=fd,
            dvars=dvars,
            eeg_artifact=eeg_artifact,
            censoring=censoring,
        )


def _convolved_confound_for_trial(
    *,
    onset: float,
    duration: float,
    frame_times: np.ndarray,
    confound_values: np.ndarray,
    hrf_model: str,
    label: str,
) -> float:
    exp_condition = np.array(
        [
            [float(onset)],
            [float(duration)],
            [1.0],
        ],
        dtype=float,
    )
    regressors, _ = compute_regressor(
        exp_condition=exp_condition,
        hrf_model=hrf_model,
        frame_times=frame_times,
        con_id="trial",
        oversampling=50,
    )
    if regressors.ndim != 2 or regressors.shape[1] != 1:
        raise ValueError(
            "Trial-wise FD computation requires a single-column canonical HRF regressor."
        )
    weights = regressors[:, 0].astype(float, copy=False)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError(
            f"{label} regressor has non-positive support at onset={onset}, duration={duration}."
        )
    return float(np.sum(weights * confound_values) / weight_sum)


def _confounds_path_for_run(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
    run_num: int,
) -> Path:
    confounds_path = discover_confounds(
        bids_derivatives=deriv_root,
        subject=_subject_bids(subject),
        task=task,
        run_num=int(run_num),
    )
    if confounds_path is None or not confounds_path.exists():
        raise FileNotFoundError(
            f"Missing confounds TSV for motion nuisance: sub-{subject}, task-{task}, run-{int(run_num):02d}."
        )
    return confounds_path


def _motion_nuisance_table(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    trial_table: pd.DataFrame,
    output_column: str,
    source_column: str,
    hrf_model: str,
    fmri_space: str,
    label: str,
) -> pd.DataFrame:
    bids_derivatives = deriv_root
    subject_raw = _subject_raw(subject)
    rows: List[pd.DataFrame] = []
    for run_num, run_table in trial_table.groupby("run_num", sort=True):
        bold_path = discover_fmriprep_preproc_bold(
            bids_derivatives=bids_derivatives,
            subject=subject_raw,
            task=task,
            run_num=int(run_num),
            space=fmri_space,
        )
        if bold_path is None:
            raise FileNotFoundError(
                f"Missing preprocessed BOLD for {label}: sub-{subject_raw}, task-{task}, run-{int(run_num):02d}."
            )
        confounds_path = _confounds_path_for_run(
            deriv_root=bids_derivatives,
            subject=subject_raw,
            task=task,
            run_num=int(run_num),
        )

        confounds = pd.read_csv(confounds_path, sep="\t")
        if source_column not in confounds.columns:
            raise ValueError(
                f"Confounds file {confounds_path} is missing {source_column!r} required for {label}."
            )
        confound_values = pd.to_numeric(
            confounds[source_column],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        if not np.all(np.isfinite(confound_values)):
            raise ValueError(f"{label} confounds contain non-finite values for {confounds_path}.")

        tr = float(get_tr_from_bold(bold_path))
        n_scans = int(nib.load(str(bold_path)).shape[3])
        if n_scans != confound_values.size:
            raise ValueError(
                f"{label} vector length mismatch for {confounds_path}: confounds={confound_values.size}, scans={n_scans}."
            )
        frame_times = np.arange(n_scans, dtype=float) * tr
        trial_values = [
            _convolved_confound_for_trial(
                onset=float(row.onset),
                duration=float(row.duration),
                frame_times=frame_times,
                confound_values=confound_values,
                hrf_model=hrf_model,
                label=label,
            )
            for row in run_table.itertuples(index=False)
        ]
        run_out = run_table.copy()
        run_out[output_column] = np.asarray(trial_values, dtype=float)
        rows.append(run_out)

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out.sort_values(["run_num", "onset", "duration"]).reset_index(drop=True)


def compute_fd_table(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    trial_table: pd.DataFrame,
    nuisance_cfg: CouplingNuisanceConfig,
    fmri_space: str,
) -> pd.DataFrame:
    if not nuisance_cfg.fd.enabled:
        return trial_table.copy()
    return _motion_nuisance_table(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        trial_table=trial_table,
        output_column=nuisance_cfg.fd.output_column,
        source_column=nuisance_cfg.fd.source_column,
        hrf_model=nuisance_cfg.fd.hrf_model,
        fmri_space=fmri_space,
        label="FD nuisance",
    )


def compute_dvars_table(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    trial_table: pd.DataFrame,
    nuisance_cfg: CouplingNuisanceConfig,
    fmri_space: str,
) -> pd.DataFrame:
    if not nuisance_cfg.dvars.enabled:
        return trial_table.copy()
    return _motion_nuisance_table(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        trial_table=trial_table,
        output_column=nuisance_cfg.dvars.output_column,
        source_column=nuisance_cfg.dvars.source_column,
        hrf_model=nuisance_cfg.dvars.hrf_model,
        fmri_space=fmri_space,
        label="DVARS nuisance",
    )


def _global_amplitude_component(
    *,
    epochs: Any,
    mask: np.ndarray,
) -> np.ndarray:
    eeg_data = np.asarray(epochs.get_data(picks="eeg"), dtype=float)
    if eeg_data.ndim != 3 or eeg_data.shape[1] == 0:
        raise ValueError("Global EEG amplitude requires at least one EEG channel.")
    return metric_rms(eeg_data[:, :, mask])


def _ecg_amplitude_component(
    *,
    epochs: Any,
    channels: Sequence[str],
    mask: np.ndarray,
) -> np.ndarray:
    picks = pick_channels(
        epochs.info,
        channels,
        path="eeg_bold_coupling.nuisance.eeg_artifact.ecg_amplitude.channels",
    )
    data = np.asarray(epochs.get_data(picks=picks), dtype=float)
    return metric_rms(data[:, :, mask])


def _peripheral_band_power_component(
    *,
    epochs: Any,
    channels: Sequence[str],
    band: Tuple[float, float],
    mask: np.ndarray,
) -> np.ndarray:
    return band_power_metric(
        epochs=epochs,
        channels=channels,
        band=band,
        mask=mask,
        path="eeg_bold_coupling.nuisance.eeg_artifact.peripheral_band_power.channels",
    )


def compute_eeg_artifact_table(
    *,
    epochs: Any,
    clean_events: pd.DataFrame,
    eeg_table: pd.DataFrame,
    nuisance_cfg: CouplingNuisanceConfig,
) -> pd.DataFrame:
    if not nuisance_cfg.eeg_artifact.enabled:
        return eeg_table.copy()

    times = np.asarray(epochs.times, dtype=float)
    run_numbers = pd.to_numeric(eeg_table["run_num"], errors="raise").to_numpy(dtype=int)
    components: Dict[str, np.ndarray] = {}

    if nuisance_cfg.eeg_artifact.global_amplitude.enabled:
        mask = window_mask(
            times,
            nuisance_cfg.eeg_artifact.global_amplitude.window,
            path="eeg_bold_coupling.nuisance.eeg_artifact.global_amplitude.window",
        )
        components["global_amplitude"] = _global_amplitude_component(
            epochs=epochs,
            mask=mask,
        )

    if nuisance_cfg.eeg_artifact.ecg_amplitude.enabled:
        mask = window_mask(
            times,
            nuisance_cfg.eeg_artifact.ecg_amplitude.window,
            path="eeg_bold_coupling.nuisance.eeg_artifact.ecg_amplitude.window",
        )
        components["ecg_amplitude"] = _ecg_amplitude_component(
            epochs=epochs,
            channels=nuisance_cfg.eeg_artifact.ecg_amplitude.channels,
            mask=mask,
        )

    if nuisance_cfg.eeg_artifact.peripheral_band_power.enabled:
        mask = window_mask(
            times,
            nuisance_cfg.eeg_artifact.peripheral_band_power.window,
            path="eeg_bold_coupling.nuisance.eeg_artifact.peripheral_band_power.window",
        )
        components["peripheral_band_power"] = _peripheral_band_power_component(
            epochs=epochs,
            channels=nuisance_cfg.eeg_artifact.peripheral_band_power.channels,
            band=nuisance_cfg.eeg_artifact.peripheral_band_power.band,
            mask=mask,
        )

    for column in nuisance_cfg.eeg_artifact.event_numeric_columns:
        if column not in clean_events.columns:
            raise ValueError(
                f"Configured EEG artifact event_numeric_column {column!r} is missing from clean events."
            )
        values = pd.to_numeric(clean_events[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Configured EEG artifact event_numeric_column {column!r} contains non-finite values."
            )
        components[f"event_{_sanitize_name(column)}"] = values

    if not components:
        raise ValueError(
            "EEG artifact computation was enabled but no components were resolved."
        )
    resolved_components = tuple(sorted(components))
    required_components = tuple(sorted(nuisance_cfg.eeg_artifact.required_components))
    if resolved_components != required_components:
        raise ValueError(
            "Resolved EEG artifact components do not match "
            f"eeg_bold_coupling.nuisance.eeg_artifact.required_components. "
            f"Resolved={resolved_components}, required={required_components}."
        )

    out = eeg_table.copy()
    z_columns: List[str] = []
    for name, values in components.items():
        if values.shape[0] != len(out):
            raise ValueError(
                f"EEG artifact component {name!r} length mismatch: {values.shape[0]} vs {len(out)} trials."
            )
        raw_column = f"{nuisance_cfg.eeg_artifact.output_column}_{name}_raw"
        z_column = f"{nuisance_cfg.eeg_artifact.output_column}_{name}_z"
        out[raw_column] = np.asarray(values, dtype=float)
        out[z_column] = _robust_zscore_by_run(
            out[raw_column].to_numpy(dtype=float),
            run_numbers=run_numbers,
        )
        z_columns.append(z_column)

    z_matrix = out[z_columns].to_numpy(dtype=float)
    out[nuisance_cfg.eeg_artifact.output_column] = _positive_artifact_burden(z_matrix)
    return out


def _exclude_reason_columns(
    *,
    merged: pd.DataFrame,
    model_terms: Sequence[str],
    nuisance_cfg: CouplingNuisanceConfig,
) -> pd.DataFrame:
    out = merged.copy()
    out["exclude_fd_threshold"] = False
    out["exclude_dvars_threshold"] = False
    out["exclude_eeg_artifact_component"] = False
    out["exclude_eeg_artifact_composite"] = False
    out["exclude_non_finite_model_term"] = False

    if nuisance_cfg.fd.enabled and nuisance_cfg.fd.censor_above is not None:
        out["exclude_fd_threshold"] = (
            pd.to_numeric(out[nuisance_cfg.fd.output_column], errors="coerce")
            > float(nuisance_cfg.fd.censor_above)
        )

    if nuisance_cfg.dvars.enabled and nuisance_cfg.dvars.censor_above is not None:
        out["exclude_dvars_threshold"] = (
            pd.to_numeric(out[nuisance_cfg.dvars.output_column], errors="coerce")
            > float(nuisance_cfg.dvars.censor_above)
        )

    if nuisance_cfg.eeg_artifact.enabled and nuisance_cfg.eeg_artifact.component_z_threshold is not None:
        z_columns = [
            column
            for column in out.columns
            if column.startswith(f"{nuisance_cfg.eeg_artifact.output_column}_")
            and column.endswith("_z")
        ]
        if not z_columns:
            raise ValueError("EEG artifact component z columns are missing.")
        threshold = float(nuisance_cfg.eeg_artifact.component_z_threshold)
        out["exclude_eeg_artifact_component"] = (
            out[z_columns].to_numpy(dtype=float).max(axis=1) > threshold
        )

    if nuisance_cfg.eeg_artifact.enabled and nuisance_cfg.eeg_artifact.composite_threshold is not None:
        out["exclude_eeg_artifact_composite"] = (
            pd.to_numeric(out[nuisance_cfg.eeg_artifact.output_column], errors="coerce")
            > float(nuisance_cfg.eeg_artifact.composite_threshold)
        )

    if nuisance_cfg.censoring.require_all_model_terms_finite and model_terms:
        finite_mask = np.ones(len(out), dtype=bool)
        for term in model_terms:
            if term not in out.columns:
                raise ValueError(f"Configured model term {term!r} is missing from merged trial table.")
            values = pd.to_numeric(out[term], errors="coerce").to_numpy(dtype=float)
            finite_mask &= np.isfinite(values)
        out["exclude_non_finite_model_term"] = ~finite_mask

    exclusion_columns = [
        "exclude_fd_threshold",
        "exclude_dvars_threshold",
        "exclude_eeg_artifact_component",
        "exclude_eeg_artifact_composite",
        "exclude_non_finite_model_term",
    ]
    out["keep_trial"] = ~out[exclusion_columns].any(axis=1)
    reasons: List[str] = []
    for row in out[exclusion_columns].itertuples(index=False):
        active = [
            name.replace("exclude_", "")
            for name, flag in zip(exclusion_columns, row)
            if bool(flag)
        ]
        reasons.append(",".join(active))
    out["exclude_reason"] = reasons
    return out


def apply_trial_censoring(
    *,
    merged_table: pd.DataFrame,
    model_terms: Sequence[str],
    nuisance_cfg: CouplingNuisanceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    qc_table = _exclude_reason_columns(
        merged=merged_table,
        model_terms=model_terms,
        nuisance_cfg=nuisance_cfg,
    )
    kept = qc_table.loc[qc_table["keep_trial"]].reset_index(drop=True)
    summary = {
        "n_trials_input": int(len(qc_table)),
        "n_trials_kept": int(qc_table["keep_trial"].sum()),
        "n_trials_excluded": int((~qc_table["keep_trial"]).sum()),
        "n_excluded_fd_threshold": int(qc_table["exclude_fd_threshold"].sum()),
        "n_excluded_dvars_threshold": int(qc_table["exclude_dvars_threshold"].sum()),
        "n_excluded_eeg_artifact_component": int(
            qc_table["exclude_eeg_artifact_component"].sum()
        ),
        "n_excluded_eeg_artifact_composite": int(
            qc_table["exclude_eeg_artifact_composite"].sum()
        ),
        "n_excluded_non_finite_model_term": int(
            qc_table["exclude_non_finite_model_term"].sum()
        ),
        "model_terms": list(model_terms),
        "nuisance_config": asdict(nuisance_cfg),
    }
    return kept, qc_table, summary


def write_qc_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(summary), indent=2, default=str), encoding="utf-8")


__all__ = [
    "CouplingNuisanceConfig",
    "apply_trial_censoring",
    "compute_dvars_table",
    "compute_eeg_artifact_table",
    "compute_fd_table",
    "write_qc_summary",
]
