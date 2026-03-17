"""Trial-wise cortical EEG-BOLD coupling analysis."""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.surface import PolyData, PolyMesh, SurfaceImage, vol_to_surf
from scipy.signal import hilbert
from scipy.spatial import cKDTree

from eeg_pipeline.analysis.features.source_localization import _setup_surface_forward_model_configured
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_nuisance import (
    CouplingNuisanceConfig,
    apply_trial_censoring,
    compute_dvars_table,
    compute_eeg_artifact_table,
    compute_fd_table,
    write_qc_summary,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_fmri_qc import (
    run_subject_fmri_qc,
    write_subject_fmri_qc,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_permutation import (
    aggregate_primary_permutation_effects,
    fit_subject_primary_permutation_effects,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_roi_builder import (
    build_eeg_bold_rois,
    built_rois_as_runtime_specs,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_residualized import (
    aggregate_residualized_correlations,
    fit_subject_residualized_correlations,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_signatures import (
    LocalSignatureExpressionConfig,
    normalize_signature_weights,
    resolve_signature_paths,
    sample_signatures_to_subject_surface,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_sensitivity import (
    CouplingSensitivityConfig,
    filter_painful_trials,
    resolve_sensitivity_beta_dir,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_source import (
    iterate_band_specific_lcmv_estimates,
    compute_source_estimates,
)
from eeg_pipeline.analysis.paradigms.pain.eeg_bold_statistics import (
    CellSpec,
    CouplingStatisticsConfig,
    finalize_group_results,
    fit_mixedlm_cell,
    summarize_subject_cells,
)
from eeg_pipeline.infra.paths import ensure_dir, resolve_deriv_root
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.feature_alignment import require_trial_id_column
from eeg_pipeline.utils.data.preprocessing import extract_run_number
from fmri_pipeline.analysis.trial_signatures import (
    TrialInfo,
    TrialSignatureExtractionConfig,
    _build_lss_events,
    _discover_runs,
    run_trial_signature_extraction_for_subject,
)
from fmri_pipeline.utils.bold_discovery import (
    build_first_level_model as _build_first_level_model,
    get_tr_from_bold as _get_tr_from_bold,
    select_confounds as _select_confounds,
    validate_design_matrices as _validate_design_matrices,
)
from fmri_pipeline.utils.text import safe_slug as _safe_slug


LOGGER = logging.getLogger(__name__)
_EPS = 1.0e-12
_RUN_COLUMN_CANDIDATES = ("block", "run_id", "run", "session")
_SURFACE_DEPTH_FRACTIONS = tuple(np.linspace(0.0, 1.0, 7))


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _require_sequence(
    value: Any,
    *,
    path: str,
) -> Tuple[Any, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list.")
    return tuple(value)


def _require_float_pair(value: Any, *, path: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be a 2-item list [start, end].")
    start = float(value[0])
    end = float(value[1])
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError(f"{path} values must be finite.")
    if not start < end:
        raise ValueError(f"{path} must satisfy start < end.")
    return start, end


def _parse_run_label(run_value: Any) -> int:
    text = str(run_value).strip()
    if not text:
        raise ValueError("Run label is blank.")
    extracted = extract_run_number(Path(text))
    if extracted is not None:
        return int(extracted)
    match = re.search(r"(\d+)$", text)
    if match is None:
        raise ValueError(f"Could not parse run number from {run_value!r}.")
    return int(match.group(1))


def _subject_raw(subject: str) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text.replace("sub-", "", 1)
    return text


def _subject_bids(subject: str) -> str:
    return f"sub-{_subject_raw(subject)}"


def _make_trial_number_key(
    *,
    run_num: int,
    trial_number: int,
) -> str:
    return f"{int(run_num)}|trial-{int(trial_number):03d}"


def _contrast_vector_for_column(columns: Sequence[str], column_name: str) -> np.ndarray:
    try:
        index = list(columns).index(str(column_name))
    except ValueError as exc:
        raise KeyError(f"Design matrix column not found: {column_name!r}") from exc
    vector = np.zeros(len(columns), dtype=float)
    vector[index] = 1.0
    return vector


def _resolve_column(
    table: pd.DataFrame,
    *,
    explicit: Optional[str],
    aliases: Sequence[str],
    path: str,
) -> str:
    candidates: List[str] = []
    if explicit:
        candidates.append(str(explicit).strip())
    for alias in aliases:
        alias_text = str(alias).strip()
        if not alias_text:
            continue
        candidates.append(alias_text)
        if not alias_text.startswith("events_"):
            candidates.append(f"events_{alias_text}")
    for candidate in candidates:
        if candidate in table.columns:
            return candidate
    raise ValueError(
        f"Could not resolve {path}. Tried: {candidates}. "
        f"Available columns: {list(table.columns)}"
    )


def _resolve_optional_numeric_column(
    table: pd.DataFrame,
    *,
    explicit: Optional[str],
    aliases: Sequence[str],
) -> Optional[str]:
    candidates: List[str] = []
    if explicit:
        candidates.append(str(explicit).strip())
    for alias in aliases:
        alias_text = str(alias).strip()
        if not alias_text:
            continue
        candidates.append(alias_text)
        if not alias_text.startswith("events_"):
            candidates.append(f"events_{alias_text}")
    for candidate in candidates:
        if candidate not in table.columns:
            continue
        values = pd.to_numeric(table[candidate], errors="coerce")
        if np.any(np.isfinite(values.to_numpy(dtype=float))):
            return candidate
    return None


def _resolve_optional_trial_number_column(table: pd.DataFrame) -> Optional[str]:
    for candidate in ("trial_number", "events_trial_number", "events_trial_index"):
        if candidate not in table.columns:
            continue
        values = pd.to_numeric(table[candidate], errors="coerce")
        if np.any(np.isfinite(values.to_numpy(dtype=float))):
            return candidate
    return None


def _resolve_run_series(table: pd.DataFrame) -> Optional[pd.Series]:
    def _safe_parse(value: Any) -> float:
        text = str(value).strip()
        if not text:
            return np.nan
        try:
            return float(_parse_run_label(text))
        except ValueError:
            return np.nan

    for candidate in _RUN_COLUMN_CANDIDATES:
        if candidate not in table.columns:
            continue
        series = table[candidate]
        numeric = pd.to_numeric(series, errors="coerce")
        if np.any(np.isfinite(numeric.to_numpy(dtype=float))):
            return numeric
        parsed = pd.Series(
            [_safe_parse(v) for v in series],
            index=series.index,
            dtype="float64",
        )
        if np.any(np.isfinite(parsed.to_numpy(dtype=float))):
            return parsed
    return None


def _source_event_key(*, events_file: Any, events_row: Any) -> str:
    path = Path(str(events_file)).expanduser().resolve()
    row_index = int(events_row)
    return f"{path}::{row_index}"


def _run_num_from_source_events(
    *,
    events_df: pd.DataFrame,
    events_path: Path,
) -> np.ndarray:
    run_series = _resolve_run_series(events_df)
    if run_series is not None:
        run_numeric = pd.to_numeric(run_series, errors="coerce").to_numpy(dtype=float)
        if np.all(np.isfinite(run_numeric)):
            return run_numeric.astype(int)
    return np.full(len(events_df), int(_parse_run_label(events_path)), dtype=int)


def _selected_history_events(
    *,
    trials_df: pd.DataFrame,
    coupling_cfg: "EEGBOLDCouplingConfig",
    config: Any,
) -> pd.DataFrame:
    required_columns = {"events_file", "events_row"}
    if not required_columns.issubset(trials_df.columns):
        raise ValueError(
            f"LSS trials table is missing source-event columns: {sorted(required_columns - set(trials_df.columns))}"
        )

    allowed = {
        str(value).strip()
        for value in coupling_cfg.fmri.selection_values
        if str(value).strip()
    }
    if not allowed:
        raise ValueError("eeg_bold_coupling.fmri.selection_values must not be empty.")

    history_parts: List[pd.DataFrame] = []
    raw_event_paths = sorted({str(value).strip() for value in trials_df["events_file"].tolist()})
    for raw_path in raw_event_paths:
        events_path = Path(raw_path).expanduser().resolve()
        if not events_path.exists():
            raise FileNotFoundError(f"Missing source events file for trial history: {events_path}")
        events_df = pd.read_csv(events_path, sep="\t")
        if coupling_cfg.fmri.selection_column not in events_df.columns:
            raise ValueError(
                f"Source events file {events_path} is missing selection column "
                f"{coupling_cfg.fmri.selection_column!r}."
            )
        selected = events_df.loc[
            events_df[coupling_cfg.fmri.selection_column].astype(str).str.strip().isin(sorted(allowed))
        ].copy()
        if selected.empty:
            continue
        selected = selected.reset_index(drop=False).rename(columns={"index": "events_row"})
        selected["events_file"] = str(events_path)
        selected["source_event_key"] = [
            _source_event_key(events_file=events_path, events_row=row)
            for row in selected["events_row"].to_numpy(dtype=int)
        ]
        selected["run_num"] = _run_num_from_source_events(
            events_df=selected,
            events_path=events_path,
        )
        history_parts.append(selected)

    if not history_parts:
        raise ValueError("No source events matched the configured Study 2 trial selection.")

    history = pd.concat(history_parts, axis=0, ignore_index=True)
    if history["source_event_key"].duplicated().any():
        duplicates = sorted(
            history.loc[history["source_event_key"].duplicated(), "source_event_key"].unique()
        )
        raise ValueError(f"Duplicate source-event keys in full history table: {duplicates[:5]}")

    if "onset" not in history.columns or "duration" not in history.columns:
        raise ValueError("Source events history is missing onset/duration.")
    history["onset"] = pd.to_numeric(history["onset"], errors="coerce")
    history["duration"] = pd.to_numeric(history["duration"], errors="coerce")
    if not np.all(np.isfinite(history["onset"].to_numpy(dtype=float))):
        raise ValueError("Source events history onset contains non-finite values.")
    if not np.all(np.isfinite(history["duration"].to_numpy(dtype=float))):
        raise ValueError("Source events history duration contains non-finite values.")

    trial_number_column = _resolve_optional_trial_number_column(history)
    metadata = history.drop(columns=["onset", "duration"], errors="ignore").rename(
        columns=lambda column: column if str(column).startswith("events_") else f"events_{column}"
    )
    out = pd.DataFrame(
        {
            "source_event_key": history["source_event_key"].astype(str),
            "run_num": history["run_num"].to_numpy(dtype=int),
            "onset": history["onset"].to_numpy(dtype=float),
            "duration": history["duration"].to_numpy(dtype=float),
        }
    )
    if trial_number_column is not None:
        out["trial_number"] = pd.to_numeric(history[trial_number_column], errors="coerce")
    out = pd.concat([out, metadata.reset_index(drop=True)], axis=1)
    out = out.sort_values(["run_num", "onset", "duration"], kind="mergesort").reset_index(drop=True)
    out["exp_global"] = np.arange(len(out), dtype=int)
    out["block_start"] = out["run_num"].diff().fillna(1).ne(0).astype(int)
    out["trial_position"] = (out.groupby("run_num", sort=False).cumcount() + 1).astype(int)

    temperature_aliases = tuple(
        str(v).strip()
        for v in get_config_value(config, "event_columns.temperature", [])
        if str(v).strip()
    )
    temp_column = _resolve_optional_numeric_column(
        out,
        explicit=coupling_cfg.covariates.temperature_column,
        aliases=temperature_aliases,
    )
    if temp_column is not None:
        out["temperature"] = pd.to_numeric(out[temp_column], errors="coerce")
        if coupling_cfg.covariates.include_temperature_squared:
            out["temperature_sq"] = np.square(out["temperature"].to_numpy(dtype=float))
        delta_values = np.zeros(len(out), dtype=float)
        run_numbers = out["run_num"].to_numpy(dtype=int)
        temperatures = out["temperature"].to_numpy(dtype=float)
        for idx in range(1, len(out)):
            if run_numbers[idx] != run_numbers[idx - 1]:
                continue
            delta_values[idx] = temperatures[idx] - temperatures[idx - 1]
        out["delta_temperature"] = delta_values

    if coupling_cfg.covariates.site_column:
        site_column = _resolve_column(
            out,
            explicit=coupling_cfg.covariates.site_column,
            aliases=(),
            path="eeg_bold_coupling.covariates.site_column",
        )
        site_values = out[site_column].astype(str)
        out["exp_site"] = site_values.groupby(site_values).cumcount()

    for extra_column in coupling_cfg.covariates.extra_numeric_columns:
        resolved = _resolve_column(
            out,
            explicit=extra_column,
            aliases=(),
            path=f"extra numeric column {extra_column!r}",
        )
        out[extra_column] = pd.to_numeric(out[resolved], errors="coerce")

    return out


def _iter_label_components(label: Any) -> Iterable[Tuple[str, np.ndarray]]:
    label_cls_name = type(label).__name__
    if label_cls_name == "BiHemiLabel":
        if len(label.lh.vertices) > 0:
            yield "lh", np.asarray(label.lh.vertices, dtype=int)
        if len(label.rh.vertices) > 0:
            yield "rh", np.asarray(label.rh.vertices, dtype=int)
        return
    hemi = str(getattr(label, "hemi", "")).strip().lower()
    vertices = np.asarray(getattr(label, "vertices", []), dtype=int)
    if hemi not in {"lh", "rh"}:
        raise ValueError(f"Unsupported label hemisphere {hemi!r}.")
    yield hemi, vertices


@dataclass(frozen=True)
class CouplingROIConfig:
    name: str
    template_subject: str
    parcellation: Optional[str]
    annot_labels: Tuple[str, ...]
    label_files: Tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        default_template_subject: str,
    ) -> "CouplingROIConfig":
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("eeg_bold_coupling.rois.items[*].name is required.")
        template_subject = str(
            raw.get("template_subject", default_template_subject)
        ).strip()
        if not template_subject:
            raise ValueError(f"ROI {name!r} is missing template_subject.")
        annot_labels = tuple(str(v).strip() for v in _require_sequence(raw.get("annot_labels"), path=f"ROI {name} annot_labels") if str(v).strip())
        label_files = tuple(str(v).strip() for v in _require_sequence(raw.get("label_files"), path=f"ROI {name} label_files") if str(v).strip())
        if not annot_labels and not label_files:
            raise ValueError(
                f"ROI {name!r} must define annot_labels and/or label_files."
            )
        parcellation_raw = str(raw.get("parcellation", "") or "").strip()
        parcellation = parcellation_raw or None
        if annot_labels and parcellation is None:
            raise ValueError(
                f"ROI {name!r} uses annot_labels but has no parcellation."
            )
        return cls(
            name=name,
            template_subject=template_subject,
            parcellation=parcellation,
            annot_labels=annot_labels,
            label_files=label_files,
        )


@dataclass(frozen=True)
class CouplingEEGConfig:
    bands: Tuple[str, ...]
    active_window: Tuple[float, float]
    baseline_window: Tuple[float, float]
    feature_batch_size: int
    method: str
    spacing: str
    subjects_dir: Path
    mindist_mm: float
    reg: float
    snr: float
    loose: float
    depth: float


@dataclass(frozen=True)
class CouplingFMRIConfig:
    contrast_name: str
    input_source: str
    fmriprep_space: str
    require_fmriprep: bool
    extraction_method: str
    selection_column: str
    selection_values: Tuple[str, ...]
    hrf_model: str
    drift_model: Optional[str]
    high_pass_hz: float
    low_pass_hz: Optional[float]
    smoothing_fwhm: Optional[float]
    confounds_strategy: str
    lss_other_regressors: str


@dataclass(frozen=True)
class CouplingCovariateConfig:
    model_terms: Tuple[str, ...]
    temperature_column: Optional[str]
    include_temperature_squared: bool
    site_column: Optional[str]
    extra_numeric_columns: Tuple[str, ...]


@dataclass(frozen=True)
class CouplingAlignmentConfig:
    key_mode: str


@dataclass(frozen=True)
class CouplingRuntimeConfig:
    preflight: bool
    profile_stages: bool


@dataclass(frozen=True)
class TrialShuffleNegativeControlConfig:
    enabled: bool
    output_name: str

    @classmethod
    def from_config(cls, config: Any) -> "TrialShuffleNegativeControlConfig":
        raw = get_config_value(
            config,
            "eeg_bold_coupling.negative_controls.trial_shuffle",
            {},
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                "eeg_bold_coupling.negative_controls.trial_shuffle must be a mapping."
            )
        output_name = str(raw.get("output_name", "trial_shuffle")).strip()
        enabled = bool(raw.get("enabled", False))
        if enabled and not output_name:
            raise ValueError(
                "eeg_bold_coupling.negative_controls.trial_shuffle.output_name must not be blank."
            )
        return cls(enabled=enabled, output_name=output_name)


@dataclass(frozen=True)
class EEGBOLDCouplingConfig:
    output_dir: Optional[Path]
    rois: Tuple[CouplingROIConfig, ...]
    eeg: CouplingEEGConfig
    fmri: CouplingFMRIConfig
    covariates: CouplingCovariateConfig
    statistics: CouplingStatisticsConfig
    alignment: CouplingAlignmentConfig
    runtime: CouplingRuntimeConfig

    @classmethod
    def from_config(cls, config: Any) -> "EEGBOLDCouplingConfig":
        coupling_cfg = get_config_value(config, "eeg_bold_coupling", {})
        if not isinstance(coupling_cfg, dict):
            raise ValueError("eeg_bold_coupling must be a mapping.")

        output_dir = _as_path(coupling_cfg.get("output_dir"))

        rois_cfg = coupling_cfg.get("rois", {})
        if not isinstance(rois_cfg, dict):
            raise ValueError("eeg_bold_coupling.rois must be a mapping.")
        default_template_subject = str(rois_cfg.get("template_subject", "fsaverage")).strip()
        roi_items = rois_cfg.get("items", [])
        if not isinstance(roi_items, list) or not roi_items:
            raise ValueError(
                "eeg_bold_coupling.rois.items must contain at least one ROI definition."
            )
        rois = tuple(
            CouplingROIConfig.from_mapping(
                raw=item,
                default_template_subject=default_template_subject,
            )
            for item in roi_items
        )

        eeg_cfg = coupling_cfg.get("eeg", {})
        if not isinstance(eeg_cfg, dict):
            raise ValueError("eeg_bold_coupling.eeg must be a mapping.")
        subjects_dir = _as_path(eeg_cfg.get("subjects_dir"))
        if subjects_dir is None:
            subjects_dir = _as_path(get_config_value(config, "paths.freesurfer_dir", None))
        if subjects_dir is None:
            raise ValueError(
                "eeg_bold_coupling.eeg.subjects_dir or paths.freesurfer_dir is required."
            )
        bands = tuple(
            str(band).strip()
            for band in _require_sequence(eeg_cfg.get("bands"), path="eeg_bold_coupling.eeg.bands")
            if str(band).strip()
        )
        if not bands:
            raise ValueError("eeg_bold_coupling.eeg.bands must not be empty.")
        eeg = CouplingEEGConfig(
            bands=bands,
            active_window=_require_float_pair(
                eeg_cfg.get("active_window"),
                path="eeg_bold_coupling.eeg.active_window",
            ),
            baseline_window=_require_float_pair(
                eeg_cfg.get("baseline_window"),
                path="eeg_bold_coupling.eeg.baseline_window",
            ),
            feature_batch_size=int(eeg_cfg.get("feature_batch_size", 256)),
            method=str(eeg_cfg.get("method", "lcmv")).strip().lower(),
            spacing=str(eeg_cfg.get("spacing", "oct6")).strip(),
            subjects_dir=subjects_dir,
            mindist_mm=float(eeg_cfg.get("mindist_mm", 5.0)),
            reg=float(eeg_cfg.get("reg", 0.05)),
            snr=float(eeg_cfg.get("snr", 3.0)),
            loose=float(eeg_cfg.get("loose", 0.2)),
            depth=float(eeg_cfg.get("depth", 0.8)),
        )
        if eeg.feature_batch_size <= 0:
            raise ValueError(
                "eeg_bold_coupling.eeg.feature_batch_size must be positive."
            )
        if eeg.method not in {"lcmv", "eloreta", "dspm", "wmne"}:
            raise ValueError(
                "eeg_bold_coupling.eeg.method must be one of "
                "'lcmv', 'eloreta', 'dspm', or 'wmne'."
            )

        fmri_cfg = coupling_cfg.get("fmri", {})
        if not isinstance(fmri_cfg, dict):
            raise ValueError("eeg_bold_coupling.fmri must be a mapping.")
        selection_values = tuple(
            str(v).strip()
            for v in _require_sequence(
                fmri_cfg.get("selection_values"),
                path="eeg_bold_coupling.fmri.selection_values",
            )
            if str(v).strip()
        )
        if not selection_values:
            raise ValueError(
                "eeg_bold_coupling.fmri.selection_values must not be empty."
            )
        fmri = CouplingFMRIConfig(
            contrast_name=str(fmri_cfg.get("contrast_name", "eeg_bold_coupling")).strip(),
            input_source=str(fmri_cfg.get("input_source", "fmriprep")).strip().lower(),
            fmriprep_space=str(fmri_cfg.get("fmriprep_space", "T1w")).strip(),
            require_fmriprep=bool(fmri_cfg.get("require_fmriprep", True)),
            extraction_method=str(
                fmri_cfg.get("extraction_method", "surface_glm")
            ).strip().lower(),
            selection_column=str(fmri_cfg.get("selection_column", "")).strip(),
            selection_values=selection_values,
            hrf_model=str(fmri_cfg.get("hrf_model", "spm")).strip().lower(),
            drift_model=(
                str(fmri_cfg.get("drift_model", "")).strip().lower() or None
            ),
            high_pass_hz=float(fmri_cfg.get("high_pass_hz", 0.008)),
            low_pass_hz=(
                None
                if fmri_cfg.get("low_pass_hz", None) in {None, "", 0}
                else float(fmri_cfg.get("low_pass_hz"))
            ),
            smoothing_fwhm=(
                None
                if fmri_cfg.get("smoothing_fwhm", None) in {None, "", 0}
                else float(fmri_cfg.get("smoothing_fwhm"))
            ),
            confounds_strategy=str(
                fmri_cfg.get("confounds_strategy", "auto")
            ).strip(),
            lss_other_regressors=str(
                fmri_cfg.get("lss_other_regressors", "all")
            ).strip().lower(),
        )
        if not fmri.selection_column:
            raise ValueError("eeg_bold_coupling.fmri.selection_column is required.")
        if fmri.input_source != "fmriprep":
            raise ValueError(
                "eeg_bold_coupling.fmri.input_source must be 'fmriprep' for subject-surface ROI extraction."
            )
        if fmri.fmriprep_space.strip().lower() != "t1w":
            raise ValueError(
                "eeg_bold_coupling.fmri.fmriprep_space must be 'T1w' to match subject surfaces."
            )
        if fmri.extraction_method != "surface_glm":
            raise ValueError(
                "eeg_bold_coupling.fmri.extraction_method must be 'surface_glm'."
            )

        cov_cfg = coupling_cfg.get("covariates", {})
        if not isinstance(cov_cfg, dict):
            raise ValueError("eeg_bold_coupling.covariates must be a mapping.")
        model_terms = tuple(
            str(term).strip()
            for term in _require_sequence(
                cov_cfg.get("model_terms"),
                path="eeg_bold_coupling.covariates.model_terms",
            )
            if str(term).strip()
        )
        covariates = CouplingCovariateConfig(
            model_terms=model_terms,
            temperature_column=(
                str(cov_cfg.get("temperature_column", "")).strip() or None
            ),
            include_temperature_squared=bool(
                cov_cfg.get("include_temperature_squared", True)
            ),
            site_column=str(cov_cfg.get("site_column", "")).strip() or None,
            extra_numeric_columns=tuple(
                str(v).strip()
                for v in _require_sequence(
                    cov_cfg.get("extra_numeric_columns"),
                    path="eeg_bold_coupling.covariates.extra_numeric_columns",
                )
                if str(v).strip()
            ),
        )

        statistics = CouplingStatisticsConfig.from_config(config)

        align_cfg = coupling_cfg.get("alignment", {})
        if not isinstance(align_cfg, dict):
            raise ValueError("eeg_bold_coupling.alignment must be a mapping.")
        alignment = CouplingAlignmentConfig(
            key_mode=str(
                align_cfg.get("key_mode", "run_trial_number")
            ).strip().lower(),
        )
        if alignment.key_mode != "run_trial_number":
            raise ValueError(
                "eeg_bold_coupling.alignment.key_mode must be 'run_trial_number'."
            )
        runtime_cfg = coupling_cfg.get("runtime", {})
        if not isinstance(runtime_cfg, dict):
            raise ValueError("eeg_bold_coupling.runtime must be a mapping.")
        runtime = CouplingRuntimeConfig(
            preflight=bool(runtime_cfg.get("preflight", True)),
            profile_stages=bool(runtime_cfg.get("profile_stages", True)),
        )

        return cls(
            output_dir=output_dir,
            rois=rois,
            eeg=eeg,
            fmri=fmri,
            covariates=covariates,
            statistics=statistics,
            alignment=alignment,
            runtime=runtime,
        )


@dataclass(frozen=True)
class SubjectROI:
    name: str
    label: Any
    source_rows: Tuple[int, ...]
    source_weights: Tuple[float, ...]
    lh_vertices: Tuple[int, ...]
    rh_vertices: Tuple[int, ...]


@dataclass(frozen=True)
class SubjectCouplingOutputs:
    subject: str
    output_dir: Path
    analysis_cells_path: Path
    merged_trials_path: Path
    roi_manifest_path: Path


@dataclass(frozen=True)
class SubjectSurfaceInfo:
    pial_paths: Mapping[str, Path]
    white_paths: Mapping[str, Path]
    vertex_areas: Mapping[str, np.ndarray]
    white_points: Mapping[str, np.ndarray]


def _load_template_labels(
    *,
    spec: CouplingROIConfig,
    subjects_dir: Path,
) -> List[Any]:
    import mne

    labels: List[Any] = []
    if spec.annot_labels:
        annot_labels = mne.read_labels_from_annot(
            subject=spec.template_subject,
            parc=str(spec.parcellation),
            subjects_dir=str(subjects_dir),
            verbose=False,
        )
        label_map = {str(label.name): label for label in annot_labels}
        missing = [name for name in spec.annot_labels if name not in label_map]
        if missing:
            raise ValueError(
                f"ROI {spec.name!r} is missing annotation labels in "
                f"{spec.template_subject}/{spec.parcellation}: {missing}"
            )
        labels.extend(label_map[name] for name in spec.annot_labels)

    if spec.label_files:
        for label_file in spec.label_files:
            path = Path(label_file)
            if not path.is_absolute():
                path = subjects_dir / spec.template_subject / "label" / label_file
            if not path.exists():
                raise FileNotFoundError(
                    f"ROI {spec.name!r} label file not found: {path}"
                )
            labels.append(mne.read_label(str(path), subject=spec.template_subject))

    return labels


def _combine_labels(labels: Sequence[Any], *, roi_name: str) -> Any:
    if not labels:
        raise ValueError(f"ROI {roi_name!r} has no labels to combine.")
    combined = labels[0]
    for label in labels[1:]:
        combined = combined + label
    combined.name = roi_name
    return combined


def _morph_roi_label(
    *,
    label: Any,
    subject_from: str,
    subject_to: str,
    subjects_dir: Path,
) -> Any:
    label_cls_name = type(label).__name__
    if label_cls_name == "BiHemiLabel":
        lh = label.lh.morph(
            subject_to=subject_to,
            subject_from=subject_from,
            subjects_dir=str(subjects_dir),
        )
        rh = label.rh.morph(
            subject_to=subject_to,
            subject_from=subject_from,
            subjects_dir=str(subjects_dir),
        )
        combined = lh + rh
        combined.name = str(getattr(label, "name", "roi"))
        return combined
    morphed = label.morph(
        subject_to=subject_to,
        subject_from=subject_from,
        subjects_dir=str(subjects_dir),
    )
    return morphed


def _build_subject_labels(
    *,
    subject: str,
    cfg: EEGBOLDCouplingConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    return _build_subject_labels_from_specs(
        subject=subject,
        specs=cfg.rois,
        subjects_dir=cfg.eeg.subjects_dir,
        logger=logger,
    )


def _build_subject_labels_from_specs(
    *,
    subject: str,
    specs: Sequence[CouplingROIConfig],
    subjects_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    subject_labels: Dict[str, Any] = {}
    for spec in specs:
        template_labels = _load_template_labels(spec=spec, subjects_dir=subjects_dir)
        combined = _combine_labels(template_labels, roi_name=spec.name)
        morphed = _morph_roi_label(
            label=combined,
            subject_from=spec.template_subject,
            subject_to=subject,
            subjects_dir=subjects_dir,
        )
        subject_labels[spec.name] = morphed
        logger.info(
            "Prepared ROI %s for %s from template %s.",
            spec.name,
            subject,
            spec.template_subject,
        )
    return subject_labels


def _resolve_subject_output_dir(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
    cfg: EEGBOLDCouplingConfig,
) -> Path:
    subject_raw = _subject_raw(subject)
    if cfg.output_dir is not None:
        base = cfg.output_dir / _subject_bids(subject_raw)
    else:
        base = deriv_root / _subject_bids(subject_raw) / "multimodal" / "eeg_bold_coupling"
    return (
        base
        / f"task-{task}"
        / f"contrast-{_safe_slug(cfg.fmri.contrast_name)}"
    )


def _resolve_group_output_dir(
    *,
    deriv_root: Path,
    task: str,
    cfg: EEGBOLDCouplingConfig,
) -> Path:
    if cfg.output_dir is not None:
        base = cfg.output_dir / "group"
    else:
        base = deriv_root / "group" / "multimodal" / "eeg_bold_coupling"
    return (
        base
        / f"task-{task}"
        / f"contrast-{_safe_slug(cfg.fmri.contrast_name)}"
    )


class SubjectStageProfiler:
    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        subject: str,
        task: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_path = output_dir / "runtime_profile.json"
        self.subject = str(subject)
        self.task = str(task)
        self.started_at = time.time()
        self.current_stage: Optional[str] = None
        self.current_detail: Optional[str] = None
        self.current_started_at: Optional[float] = None
        self.completed_stages: List[Dict[str, Any]] = []
        self.status = "initialized"
        self.error_message = ""
        self._write()

    def _payload(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "task": self.task,
            "status": self.status,
            "current_stage": self.current_stage,
            "current_detail": self.current_detail,
            "completed_stages": self.completed_stages,
            "started_at_unix": self.started_at,
            "updated_at_unix": time.time(),
            "error_message": self.error_message,
        }

    def _write(self) -> None:
        if not self.enabled:
            return
        self.output_path.write_text(
            json.dumps(self._payload(), indent=2),
            encoding="utf-8",
        )

    @contextmanager
    def stage(self, name: str) -> Iterable[None]:
        if not self.enabled:
            yield
            return
        self.status = "running"
        self.current_stage = str(name)
        self.current_detail = None
        self.current_started_at = time.time()
        self._write()
        try:
            yield
        except Exception:
            self.status = "failed"
            self.error_message = f"Stage failed: {name}"
            self._write()
            raise
        duration_sec = float(time.time() - float(self.current_started_at))
        self.completed_stages.append(
            {
                "stage": str(name),
                "duration_sec": duration_sec,
            }
        )
        self.current_stage = None
        self.current_detail = None
        self.current_started_at = None
        self.status = "running"
        self._write()

    def touch(self, detail: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self.current_detail = None if detail is None else str(detail).strip()
        self._write()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.current_stage = None
        self.current_detail = None
        self.current_started_at = None
        self.error_message = ""
        self._write()

    def mark_failed(self, message: str) -> None:
        self.status = "failed"
        self.current_detail = None
        self.error_message = str(message).strip()
        self._write()


def _validate_runtime_paths_exist(paths: Sequence[Path], *, label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: {missing}")


def _subject_surface_dir(
    *,
    subject: str,
    subjects_dir: Path,
) -> Path:
    return subjects_dir / _subject_bids(subject) / "surf"


def _subject_bem_dir(
    *,
    subject: str,
    subjects_dir: Path,
) -> Path:
    return subjects_dir / _subject_bids(subject) / "bem"


def _run_subject_preflight(
    *,
    subject: str,
    task: str,
    config: Any,
    coupling_cfg: EEGBOLDCouplingConfig,
    output_dir: Path,
) -> None:
    deriv_root = resolve_deriv_root(config=config)
    bids_root = _as_path(get_config_value(config, "paths.bids_root", None))
    bids_fmri_root = _as_path(get_config_value(config, "paths.bids_fmri_root", None))
    if bids_root is None:
        raise ValueError("paths.bids_root is required for EEG-BOLD coupling.")
    if bids_fmri_root is None:
        raise ValueError("paths.bids_fmri_root is required for EEG-BOLD coupling.")
    _validate_runtime_paths_exist(
        [deriv_root, bids_root, bids_fmri_root, coupling_cfg.eeg.subjects_dir],
        label="required root path(s)",
    )
    subject_surface_dir = _subject_surface_dir(
        subject=subject,
        subjects_dir=coupling_cfg.eeg.subjects_dir,
    )
    subject_bem_dir = _subject_bem_dir(
        subject=subject,
        subjects_dir=coupling_cfg.eeg.subjects_dir,
    )
    required_subject_files = [
        subject_surface_dir / "lh.pial",
        subject_surface_dir / "rh.pial",
        subject_surface_dir / "lh.white",
        subject_surface_dir / "rh.white",
        subject_bem_dir / f"{_subject_bids(subject)}-trans.fif",
    ]
    bem_solutions = sorted(
        subject_bem_dir.glob(f"{_subject_bids(subject)}-*-bem-sol.fif")
    )
    if not bem_solutions:
        raise FileNotFoundError(
            f"Missing BEM solution for {_subject_bids(subject)} in {subject_bem_dir}."
        )
    _validate_runtime_paths_exist(
        required_subject_files + [bem_solutions[0]],
        label="required subject surface/BEM file(s)",
    )
    template_label_paths: List[Path] = []
    for roi in coupling_cfg.rois:
        for label_file in roi.label_files:
            label_path = Path(label_file)
            if not label_path.is_absolute():
                label_path = (
                    coupling_cfg.eeg.subjects_dir
                    / roi.template_subject
                    / "label"
                    / label_file
                )
            template_label_paths.append(label_path)
    if template_label_paths:
        _validate_runtime_paths_exist(
            template_label_paths,
            label="ROI label file(s)",
        )
    if coupling_cfg.fmri.extraction_method == "surface_glm":
        _validate_runtime_paths_exist(
            [
                subject_surface_dir / "lh.pial",
                subject_surface_dir / "rh.pial",
                subject_surface_dir / "lh.white",
                subject_surface_dir / "rh.white",
            ],
            label="surface GLM mesh file(s)",
        )
    signature_paths = resolve_signature_paths(config)
    if signature_paths:
        _validate_runtime_paths_exist(
            list(signature_paths.values()),
            label="signature map file(s)",
        )
    if not output_dir.exists():
        raise FileNotFoundError(f"Subject output directory was not created: {output_dir}")
    task_value = str(task).strip()
    if not task_value:
        raise ValueError("Task must not be blank.")


def _validate_clean_event_qc_columns(
    *,
    clean_events: pd.DataFrame,
    nuisance_cfg: CouplingNuisanceConfig,
) -> None:
    artifact_cfg = nuisance_cfg.eeg_artifact
    if not artifact_cfg.enabled:
        return
    missing = [
        str(column)
        for column in artifact_cfg.event_numeric_columns
        if str(column) not in clean_events.columns
    ]
    if missing:
        raise ValueError(
            "Clean events are missing required EEG artifact QC columns: "
            f"{missing}"
        )


def _run_group_preflight(
    *,
    config: Any,
    coupling_cfg: EEGBOLDCouplingConfig,
) -> None:
    if coupling_cfg.statistics.backend != "nlme_lme_ar1":
        raise ValueError("Unsupported group backend.")
    from eeg_pipeline.analysis.paradigms.pain.eeg_bold_statistics import (
        _resolve_rscript_path,
        _r_backend_script_path,
    )

    _resolve_rscript_path(coupling_cfg.statistics.rscript_path)
    _r_backend_script_path()


def _materialize_runtime_roi_specs(config: Any) -> None:
    built_rois = build_eeg_bold_rois(config=config, logger=LOGGER)
    if not built_rois:
        return
    config.setdefault("eeg_bold_coupling", {}).setdefault("rois", {})[
        "items"
    ] = built_rois_as_runtime_specs(
        built_rois,
        template_subject=str(
            get_config_value(
                config,
                "eeg_bold_coupling.roi_builder.template_subject",
                "fsaverage",
            )
        ).strip(),
    )


def _resolve_subject_source_model(
    *,
    subject: str,
    epochs: Any,
    config: Any,
    coupling_cfg: EEGBOLDCouplingConfig,
    logger: logging.Logger,
) -> Tuple[Any, Any]:
    subject_raw = _subject_raw(subject)
    subject_bids = _subject_bids(subject_raw)
    bem_dir = coupling_cfg.eeg.subjects_dir / subject_bids / "bem"
    trans_path = bem_dir / f"{subject_bids}-trans.fif"
    bem_candidates = sorted(bem_dir.glob(f"{subject_bids}-*-bem-sol.fif"))
    bem_path = bem_candidates[0] if bem_candidates else None
    if not trans_path.exists() or bem_path is None or not bem_path.exists():
        raise ValueError(
            f"Missing subject-specific trans/BEM for {subject_bids}."
        )
    return _setup_surface_forward_model_configured(
        info=epochs.info,
        subject=subject_bids,
        subjects_dir=str(coupling_cfg.eeg.subjects_dir),
        spacing=coupling_cfg.eeg.spacing,
        trans=str(trans_path),
        bem=str(bem_path),
        mindist_mm=coupling_cfg.eeg.mindist_mm,
        logger=logger,
    )


def _compute_source_estimates(
    *,
    epochs: Any,
    fwd: Any,
    cfg: EEGBOLDCouplingConfig,
    logger: logging.Logger,
) -> List[Any]:
    return compute_source_estimates(
        epochs=epochs,
        fwd=fwd,
        method=cfg.eeg.method,
        baseline_window=cfg.eeg.baseline_window,
        reg=cfg.eeg.reg,
        loose=cfg.eeg.loose,
        depth=cfg.eeg.depth,
        snr=cfg.eeg.snr,
        logger=logger,
    )


def _source_vertices_from_forward(fwd: Any) -> Tuple[np.ndarray, np.ndarray]:
    src = fwd.get("src") if isinstance(fwd, dict) else None
    if not isinstance(src, (list, tuple)) or len(src) != 2:
        raise ValueError("Expected a two-hemisphere surface source space in the forward model.")
    lh_vertices = np.asarray(src[0]["vertno"], dtype=int)
    rh_vertices = np.asarray(src[1]["vertno"], dtype=int)
    if lh_vertices.ndim != 1 or rh_vertices.ndim != 1:
        raise ValueError("Forward-model source vertices must be 1D arrays.")
    return lh_vertices, rh_vertices


def _map_label_vertices_to_source_weights(
    *,
    label_vertices: np.ndarray,
    source_vertices: np.ndarray,
    surface_points: np.ndarray,
    vertex_areas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if label_vertices.size == 0 or source_vertices.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    source_coords = np.asarray(surface_points[source_vertices], dtype=float)
    label_coords = np.asarray(surface_points[label_vertices], dtype=float)
    label_areas = np.asarray(vertex_areas[label_vertices], dtype=float)
    tree = cKDTree(source_coords)
    nearest = tree.query(label_coords, k=1)[1]
    weight_by_source = np.zeros(source_vertices.shape[0], dtype=float)
    np.add.at(weight_by_source, np.asarray(nearest, dtype=int), label_areas)
    keep = weight_by_source > 0
    return (
        np.asarray(source_vertices[keep], dtype=int),
        np.asarray(weight_by_source[keep], dtype=float),
    )


def _map_rois_to_source_rows(
    *,
    source_vertices: Tuple[np.ndarray, np.ndarray],
    labels: Mapping[str, Any],
    surface_info: SubjectSurfaceInfo,
) -> List[SubjectROI]:
    lh_vertices = np.asarray(source_vertices[0], dtype=int)
    rh_vertices = np.asarray(source_vertices[1], dtype=int)
    rh_offset = int(lh_vertices.size)

    rois: List[SubjectROI] = []
    for name, label in labels.items():
        source_rows: List[int] = []
        source_weights: List[float] = []
        roi_lh_vertices: List[int] = []
        roi_rh_vertices: List[int] = []
        for hemi, label_vertices in _iter_label_components(label):
            label_vertex_array = np.asarray(label_vertices, dtype=int)
            if hemi == "lh":
                weighted_vertices, weighted_areas = _map_label_vertices_to_source_weights(
                    label_vertices=label_vertex_array,
                    source_vertices=lh_vertices,
                    surface_points=np.asarray(surface_info.white_points["lh"], dtype=float),
                    vertex_areas=np.asarray(surface_info.vertex_areas["lh"], dtype=float),
                )
                matched = [int(np.searchsorted(lh_vertices, vertex)) for vertex in weighted_vertices]
                roi_lh_vertices = [int(v) for v in label_vertex_array.tolist()]
            else:
                weighted_vertices, weighted_areas = _map_label_vertices_to_source_weights(
                    label_vertices=label_vertex_array,
                    source_vertices=rh_vertices,
                    surface_points=np.asarray(surface_info.white_points["rh"], dtype=float),
                    vertex_areas=np.asarray(surface_info.vertex_areas["rh"], dtype=float),
                )
                matched = [
                    rh_offset + int(np.searchsorted(rh_vertices, vertex))
                    for vertex in weighted_vertices
                ]
                roi_rh_vertices = [int(v) for v in label_vertex_array.tolist()]
            source_rows.extend(int(row) for row in matched)
            source_weights.extend(float(v) for v in weighted_areas.tolist())
        if not source_rows:
            raise ValueError(
                f"ROI {name!r} has no vertices in the subject source space."
            )
        unique_rows = np.asarray(source_rows, dtype=int)
        unique_weights = np.asarray(source_weights, dtype=float)
        order = np.argsort(unique_rows)
        unique_rows = unique_rows[order]
        unique_weights = unique_weights[order]
        dedup_rows, inverse = np.unique(unique_rows, return_inverse=True)
        dedup_weights = np.zeros(dedup_rows.shape[0], dtype=float)
        np.add.at(dedup_weights, inverse, unique_weights)
        rois.append(
            SubjectROI(
                name=name,
                label=label,
                source_rows=tuple(int(v) for v in dedup_rows.tolist()),
                source_weights=tuple(float(v) for v in dedup_weights.tolist()),
                lh_vertices=tuple(sorted(set(roi_lh_vertices))),
                rh_vertices=tuple(sorted(set(roi_rh_vertices))),
            )
        )
    return rois


def _build_window_mask(
    *,
    times: np.ndarray,
    window: Tuple[float, float],
    label: str,
) -> np.ndarray:
    start, end = window
    mask = (times >= float(start)) & (times <= float(end))
    if not np.any(mask):
        raise ValueError(
            f"{label} window {window} does not overlap epoch times."
        )
    return mask


def _extract_trialwise_eeg_features(
    *,
    stcs: Sequence[Any],
    rois: Sequence[SubjectROI],
    bands: Sequence[str],
    frequency_bands: Mapping[str, Sequence[float]],
    active_mask: np.ndarray,
    baseline_mask: np.ndarray,
    sfreq: float,
    feature_batch_size: int,
) -> pd.DataFrame:
    from mne.filter import filter_data

    if not stcs:
        raise ValueError("No source estimates were computed.")
    first_block = np.asarray(stcs[0].data, dtype=float)
    if first_block.ndim != 2:
        raise ValueError("Source estimates must be scalar surface estimates.")
    n_sources, n_times = first_block.shape
    for stc in stcs[1:]:
        shape = np.asarray(stc.data).shape
        if shape != (n_sources, n_times):
            raise ValueError(
                "All source estimates must share the same (n_sources, n_times) shape."
            )
    n_epochs = len(stcs)
    rows: Dict[str, np.ndarray] = {}
    roi_rows = {
        roi.name: np.asarray(roi.source_rows, dtype=int)
        for roi in rois
    }
    roi_weights = {
        roi.name: np.asarray(roi.source_weights, dtype=float)
        for roi in rois
    }
    requested_rows = np.unique(
        np.concatenate([source_rows for source_rows in roi_rows.values()])
    )
    if requested_rows.size == 0:
        raise ValueError("No ROI source vertices were available for EEG extraction.")
    power_floor = float(np.finfo(float).tiny)
    roi_positions: Dict[str, np.ndarray] = {}
    for roi in rois:
        positions = np.searchsorted(requested_rows, roi_rows[roi.name])
        if not np.array_equal(requested_rows[positions], roi_rows[roi.name]):
            raise ValueError(
                f"ROI {roi.name!r} contains source rows outside the requested vertex set."
            )
        roi_positions[roi.name] = positions

    for band in bands:
        if band not in frequency_bands:
            raise ValueError(f"Unknown band {band!r}.")
        fmin, fmax = float(frequency_bands[band][0]), float(frequency_bands[band][1])
        roi_active_sums = {
            roi.name: np.zeros(n_epochs, dtype=float)
            for roi in rois
        }
        roi_baseline_sums = {
            roi.name: np.zeros(n_epochs, dtype=float)
            for roi in rois
        }
        roi_counts = {
            roi.name: 0.0
            for roi in rois
        }
        for start in range(0, requested_rows.size, feature_batch_size):
            stop = min(start + feature_batch_size, requested_rows.size)
            batch_source_rows = requested_rows[start:stop]
            batch_positions: Dict[str, np.ndarray] = {}
            batch_weights: Dict[str, np.ndarray] = {}
            for roi in rois:
                positions = roi_positions[roi.name]
                within_batch = positions[
                    (positions >= start) & (positions < stop)
                ] - start
                if within_batch.size == 0:
                    continue
                batch_positions[roi.name] = within_batch
                full_weights = roi_weights[roi.name]
                roi_batch_weights = full_weights[
                    (positions >= start) & (positions < stop)
                ]
                batch_weights[roi.name] = roi_batch_weights
                roi_counts[roi.name] += float(np.sum(roi_batch_weights))
            for epoch_index in range(n_epochs):
                epoch_batch = np.asarray(
                    stcs[epoch_index].data[batch_source_rows, :],
                    dtype=float,
                )
                filtered = filter_data(
                    epoch_batch,
                    sfreq=sfreq,
                    l_freq=fmin,
                    h_freq=fmax,
                    method="iir",
                    iir_params={"order": 4, "ftype": "butter"},
                    phase="zero",
                    copy=True,
                    verbose=False,
                )
                analytic = hilbert(filtered, axis=-1)
                power = np.abs(analytic) ** 2
                active = np.nanmean(power[:, active_mask], axis=-1)
                baseline = np.nanmean(power[:, baseline_mask], axis=-1)
                for roi_name, within_batch in batch_positions.items():
                    weight_vector = np.asarray(batch_weights[roi_name], dtype=float)
                    roi_active_sums[roi_name][epoch_index] += float(
                        np.nansum(active[within_batch] * weight_vector)
                    )
                    roi_baseline_sums[roi_name][epoch_index] += float(
                        np.nansum(baseline[within_batch] * weight_vector)
                    )
        for roi in rois:
            column = f"eeg_{roi.name}_{band}"
            if roi_counts[roi.name] <= 0:
                raise ValueError(
                    f"ROI {roi.name!r} does not overlap any source vertices."
                )
            roi_active_mean = roi_active_sums[roi.name] / float(roi_counts[roi.name])
            roi_baseline_mean = roi_baseline_sums[roi.name] / float(roi_counts[roi.name])
            rows[column] = 10.0 * np.log10(
                np.maximum(roi_active_mean, power_floor)
                / np.maximum(roi_baseline_mean, power_floor)
            )

    return pd.DataFrame(rows)


def _extract_trialwise_band_features_from_stc_stream(
    *,
    stcs: Iterable[Any],
    rois: Sequence[SubjectROI],
    band: str,
    active_mask: np.ndarray,
    baseline_mask: np.ndarray,
) -> pd.DataFrame:
    stc_iterator = iter(stcs)
    try:
        first_stc = next(stc_iterator)
    except StopIteration as exc:
        raise ValueError("No source estimates were computed.") from exc
    first_block = np.asarray(first_stc.data, dtype=float)
    if first_block.ndim != 2:
        raise ValueError("Source estimates must be scalar surface estimates.")
    n_sources, n_times = first_block.shape
    power_floor = float(np.finfo(float).tiny)
    roi_rows = {
        roi.name: np.asarray(roi.source_rows, dtype=int)
        for roi in rois
    }
    roi_weights = {
        roi.name: np.asarray(roi.source_weights, dtype=float)
        for roi in rois
    }
    for roi in rois:
        source_rows = roi_rows[roi.name]
        source_weights = roi_weights[roi.name]
        if source_rows.size == 0:
            raise ValueError(f"ROI {roi.name!r} has no source rows.")
        if source_weights.shape != source_rows.shape:
            raise ValueError(
                f"ROI {roi.name!r} source weights do not match source rows."
            )
        total_weight = float(np.sum(source_weights))
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError(
                f"ROI {roi.name!r} has non-positive total source weight."
            )
    requested_rows = np.unique(
        np.concatenate([source_rows for source_rows in roi_rows.values()])
    )
    if requested_rows.size == 0:
        raise ValueError("No ROI source vertices were available for EEG extraction.")
    roi_positions: Dict[str, np.ndarray] = {}
    roi_total_weights: Dict[str, float] = {}
    for roi in rois:
        positions = np.searchsorted(requested_rows, roi_rows[roi.name])
        if not np.array_equal(requested_rows[positions], roi_rows[roi.name]):
            raise ValueError(
                f"ROI {roi.name!r} contains source rows outside the requested vertex set."
            )
        roi_positions[roi.name] = positions
        roi_total_weights[roi.name] = float(np.sum(roi_weights[roi.name]))
    roi_active_values = {roi.name: [] for roi in rois}
    roi_baseline_values = {roi.name: [] for roi in rois}

    def append_epoch(stc: Any) -> None:
        shape = np.asarray(stc.data).shape
        if shape != (n_sources, n_times):
            raise ValueError(
                "All source estimates must share the same (n_sources, n_times) shape."
            )
        source_data = np.asarray(stc.data[requested_rows, :], dtype=float)
        analytic = hilbert(source_data, axis=-1)
        power = np.abs(analytic) ** 2
        active_power = np.nanmean(power[:, active_mask], axis=-1)
        baseline_power = np.nanmean(power[:, baseline_mask], axis=-1)
        for roi in rois:
            weights = roi_weights[roi.name]
            positions = roi_positions[roi.name]
            total_weight = roi_total_weights[roi.name]
            roi_active_values[roi.name].append(
                float(np.sum(active_power[positions] * weights) / total_weight)
            )
            roi_baseline_values[roi.name].append(
                float(np.sum(baseline_power[positions] * weights) / total_weight)
            )

    append_epoch(first_stc)
    for stc in stc_iterator:
        append_epoch(stc)

    rows: Dict[str, np.ndarray] = {}
    for roi in rois:
        roi_active = np.asarray(roi_active_values[roi.name], dtype=float)
        roi_baseline = np.asarray(roi_baseline_values[roi.name], dtype=float)
        rows[f"eeg_{roi.name}_{band}"] = 10.0 * np.log10(
            np.maximum(roi_active, power_floor)
            / np.maximum(roi_baseline, power_floor)
        )
    return pd.DataFrame(rows)


def _extract_trialwise_eeg_table(
    *,
    epochs: Any,
    clean_events: pd.DataFrame,
    fwd: Any,
    rois: Sequence[SubjectROI],
    cfg: EEGBOLDCouplingConfig,
    frequency_bands: Mapping[str, Sequence[float]],
    active_mask: np.ndarray,
    baseline_mask: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    if cfg.eeg.method == "lcmv":
        eeg_feature_tables = []
        for band in cfg.eeg.bands:
            band_stcs = iterate_band_specific_lcmv_estimates(
                epochs=epochs,
                fwd=fwd,
                band=band,
                frequency_bands=frequency_bands,
                baseline_window=cfg.eeg.baseline_window,
                reg=cfg.eeg.reg,
                logger=logger,
            )
            eeg_feature_tables.append(
                _extract_trialwise_band_features_from_stc_stream(
                    stcs=band_stcs,
                    rois=rois,
                    band=band,
                    active_mask=active_mask,
                    baseline_mask=baseline_mask,
                )
            )
        eeg_features = pd.concat(eeg_feature_tables, axis=1)
    else:
        stcs = _compute_source_estimates(
            epochs=epochs,
            fwd=fwd,
            cfg=cfg,
            logger=logger,
        )
        eeg_features = _extract_trialwise_eeg_features(
            stcs=stcs,
            rois=rois,
            bands=cfg.eeg.bands,
            frequency_bands=frequency_bands,
            active_mask=active_mask,
            baseline_mask=baseline_mask,
            sfreq=float(epochs.info["sfreq"]),
            feature_batch_size=cfg.eeg.feature_batch_size,
        )
    logger.info("Computed EEG source-power features for %d trials", len(eeg_features))
    eeg_table = _prepare_eeg_trial_table(
        events_df=clean_events,
        eeg_features=eeg_features,
        alignment_cfg=cfg.alignment,
    )
    return eeg_table


def _prepare_eeg_trial_table(
    *,
    events_df: pd.DataFrame,
    eeg_features: pd.DataFrame,
    alignment_cfg: CouplingAlignmentConfig,
) -> pd.DataFrame:
    require_trial_id_column(events_df, context="EEG clean events")
    if "onset" not in events_df.columns or "duration" not in events_df.columns:
        raise ValueError("EEG clean events must contain onset and duration.")
    run_series = _resolve_run_series(events_df)
    if run_series is None:
        raise ValueError(
            "EEG clean events are missing a usable run/block column."
        )
    run_numeric = pd.to_numeric(run_series, errors="coerce")
    if not np.all(np.isfinite(run_numeric.to_numpy(dtype=float))):
        raise ValueError("EEG clean events run column contains non-finite values.")
    onset = pd.to_numeric(events_df["onset"], errors="coerce")
    duration = pd.to_numeric(events_df["duration"], errors="coerce")
    if not np.all(np.isfinite(onset.to_numpy(dtype=float))):
        raise ValueError("EEG clean events onset contains non-finite values.")
    if not np.all(np.isfinite(duration.to_numpy(dtype=float))):
        raise ValueError("EEG clean events duration contains non-finite values.")

    out = events_df[["trial_id"]].copy()
    out["run_num"] = run_numeric.to_numpy(dtype=int)
    out["onset"] = onset.to_numpy(dtype=float)
    out["duration"] = duration.to_numpy(dtype=float)
    trial_number_column = _resolve_optional_trial_number_column(events_df)
    if trial_number_column is None:
        raise ValueError(
            "EEG clean events are missing a usable trial_number column for run_trial_number alignment."
        )
    out["trial_number"] = pd.to_numeric(
        events_df[trial_number_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(out["trial_number"].to_numpy(dtype=float))):
        raise ValueError("EEG clean-event trial_number contains non-finite values.")
    out["trial_key"] = [
        _make_trial_number_key(
            run_num=int(run_num),
            trial_number=int(trial_number),
        )
        for run_num, trial_number in zip(
            out["run_num"].to_numpy(dtype=int),
            out["trial_number"].to_numpy(dtype=float),
        )
    ]
    metadata = events_df.copy()
    metadata = metadata.drop(columns=["trial_id", "onset", "duration"], errors="ignore")
    metadata = metadata.rename(
        columns={column: f"events_{column}" for column in metadata.columns}
    )
    if len(out) != len(eeg_features):
        raise ValueError(
            "EEG features row count does not match EEG clean events row count."
        )
    return pd.concat(
        [
            out.reset_index(drop=True),
            metadata.reset_index(drop=True),
            eeg_features.reset_index(drop=True),
        ],
        axis=1,
    )


def _build_lss_config(
    *,
    task: str,
    cfg: EEGBOLDCouplingConfig,
) -> TrialSignatureExtractionConfig:
    return TrialSignatureExtractionConfig(
        input_source=cfg.fmri.input_source,
        fmriprep_space=cfg.fmri.fmriprep_space,
        require_fmriprep=cfg.fmri.require_fmriprep,
        runs=None,
        task=task,
        name=cfg.fmri.contrast_name,
        condition_a_column="unused_condition_a",
        condition_a_value="unused_condition_a",
        condition_b_column="unused_condition_b",
        condition_b_value="unused_condition_b",
        hrf_model=cfg.fmri.hrf_model,
        drift_model=cfg.fmri.drift_model,
        high_pass_hz=cfg.fmri.high_pass_hz,
        low_pass_hz=cfg.fmri.low_pass_hz,
        smoothing_fwhm=cfg.fmri.smoothing_fwhm,
        confounds_strategy=cfg.fmri.confounds_strategy,
        method="lss",
        include_other_events=True,
        lss_other_regressors=cfg.fmri.lss_other_regressors,
        signature_group_column=cfg.fmri.selection_column,
        signature_group_values=cfg.fmri.selection_values,
        signature_group_scope="across_runs",
        write_trial_betas=True,
        write_trial_variances=True,
        write_condition_betas=False,
    )


def _run_subject_lss(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    output_dir: Path,
    config: Any,
    coupling_cfg: EEGBOLDCouplingConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    bids_fmri_root = _as_path(get_config_value(config, "paths.bids_fmri_root", None))
    if bids_fmri_root is None:
        raise ValueError("paths.bids_fmri_root is required for EEG-BOLD coupling.")
    bids_derivatives = _as_path(get_config_value(config, "paths.deriv_root", None))
    result = run_trial_signature_extraction_for_subject(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        deriv_root=deriv_root,
        subject=_subject_raw(subject),
        cfg=_build_lss_config(task=task, cfg=coupling_cfg),
        signature_root=None,
        signature_specs=None,
        output_dir=output_dir / "lss",
        progress_callback=progress_callback,
    )
    output_dir = _as_path(result.get("output_dir"))
    if output_dir is None:
        raise ValueError("LSS output_dir was not returned.")
    return output_dir


def _prepare_trial_table(
    *,
    trials_df: pd.DataFrame,
    coupling_cfg: EEGBOLDCouplingConfig,
    config: Any,
) -> pd.DataFrame:
    required = {"run", "trial_index", "onset", "duration"}
    if not required.issubset(set(trials_df.columns)):
        raise ValueError(
            f"LSS trials table is missing required columns: {sorted(required - set(trials_df.columns))}"
        )
    required_history_columns = {"events_file", "events_row"}
    if not required_history_columns.issubset(set(trials_df.columns)):
        raise ValueError(
            "LSS trials table must include events_file and events_row to recover full session history."
        )
    out = trials_df.copy()
    out["run_num"] = out["run"].map(_parse_run_label)
    out["onset"] = pd.to_numeric(out["onset"], errors="coerce")
    out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
    if not np.all(np.isfinite(out["onset"].to_numpy(dtype=float))):
        raise ValueError("LSS trials table onset contains non-finite values.")
    if not np.all(np.isfinite(out["duration"].to_numpy(dtype=float))):
        raise ValueError("LSS trials table duration contains non-finite values.")
    trial_number_column = _resolve_optional_trial_number_column(out)
    if trial_number_column is None:
        raise ValueError(
            "LSS trials table is missing a usable trial_number column for run_trial_number alignment."
        )
    out["trial_number"] = pd.to_numeric(out[trial_number_column], errors="coerce")
    if not np.all(np.isfinite(out["trial_number"].to_numpy(dtype=float))):
        raise ValueError("LSS trials table trial_number contains non-finite values.")
    out["trial_key"] = [
        _make_trial_number_key(
            run_num=int(run_num),
            trial_number=int(trial_number),
        )
        for run_num, trial_number in zip(
            out["run_num"].to_numpy(dtype=int),
            out["trial_number"].to_numpy(dtype=float),
        )
    ]
    if out["trial_key"].duplicated().any():
        dupes = sorted(out.loc[out["trial_key"].duplicated(), "trial_key"].unique())
        raise ValueError(f"LSS trials table has duplicate trial keys: {dupes[:5]}")

    out["source_event_key"] = [
        _source_event_key(events_file=events_file, events_row=events_row)
        for events_file, events_row in zip(out["events_file"], out["events_row"])
    ]
    history = _selected_history_events(
        trials_df=trials_df,
        coupling_cfg=coupling_cfg,
        config=config,
    )
    history_columns = [
        "source_event_key",
        "exp_global",
        "block_start",
        "trial_position",
        "temperature",
        "temperature_sq",
        "delta_temperature",
        "exp_site",
        *coupling_cfg.covariates.extra_numeric_columns,
    ]
    available_history_columns = [
        column for column in history_columns if column in history.columns
    ]
    out = out.merge(
        history[available_history_columns],
        on="source_event_key",
        how="left",
        validate="one_to_one",
    )
    derived_history_columns = [
        column for column in available_history_columns if column != "source_event_key"
    ]
    if not derived_history_columns:
        raise ValueError("No full-session trial history covariates were derived.")
    if out[derived_history_columns].isna().all(axis=None):
        raise ValueError("Full-session trial history covariates failed to merge onto selected trials.")
    return out.sort_values(["run_num", "onset", "duration"]).reset_index(drop=True)


def _validate_model_terms_present(
    *,
    table: pd.DataFrame,
    terms: Sequence[str],
) -> None:
    missing_terms = [term for term in terms if term not in table.columns]
    if missing_terms:
        raise ValueError(
            f"Configured model terms are missing from the merged trial table: {missing_terms}"
        )


def _beta_path_for_trial(
    *,
    subject: str,
    task: str,
    beta_root: Path,
    run_label: str,
    trial_index: int,
) -> Path:
    sub_label = _subject_bids(subject)
    return (
        beta_root
        / "trial_betas"
        / run_label
        / f"{sub_label}_task-{task}_{run_label}_trial-{int(trial_index):03d}_beta.nii.gz"
    )


def _variance_path_for_trial(
    *,
    subject: str,
    task: str,
    beta_root: Path,
    run_label: str,
    trial_index: int,
) -> Path:
    sub_label = _subject_bids(subject)
    return (
        beta_root
        / "trial_betas"
        / run_label
        / f"{sub_label}_task-{task}_{run_label}_trial-{int(trial_index):03d}_var.nii.gz"
    )


def _subject_surface_paths(
    *,
    subject: str,
    subjects_dir: Path,
) -> Dict[str, Dict[str, Path]]:
    surf_dir = subjects_dir / _subject_bids(subject) / "surf"
    hemis: Dict[str, Dict[str, Path]] = {}
    for hemi in ("lh", "rh"):
        pial = surf_dir / f"{hemi}.pial"
        white = surf_dir / f"{hemi}.white"
        if not pial.exists() or not white.exists():
            raise FileNotFoundError(
                f"Missing surface files for {subject}: {pial} / {white}"
            )
        hemis[hemi] = {"pial": pial, "white": white}
    return hemis


def _vertex_areas(
    *,
    points: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Surface points must have shape (n_vertices, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Surface triangles must have shape (n_triangles, 3).")
    tri_points = np.asarray(points[triangles], dtype=float)
    edge_a = tri_points[:, 1, :] - tri_points[:, 0, :]
    edge_b = tri_points[:, 2, :] - tri_points[:, 0, :]
    tri_area = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    vertex_area = np.zeros(points.shape[0], dtype=float)
    for column in range(3):
        np.add.at(vertex_area, triangles[:, column], tri_area / 3.0)
    return vertex_area


def _read_surface_vertex_areas(surface_path: Path) -> np.ndarray:
    import mne

    points, triangles = mne.read_surface(str(surface_path), verbose=False)
    return _vertex_areas(
        points=np.asarray(points, dtype=float),
        triangles=np.asarray(triangles, dtype=int),
    )


def _read_surface_points(surface_path: Path) -> np.ndarray:
    import mne

    points, _triangles = mne.read_surface(str(surface_path), verbose=False)
    return np.asarray(points, dtype=float)


def _subject_surface_info(
    *,
    subject: str,
    subjects_dir: Path,
) -> SubjectSurfaceInfo:
    surface_paths = _subject_surface_paths(
        subject=subject,
        subjects_dir=subjects_dir,
    )
    pial_paths = {
        hemi: hemi_paths["pial"]
        for hemi, hemi_paths in surface_paths.items()
    }
    white_paths = {
        hemi: hemi_paths["white"]
        for hemi, hemi_paths in surface_paths.items()
    }
    vertex_areas = {
        hemi: (
            _read_surface_vertex_areas(pial_paths[hemi])
            + _read_surface_vertex_areas(white_paths[hemi])
        )
        / 2.0
        for hemi in ("lh", "rh")
    }
    white_points = {
        hemi: _read_surface_points(white_paths[hemi])
        for hemi in ("lh", "rh")
    }
    return SubjectSurfaceInfo(
        pial_paths=pial_paths,
        white_paths=white_paths,
        vertex_areas=vertex_areas,
        white_points=white_points,
    )


def _extract_local_expression(
    *,
    roi: SubjectROI,
    lh_texture: np.ndarray,
    rh_texture: np.ndarray,
    lh_vertex_areas: np.ndarray,
    rh_vertex_areas: np.ndarray,
    signature_weights: Mapping[str, Dict[str, np.ndarray]],
    expression_cfg: LocalSignatureExpressionConfig,
) -> Dict[str, float]:
    rows: Dict[str, float] = {}
    for signature_name, hemi_weights in signature_weights.items():
        values: List[np.ndarray] = []
        weights: List[np.ndarray] = []
        if roi.lh_vertices:
            vertices = np.asarray(roi.lh_vertices, dtype=int)
            roi_weights = np.asarray(hemi_weights["lh"], dtype=float)[vertices]
            roi_values = np.asarray(lh_texture, dtype=float)[vertices]
            roi_areas = np.asarray(lh_vertex_areas, dtype=float)[vertices]
            mask = np.abs(roi_weights) > float(expression_cfg.abs_weight_threshold)
            if np.any(mask):
                values.append(roi_values[mask])
                weights.append(roi_weights[mask] * roi_areas[mask])
        if roi.rh_vertices:
            vertices = np.asarray(roi.rh_vertices, dtype=int)
            roi_weights = np.asarray(hemi_weights["rh"], dtype=float)[vertices]
            roi_values = np.asarray(rh_texture, dtype=float)[vertices]
            roi_areas = np.asarray(rh_vertex_areas, dtype=float)[vertices]
            mask = np.abs(roi_weights) > float(expression_cfg.abs_weight_threshold)
            if np.any(mask):
                values.append(roi_values[mask])
                weights.append(roi_weights[mask] * roi_areas[mask])
        if not values or not weights:
            continue
        value_vector = np.concatenate(values)
        weight_vector = normalize_signature_weights(
            weights=np.concatenate(weights),
            mode=expression_cfg.normalize_weights,
        )
        rows[f"bold_localexpr_{signature_name}_{roi.name}"] = float(
            np.sum(value_vector * weight_vector)
        )
    return rows


def _collect_roi_surface_values(
    *,
    roi: SubjectROI,
    lh_values: np.ndarray,
    rh_values: np.ndarray,
    lh_vertex_areas: np.ndarray,
    rh_vertex_areas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    value_parts: List[np.ndarray] = []
    area_parts: List[np.ndarray] = []
    if roi.lh_vertices:
        lh_vertices = np.asarray(roi.lh_vertices, dtype=int)
        value_parts.append(np.asarray(lh_values, dtype=float)[lh_vertices])
        area_parts.append(np.asarray(lh_vertex_areas, dtype=float)[lh_vertices])
    if roi.rh_vertices:
        rh_vertices = np.asarray(roi.rh_vertices, dtype=int)
        value_parts.append(np.asarray(rh_values, dtype=float)[rh_vertices])
        area_parts.append(np.asarray(rh_vertex_areas, dtype=float)[rh_vertices])
    if not value_parts or not area_parts:
        raise ValueError(
            f"ROI {roi.name!r} has no surface vertices for cortical extraction."
        )
    values = np.concatenate(value_parts)
    areas = np.concatenate(area_parts)
    finite_mask = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
    if not np.any(finite_mask):
        raise ValueError(
            f"ROI {roi.name!r} has no finite positive-area vertices for cortical extraction."
        )
    return values[finite_mask], areas[finite_mask]


def _area_weighted_mean(
    *,
    values: np.ndarray,
    areas: np.ndarray,
) -> float:
    total_area = float(np.sum(areas))
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Area-weighted mean requires strictly positive total surface area.")
    return float(np.sum(values * areas) / total_area)


def _area_weighted_variance_of_mean(
    *,
    variances: np.ndarray,
    areas: np.ndarray,
) -> float:
    total_area = float(np.sum(areas))
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError(
            "Area-weighted variance summary requires strictly positive total surface area."
        )
    normalized = np.asarray(areas, dtype=float) / total_area
    return float(np.sum(np.square(normalized) * np.asarray(variances, dtype=float)))


def _sample_volume_to_surface(
    *,
    img: Any,
    pial_path: Path,
    white_path: Path,
) -> np.ndarray:
    values = vol_to_surf(
        img,
        surf_mesh=str(pial_path),
        inner_mesh=str(white_path),
        kind="depth",
        depth=_SURFACE_DEPTH_FRACTIONS,
    )
    return np.asarray(values, dtype=float)


def _coerce_surface_matrix(
    *,
    values: np.ndarray,
    n_vertices: int,
    label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.shape[0] != n_vertices:
            raise ValueError(
                f"{label} surface vector does not match vertex count {n_vertices}."
            )
        return array.reshape(n_vertices, 1)
    if array.ndim != 2:
        raise ValueError(f"{label} surface data must be 1D or 2D.")
    if array.shape[0] == n_vertices:
        return array
    if array.shape[1] == n_vertices:
        return array.T
    raise ValueError(
        f"{label} surface matrix shape {array.shape} does not match vertex count {n_vertices}."
    )


def _surface_image_parts(surface_img: SurfaceImage) -> Tuple[np.ndarray, np.ndarray]:
    left = np.asarray(surface_img.data.parts["left"], dtype=float)
    right = np.asarray(surface_img.data.parts["right"], dtype=float)
    return np.squeeze(left), np.squeeze(right)


def _build_surface_run_image(
    *,
    bold_img: Any,
    surface_info: SubjectSurfaceInfo,
) -> SurfaceImage:
    left = _coerce_surface_matrix(
        values=_sample_volume_to_surface(
            img=bold_img,
            pial_path=surface_info.pial_paths["lh"],
            white_path=surface_info.white_paths["lh"],
        ),
        n_vertices=surface_info.vertex_areas["lh"].shape[0],
        label="Left-hemisphere BOLD",
    )
    right = _coerce_surface_matrix(
        values=_sample_volume_to_surface(
            img=bold_img,
            pial_path=surface_info.pial_paths["rh"],
            white_path=surface_info.white_paths["rh"],
        ),
        n_vertices=surface_info.vertex_areas["rh"].shape[0],
        label="Right-hemisphere BOLD",
    )
    return SurfaceImage(
        mesh=PolyMesh(
            left=str(surface_info.pial_paths["lh"]),
            right=str(surface_info.pial_paths["rh"]),
        ),
        data=PolyData(left=left, right=right),
    )


def _trial_infos_by_run(
    trial_table: pd.DataFrame,
) -> Dict[int, List[TrialInfo]]:
    required = {
        "run_num",
        "run",
        "trial_index",
        "condition",
        "onset",
        "duration",
        "original_trial_type",
        "events_file",
        "events_row",
    }
    missing = required - set(trial_table.columns)
    if missing:
        raise ValueError(
            f"Trial table is missing required columns for surface GLM extraction: {sorted(missing)}"
        )
    grouped: Dict[int, List[TrialInfo]] = {}
    for row in trial_table.sort_values(["run_num", "onset", "duration"]).itertuples(index=False):
        extra = {
            str(column).replace("events_", "", 1): str(getattr(row, column))
            for column in trial_table.columns
            if str(column).startswith("events_")
        }
        grouped.setdefault(int(row.run_num), []).append(
            TrialInfo(
                run=int(row.run_num),
                run_label=str(row.run),
                trial_index=int(row.trial_index),
                condition=str(row.condition),
                regressor="target",
                onset=float(row.onset),
                duration=float(row.duration),
                original_trial_type=str(row.original_trial_type),
                source_events_path=Path(str(row.events_file)).expanduser().resolve(),
                source_row=int(row.events_row),
                extra=extra,
            )
        )
    return grouped


def _surface_record_for_trial(
    *,
    trial_key: str,
    rois: Sequence[SubjectROI],
    surface_info: SubjectSurfaceInfo,
    beta_surface: SurfaceImage,
    variance_surface: Optional[SurfaceImage],
    signature_weights: Optional[Mapping[str, Dict[str, np.ndarray]]],
    expression_cfg: Optional[LocalSignatureExpressionConfig],
) -> Dict[str, Any]:
    lh_beta, rh_beta = _surface_image_parts(beta_surface)
    lh_variance = rh_variance = None
    if variance_surface is not None:
        lh_variance, rh_variance = _surface_image_parts(variance_surface)
    record: Dict[str, Any] = {"trial_key": str(trial_key)}
    for roi in rois:
        values, areas = _collect_roi_surface_values(
            roi=roi,
            lh_values=np.asarray(lh_beta, dtype=float),
            rh_values=np.asarray(rh_beta, dtype=float),
            lh_vertex_areas=surface_info.vertex_areas["lh"],
            rh_vertex_areas=surface_info.vertex_areas["rh"],
        )
        record[f"bold_{roi.name}"] = _area_weighted_mean(values=values, areas=areas)
        if lh_variance is not None and rh_variance is not None:
            variance_values, variance_areas = _collect_roi_surface_values(
                roi=roi,
                lh_values=np.asarray(lh_variance, dtype=float),
                rh_values=np.asarray(rh_variance, dtype=float),
                lh_vertex_areas=surface_info.vertex_areas["lh"],
                rh_vertex_areas=surface_info.vertex_areas["rh"],
            )
            record[f"boldvar_{roi.name}"] = _area_weighted_variance_of_mean(
                variances=variance_values,
                areas=variance_areas,
            )
    if (
        signature_weights is not None
        and expression_cfg is not None
        and expression_cfg.enabled
    ):
        for roi in rois:
            record.update(
                _extract_local_expression(
                    roi=roi,
                    lh_texture=np.asarray(lh_beta, dtype=float),
                    rh_texture=np.asarray(rh_beta, dtype=float),
                    lh_vertex_areas=surface_info.vertex_areas["lh"],
                    rh_vertex_areas=surface_info.vertex_areas["rh"],
                    signature_weights=signature_weights,
                    expression_cfg=expression_cfg,
                )
            )
    return record


def _extract_trialwise_bold_features_surface_glm(
    *,
    subject: str,
    task: str,
    trial_table: pd.DataFrame,
    rois: Sequence[SubjectROI],
    surface_info: SubjectSurfaceInfo,
    config: Any,
    coupling_cfg: EEGBOLDCouplingConfig,
    require_variance: bool,
    signature_weights: Optional[Mapping[str, Dict[str, np.ndarray]]] = None,
    expression_cfg: Optional[LocalSignatureExpressionConfig] = None,
    logger: Optional[logging.Logger] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    subject_label = _subject_raw(subject)
    bids_fmri_root = _as_path(get_config_value(config, "paths.bids_fmri_root", None))
    bids_derivatives = _as_path(get_config_value(config, "paths.deriv_root", None))
    if bids_fmri_root is None:
        raise ValueError("paths.bids_fmri_root is required for surface GLM extraction.")
    lss_cfg = _build_lss_config(task=task, cfg=coupling_cfg)
    discovered_runs = _discover_runs(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject_label,
        task=task,
        runs=sorted(set(int(v) for v in trial_table["run_num"].tolist())),
        input_source=coupling_cfg.fmri.input_source,
        fmriprep_space=coupling_cfg.fmri.fmriprep_space,
        require_fmriprep=coupling_cfg.fmri.require_fmriprep,
    )
    runs_by_number = {
        int(run_num): (bold_path, events_path, confounds_path)
        for run_num, bold_path, events_path, confounds_path in discovered_runs
    }
    trials_by_run = _trial_infos_by_run(trial_table)
    trial_key_lookup = {
        (int(row.run_num), int(row.trial_index)): str(row.trial_key)
        for row in trial_table.itertuples(index=False)
    }
    rows: List[Dict[str, Any]] = []
    for run_num, trials in sorted(trials_by_run.items()):
        if progress_callback is not None:
            progress_callback(f"surface_glm {subject_label} run-{run_num:02d}")
        if run_num not in runs_by_number:
            raise FileNotFoundError(
                f"Could not resolve fMRI run inputs for run {run_num}."
            )
        bold_path, events_path, confounds_path = runs_by_number[run_num]
        events_df = pd.read_csv(events_path, sep="\t")
        confounds = None
        if str(coupling_cfg.fmri.confounds_strategy).strip().lower() not in {"none", "no", "off"}:
            confounds, confound_columns = _select_confounds(
                confounds_path,
                coupling_cfg.fmri.confounds_strategy,
                logger=logger,
            )
            if confounds is None or not confound_columns:
                raise ValueError(
                    f"Surface GLM requires confounds for {bold_path.name}."
                )
        bold_img = nib.load(str(bold_path))
        surface_run = _build_surface_run_image(
            bold_img=bold_img,
            surface_info=surface_info,
        )
        tr = _get_tr_from_bold(bold_path)
        for trial in trials:
            if progress_callback is not None:
                progress_callback(
                    f"surface_glm {subject_label} {trial.run_label} "
                    f"trial-{trial.trial_index:03d}"
                )
            lss_events = _build_lss_events(
                trial=trial,
                all_trials=trials,
                original_events_df=events_df,
                cfg=lss_cfg,
            )
            flm = _build_first_level_model(
                tr=tr,
                cfg=lss_cfg,
                mask_img=None,
                logger=logger,
            )
            flm.fit(surface_run, events=lss_events, confounds=confounds)
            _validate_design_matrices(
                flm,
                context=(
                    f"Surface LSS GLM ({bold_path.name}, {trial.run_label}, "
                    f"trial {trial.trial_index:03d})"
                ),
                min_residual_dof=1,
            )
            design_matrix = flm.design_matrices_[0]
            contrast = _contrast_vector_for_column(
                list(getattr(design_matrix, "columns", [])),
                "target",
            )
            beta_surface = flm.compute_contrast(contrast, output_type="effect_size")
            variance_surface = None
            if require_variance:
                variance_surface = flm.compute_contrast(
                    contrast,
                    output_type="effect_variance",
                )
            rows.append(
                _surface_record_for_trial(
                    trial_key=trial_key_lookup[
                        (int(trial.run), int(trial.trial_index))
                    ],
                    rois=rois,
                    surface_info=surface_info,
                    beta_surface=beta_surface,
                    variance_surface=variance_surface,
                    signature_weights=signature_weights,
                    expression_cfg=expression_cfg,
                )
            )
    return pd.DataFrame(rows)


def _extract_trialwise_bold_features(
    *,
    subject: str,
    task: str,
    beta_root: Path,
    trial_table: pd.DataFrame,
    rois: Sequence[SubjectROI],
    subjects_dir: Path,
    require_variance: bool,
    signature_weights: Optional[Mapping[str, Dict[str, np.ndarray]]] = None,
    expression_cfg: Optional[LocalSignatureExpressionConfig] = None,
) -> pd.DataFrame:
    surface_info = _subject_surface_info(
        subject=subject,
        subjects_dir=subjects_dir,
    )
    rows: List[Dict[str, Any]] = []
    for row in trial_table.itertuples(index=False):
        beta_path = _beta_path_for_trial(
            subject=subject,
            task=task,
            beta_root=beta_root,
            run_label=str(row.run),
            trial_index=int(row.trial_index),
        )
        if not beta_path.exists():
            raise FileNotFoundError(f"Missing trial beta image: {beta_path}")
        variance_path = _variance_path_for_trial(
            subject=subject,
            task=task,
            beta_root=beta_root,
            run_label=str(row.run),
            trial_index=int(row.trial_index),
        )
        if require_variance and not variance_path.exists():
            raise FileNotFoundError(
                f"Missing trial variance image required for weighted modeling: {variance_path}"
            )
        img = nib.load(str(beta_path))
        lh_texture = _sample_volume_to_surface(
            img=img,
            pial_path=surface_info.pial_paths["lh"],
            white_path=surface_info.white_paths["lh"],
        )
        rh_texture = _sample_volume_to_surface(
            img=img,
            pial_path=surface_info.pial_paths["rh"],
            white_path=surface_info.white_paths["rh"],
        )
        lh_variance_texture: Optional[np.ndarray] = None
        rh_variance_texture: Optional[np.ndarray] = None
        if variance_path.exists():
            variance_img = nib.load(str(variance_path))
            lh_variance_texture = _sample_volume_to_surface(
                img=variance_img,
                pial_path=surface_info.pial_paths["lh"],
                white_path=surface_info.white_paths["lh"],
            )
            rh_variance_texture = _sample_volume_to_surface(
                img=variance_img,
                pial_path=surface_info.pial_paths["rh"],
                white_path=surface_info.white_paths["rh"],
            )
        record: Dict[str, Any] = {"trial_key": str(row.trial_key)}
        for roi in rois:
            roi_values, roi_areas = _collect_roi_surface_values(
                roi=roi,
                lh_values=np.asarray(lh_texture, dtype=float),
                rh_values=np.asarray(rh_texture, dtype=float),
                lh_vertex_areas=np.asarray(surface_info.vertex_areas["lh"], dtype=float),
                rh_vertex_areas=np.asarray(surface_info.vertex_areas["rh"], dtype=float),
            )
            record[f"bold_{roi.name}"] = _area_weighted_mean(
                values=roi_values,
                areas=roi_areas,
            )
            if lh_variance_texture is not None and rh_variance_texture is not None:
                roi_variances, variance_areas = _collect_roi_surface_values(
                    roi=roi,
                    lh_values=lh_variance_texture,
                    rh_values=rh_variance_texture,
                    lh_vertex_areas=np.asarray(surface_info.vertex_areas["lh"], dtype=float),
                    rh_vertex_areas=np.asarray(surface_info.vertex_areas["rh"], dtype=float),
                )
                record[f"boldvar_{roi.name}"] = _area_weighted_variance_of_mean(
                    variances=roi_variances,
                    areas=variance_areas,
                )
            if signature_weights is not None and expression_cfg is not None:
                record.update(
                    _extract_local_expression(
                        roi=roi,
                        lh_texture=np.asarray(lh_texture, dtype=float),
                        rh_texture=np.asarray(rh_texture, dtype=float),
                        lh_vertex_areas=np.asarray(surface_info.vertex_areas["lh"], dtype=float),
                        rh_vertex_areas=np.asarray(surface_info.vertex_areas["rh"], dtype=float),
                        signature_weights=signature_weights,
                        expression_cfg=expression_cfg,
                    )
                )
        rows.append(record)
    return pd.DataFrame(rows)


def _merge_trialwise_tables(
    *,
    eeg_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    bold_table: pd.DataFrame,
) -> pd.DataFrame:
    def _coalesce_column(
        table: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        if name in table.columns:
            return table
        left_name = f"{name}_x"
        right_name = f"{name}_y"
        if left_name not in table.columns and right_name not in table.columns:
            raise KeyError(name)
        if left_name not in table.columns:
            return table.rename(columns={right_name: name})
        if right_name not in table.columns:
            return table.rename(columns={left_name: name})
        left = table[left_name]
        right = table[right_name]
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        has_numeric_values = bool(
            np.isfinite(left_num.to_numpy(dtype=float)).any()
            or np.isfinite(right_num.to_numpy(dtype=float)).any()
        )
        same_numeric = has_numeric_values and np.allclose(
            left_num.to_numpy(dtype=float),
            right_num.to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        same_text = left.fillna("__nan__").astype(str).equals(
            right.fillna("__nan__").astype(str)
        )
        if not same_numeric and not same_text:
            raise ValueError(
                f"Merged trial column {name!r} differs between EEG and fMRI tables."
            )
        table[name] = left
        return table.drop(columns=[left_name, right_name])

    trial_subset = trial_table.copy()
    merged = eeg_table.copy().merge(
        trial_subset,
        on="trial_key",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No trials matched between EEG and LSS trial tables.")
    merged = merged.merge(
        bold_table,
        on="trial_key",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No trials remained after merging BOLD ROI values.")
    for column in ("onset", "duration"):
        left_name = f"{column}_x"
        right_name = f"{column}_y"
        if left_name in merged.columns and right_name in merged.columns:
            eeg_column = f"eeg_{column}"
            merged[eeg_column] = pd.to_numeric(merged[left_name], errors="coerce")
            merged[column] = pd.to_numeric(merged[right_name], errors="coerce")
            if not np.all(np.isfinite(merged[eeg_column].to_numpy(dtype=float))):
                raise ValueError(f"Merged EEG timing column {eeg_column!r} contains non-finite values.")
            if not np.all(np.isfinite(merged[column].to_numpy(dtype=float))):
                raise ValueError(f"Merged trial timing column {column!r} contains non-finite values.")
            merged = merged.drop(columns=[left_name, right_name])
    overlapping_event_columns = sorted(
        {
            str(column[:-2])
            for column in merged.columns
            if str(column).startswith("events_") and str(column).endswith("_x")
        }
        & {
            str(column[:-2])
            for column in merged.columns
            if str(column).startswith("events_") and str(column).endswith("_y")
        }
    )
    for column in (
        "run_num",
        "trial_number",
        *overlapping_event_columns,
    ):
        if f"{column}_x" in merged.columns or f"{column}_y" in merged.columns:
            merged = _coalesce_column(merged, column)
    return merged.sort_values(["run_num", "onset", "duration"]).reset_index(drop=True)


def _confirmatory_cell_specs(
    *,
    rois: Sequence[SubjectROI],
    bands: Sequence[str],
    model_terms: Sequence[str],
    use_outcome_variance: bool,
    family: str = "confirmatory",
) -> List[CellSpec]:
    cells: List[CellSpec] = []
    for roi in rois:
        for band in bands:
            cells.append(
                CellSpec(
                    analysis_id=f"confirmatory__{roi.name}__{band}",
                    family=family,
                    roi=roi.name,
                    band=band,
                    predictor_column=f"eeg_{roi.name}_{band}",
                    outcome_column=f"bold_{roi.name}",
                    outcome_variance_column=(
                        f"boldvar_{roi.name}" if use_outcome_variance else None
                    ),
                    model_terms=tuple(model_terms),
                    categorical_terms=tuple(),
                )
            )
    return cells


def _local_expression_cell_specs(
    *,
    rois: Sequence[SubjectROI],
    bands: Sequence[str],
    bold_table: pd.DataFrame,
    model_terms: Sequence[str],
) -> List[CellSpec]:
    cells: List[CellSpec] = []
    for roi in rois:
        prefix = "bold_localexpr_"
        matching_columns = [
            str(column)
            for column in bold_table.columns
            if str(column).startswith(prefix) and str(column).endswith(f"_{roi.name}")
        ]
        for column in sorted(matching_columns):
            signature_name = column.replace(prefix, "", 1).rsplit(f"_{roi.name}", 1)[0]
            for band in bands:
                cells.append(
                    CellSpec(
                        analysis_id=f"local_expression__{signature_name}__{roi.name}__{band}",
                        family="secondary_local_expression",
                        roi=roi.name,
                        band=band,
                        predictor_column=f"eeg_{roi.name}_{band}",
                        outcome_column=column,
                        outcome_variance_column=None,
                        model_terms=tuple(model_terms),
                        categorical_terms=tuple(),
                    )
                )
    return cells


def _write_roi_manifest(
    *,
    rois: Sequence[SubjectROI],
    path: Path,
) -> None:
    rows = [
        {
            "roi": roi.name,
            "n_source_vertices": len(roi.source_rows),
            "source_total_area": float(np.sum(np.asarray(roi.source_weights, dtype=float))),
            "n_lh_vertices": len(roi.lh_vertices),
            "n_rh_vertices": len(roi.rh_vertices),
        }
        for roi in rois
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _resolve_sensitivity_output_dir(
    *,
    subject_output_dir: Path,
    sensitivity_name: str,
) -> Path:
    return subject_output_dir / "sensitivities" / _safe_slug(sensitivity_name)


def _resolve_secondary_output_dir(
    *,
    subject_output_dir: Path,
    secondary_name: str,
) -> Path:
    return subject_output_dir / "secondary" / _safe_slug(secondary_name)


def _resolve_negative_control_output_dir(
    *,
    subject_output_dir: Path,
    control_name: str,
) -> Path:
    return subject_output_dir / "negative_controls" / _safe_slug(control_name)


def _keep_trial_mask(table: pd.DataFrame) -> np.ndarray:
    if "keep_trial" not in table.columns:
        return np.ones(len(table), dtype=bool)
    series = table["keep_trial"]
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes"}).to_numpy(dtype=bool)


def _apply_qc_status_to_cells(
    *,
    analysis_cells: pd.DataFrame,
    qc_result: Any,
) -> pd.DataFrame:
    out = analysis_cells.copy()
    if out.empty:
        return out
    if qc_result.global_failure:
        out["status"] = "fmri_qc_failed"
        return out
    if qc_result.non_interpretable_rois:
        mask = out["roi"].astype(str).isin(sorted(qc_result.non_interpretable_rois))
        out.loc[mask, "status"] = "fmri_qc_failed"
    return out


def _write_analysis_bundle(
    *,
    output_dir: Path,
    merged_table: pd.DataFrame,
    analysis_cells: pd.DataFrame,
    extra_tables: Optional[Mapping[str, pd.DataFrame]] = None,
    extra_json: Optional[Mapping[str, Any]] = None,
) -> None:
    ensure_dir(output_dir)
    merged_table.to_csv(output_dir / "trialwise_merged.tsv", sep="\t", index=False)
    analysis_cells.to_csv(output_dir / "analysis_cells.tsv", sep="\t", index=False)
    if extra_tables is not None:
        for name, table in extra_tables.items():
            table.to_csv(output_dir / f"{name}.tsv", sep="\t", index=False)
    if extra_json is not None:
        for name, payload in extra_json.items():
            (output_dir / f"{name}.json").write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )


def _merge_nested_mapping(
    *,
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        str(key): value
        for key, value in base.items()
    }
    for key, value in overrides.items():
        key_text = str(key)
        if isinstance(value, Mapping):
            existing = merged.get(key_text, {})
            if not isinstance(existing, Mapping):
                existing = {}
            merged[key_text] = _merge_nested_mapping(
                base=dict(existing),
                overrides=value,
            )
            continue
        merged[key_text] = value
    return merged


def _build_overridden_nuisance_config(
    *,
    base_nuisance_cfg: CouplingNuisanceConfig,
    nuisance_overrides: Mapping[str, Any],
    active_window: Tuple[float, float],
    epoch_window: Tuple[float, float],
    default_hrf_model: str,
) -> CouplingNuisanceConfig:
    merged_nuisance = _merge_nested_mapping(
        base=asdict(base_nuisance_cfg),
        overrides=nuisance_overrides,
    )
    return CouplingNuisanceConfig.from_config(
        {"eeg_bold_coupling": {"nuisance": merged_nuisance}},
        active_window=active_window,
        epoch_window=epoch_window,
        default_hrf_model=default_hrf_model,
    )


def _build_alternative_roi_specs(
    *,
    roi_items: Sequence[Mapping[str, Any]],
    default_template_subject: str,
) -> Tuple[CouplingROIConfig, ...]:
    return tuple(
        CouplingROIConfig.from_mapping(
            raw=item,
            default_template_subject=default_template_subject,
        )
        for item in roi_items
    )


def _build_sensitivity_cell_specs(
    *,
    base_cells: Sequence[CellSpec],
    model_terms: Sequence[str],
    categorical_terms: Sequence[str] = (),
    outcome_suffix_map: Optional[Mapping[str, str]] = None,
    available_columns: Optional[Sequence[str]] = None,
    family: str = "sensitivity",
    predictor_column_map: Optional[Mapping[str, str]] = None,
    nonstandardized_terms_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[CellSpec]:
    cells: List[CellSpec] = []
    available = None if available_columns is None else set(str(column) for column in available_columns)
    for cell in base_cells:
        outcome_column = cell.outcome_column
        if outcome_suffix_map is not None and outcome_column in outcome_suffix_map:
            outcome_column = outcome_suffix_map[outcome_column]
        predictor_column = cell.predictor_column
        if predictor_column_map is not None and predictor_column in predictor_column_map:
            predictor_column = str(predictor_column_map[predictor_column])
        outcome_variance_column = cell.outcome_variance_column
        if available is not None and outcome_variance_column is not None:
            if outcome_variance_column not in available:
                outcome_variance_column = None
        nonstandardized_terms: Tuple[str, ...] = tuple()
        if nonstandardized_terms_map is not None:
            resolved = nonstandardized_terms_map.get(cell.predictor_column)
            if resolved is not None:
                nonstandardized_terms = tuple(str(term) for term in resolved)
        cells.append(
            CellSpec(
                analysis_id=cell.analysis_id,
                family=family,
                roi=cell.roi,
                band=cell.band,
                predictor_column=predictor_column,
                outcome_column=outcome_column,
                outcome_variance_column=outcome_variance_column,
                model_terms=tuple(model_terms),
                categorical_terms=tuple(categorical_terms),
                nonstandardized_terms=nonstandardized_terms,
            )
        )
    return cells


def _within_between_predictor_columns(
    *,
    predictor_column: str,
) -> Tuple[str, str]:
    return (
        f"{predictor_column}_within_subject",
        f"{predictor_column}_subject_mean",
    )


def _add_within_between_predictors(
    *,
    merged_table: pd.DataFrame,
    base_cells: Sequence[CellSpec],
) -> pd.DataFrame:
    out = merged_table.copy()
    for cell in base_cells:
        predictor_column = str(cell.predictor_column)
        if predictor_column not in out.columns:
            raise ValueError(
                f"Within-between sensitivity is missing predictor column {predictor_column!r}."
            )
        predictor_values = pd.to_numeric(out[predictor_column], errors="coerce")
        if not np.all(np.isfinite(predictor_values.to_numpy(dtype=float))):
            raise ValueError(
                f"Within-between sensitivity predictor {predictor_column!r} contains non-finite values."
            )
        mean_column, subject_mean_column = _within_between_predictor_columns(
            predictor_column=predictor_column
        )
        subject_mean = float(np.mean(predictor_values.to_numpy(dtype=float)))
        out[subject_mean_column] = subject_mean
        out[mean_column] = predictor_values.to_numpy(dtype=float) - subject_mean
    return out


def _make_temperature_factor(
    *,
    merged_table: pd.DataFrame,
    temperature_column: str,
    factor_column: str,
    max_levels: Optional[int],
) -> pd.DataFrame:
    if temperature_column not in merged_table.columns:
        raise ValueError(
            f"Temperature-categorical sensitivity is missing {temperature_column!r}."
        )
    out = merged_table.copy()
    values = pd.to_numeric(out[temperature_column], errors="coerce")
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError("Temperature-categorical sensitivity temperature values are non-finite.")
    unique_levels = sorted(values.unique().tolist())
    if max_levels is not None and len(unique_levels) > int(max_levels):
        raise ValueError(
            f"Temperature-categorical sensitivity expected <= {max_levels} levels, found {len(unique_levels)}."
        )
    out[factor_column] = values.map(lambda v: f"{float(v):g}")
    return out


def _bold_concordance_table(
    *,
    primary_bold_table: pd.DataFrame,
    alternative_bold_table: pd.DataFrame,
) -> pd.DataFrame:
    merged = primary_bold_table.merge(
        alternative_bold_table,
        on="trial_key",
        how="inner",
        suffixes=("_lss", "_alt"),
        validate="one_to_one",
    )
    rows: List[Dict[str, Any]] = []
    for column in primary_bold_table.columns:
        if column == "trial_key" or not str(column).startswith("bold_"):
            continue
        left = pd.to_numeric(merged[f"{column}_lss"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_alt"], errors="coerce")
        mask = np.isfinite(left.to_numpy(dtype=float)) & np.isfinite(right.to_numpy(dtype=float))
        if not np.any(mask):
            continue
        corr = np.corrcoef(left.to_numpy(dtype=float)[mask], right.to_numpy(dtype=float)[mask])[0, 1]
        rows.append(
            {
                "outcome_column": column,
                "n_trials": int(mask.sum()),
                "pearson_r": float(corr),
            }
        )
    return pd.DataFrame(rows)


def _subject_negative_control_seed(
    *,
    subject: str,
    config: Any,
) -> int:
    base_seed = int(get_config_value(config, "project.random_state", 42))
    subject_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(subject)))
    return int(base_seed + subject_offset)


def _shuffle_bold_table_within_run(
    *,
    subject: str,
    trial_table: pd.DataFrame,
    bold_table: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    trial_lookup = trial_table[["trial_key", "run_num"]].copy()
    annotated = trial_lookup.merge(
        bold_table,
        on="trial_key",
        how="inner",
        validate="one_to_one",
    )
    if annotated.empty:
        raise ValueError("Negative-control trial shuffling found no matched BOLD trials.")
    rng = np.random.default_rng(
        _subject_negative_control_seed(subject=subject, config=config)
    )
    shuffled_parts: List[pd.DataFrame] = []
    for run_num, run_table in annotated.groupby("run_num", sort=True):
        out = run_table.copy().reset_index(drop=True)
        if len(out) < 2:
            raise ValueError(
                f"Negative-control trial shuffling requires at least two trials in run {int(run_num)}."
            )
        shuffled_keys = rng.permutation(out["trial_key"].to_numpy(copy=True))
        out["trial_key"] = shuffled_keys
        shuffled_parts.append(out.drop(columns=["run_num"]))
    shuffled = pd.concat(shuffled_parts, axis=0, ignore_index=True)
    if shuffled["trial_key"].duplicated().any():
        raise ValueError("Negative-control trial shuffling produced duplicate trial keys.")
    return shuffled


def _run_subject_negative_controls(
    *,
    subject: str,
    config: Any,
    subject_output_dir: Path,
    trial_table: pd.DataFrame,
    eeg_table: pd.DataFrame,
    bold_table: pd.DataFrame,
    primary_cells: Sequence[CellSpec],
    coupling_cfg: EEGBOLDCouplingConfig,
    nuisance_cfg: CouplingNuisanceConfig,
    fmri_qc: Any,
) -> None:
    trial_shuffle_cfg = TrialShuffleNegativeControlConfig.from_config(config)
    if not trial_shuffle_cfg.enabled:
        return

    shuffled_bold = _shuffle_bold_table_within_run(
        subject=subject,
        trial_table=trial_table,
        bold_table=bold_table,
        config=config,
    )
    shuffled_merged = _merge_trialwise_tables(
        eeg_table=eeg_table,
        trial_table=trial_table,
        bold_table=shuffled_bold,
    )
    _validate_model_terms_present(
        table=shuffled_merged,
        terms=coupling_cfg.covariates.model_terms,
    )
    shuffled_kept, shuffled_qc, _shuffled_summary = apply_trial_censoring(
        merged_table=shuffled_merged,
        model_terms=coupling_cfg.covariates.model_terms,
        nuisance_cfg=nuisance_cfg,
    )
    shuffled_cells = summarize_subject_cells(
        subject=subject,
        merged_table=shuffled_kept,
        cell_specs=_build_sensitivity_cell_specs(
            base_cells=primary_cells,
            model_terms=coupling_cfg.covariates.model_terms,
            available_columns=shuffled_kept.columns,
            family="negative_control_trial_shuffle",
        ),
        stats_cfg=coupling_cfg.statistics,
    )
    out_dir = _resolve_negative_control_output_dir(
        subject_output_dir=subject_output_dir,
        control_name=trial_shuffle_cfg.output_name,
    )
    _write_analysis_bundle(
        output_dir=out_dir,
        merged_table=shuffled_qc,
        analysis_cells=_apply_qc_status_to_cells(
            analysis_cells=shuffled_cells,
            qc_result=fmri_qc,
        ),
        extra_tables={"trialwise_bold": shuffled_bold},
    )


def _run_subject_sensitivity_analyses(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    subject_output_dir: Path,
    epochs: Any,
    clean_events: pd.DataFrame,
    fwd: Any,
    surface_info: SubjectSurfaceInfo,
    rois: Sequence[SubjectROI],
    trial_table: pd.DataFrame,
    eeg_table_base: pd.DataFrame,
    eeg_table: pd.DataFrame,
    merged_kept: pd.DataFrame,
    primary_bold_table: pd.DataFrame,
    primary_cells: Sequence[CellSpec],
    coupling_cfg: EEGBOLDCouplingConfig,
    nuisance_cfg: CouplingNuisanceConfig,
    signature_weights: Optional[Mapping[str, Dict[str, np.ndarray]]],
    expression_cfg: LocalSignatureExpressionConfig,
    fmri_qc: Any,
) -> None:
    sensitivity_cfg = CouplingSensitivityConfig.from_config(config)
    stats_cfg = coupling_cfg.statistics

    if sensitivity_cfg.anatomical_specificity.enabled:
        frequency_bands = get_config_value(config, "frequency_bands", {})
        times = np.asarray(epochs.times, dtype=float)
        active_mask = _build_window_mask(
            times=times,
            window=coupling_cfg.eeg.active_window,
            label="EEG active",
        )
        baseline_mask = _build_window_mask(
            times=times,
            window=coupling_cfg.eeg.baseline_window,
            label="EEG baseline",
        )
        for item in sensitivity_cfg.anatomical_specificity.items:
            alt_specs = _build_alternative_roi_specs(
                roi_items=item.rois,
                default_template_subject=str(
                    get_config_value(
                        config,
                        "eeg_bold_coupling.rois.template_subject",
                        "fsaverage",
                    )
                ).strip(),
            )
            alt_labels = _build_subject_labels_from_specs(
                subject=_subject_bids(subject),
                specs=alt_specs,
                subjects_dir=coupling_cfg.eeg.subjects_dir,
                logger=LOGGER,
            )
            alt_rois = _map_rois_to_source_rows(
                source_vertices=_source_vertices_from_forward(fwd),
                labels=alt_labels,
                surface_info=surface_info,
            )
            alt_eeg_features = _extract_trialwise_eeg_table(
                epochs=epochs,
                clean_events=clean_events,
                fwd=fwd,
                rois=alt_rois,
                cfg=coupling_cfg,
                frequency_bands=frequency_bands,
                active_mask=active_mask,
                baseline_mask=baseline_mask,
                logger=LOGGER,
            )
            alt_eeg_table = compute_eeg_artifact_table(
                epochs=epochs,
                clean_events=clean_events,
                eeg_table=alt_eeg_features,
                nuisance_cfg=nuisance_cfg,
            )
            alt_bold_table = _extract_trialwise_bold_features_surface_glm(
                subject=subject,
                task=task,
                trial_table=trial_table,
                rois=alt_rois,
                surface_info=surface_info,
                config=config,
                coupling_cfg=coupling_cfg,
                require_variance=coupling_cfg.statistics.use_outcome_variance,
                signature_weights=signature_weights,
                expression_cfg=expression_cfg if expression_cfg.enabled else None,
                logger=LOGGER,
            )
            alt_merged = _merge_trialwise_tables(
                eeg_table=alt_eeg_table,
                trial_table=trial_table,
                bold_table=alt_bold_table,
            )
            _validate_model_terms_present(
                table=alt_merged,
                terms=coupling_cfg.covariates.model_terms,
            )
            alt_kept, alt_qc_table, _alt_qc_summary = apply_trial_censoring(
                merged_table=alt_merged,
                model_terms=coupling_cfg.covariates.model_terms,
                nuisance_cfg=nuisance_cfg,
            )
            alt_cells = summarize_subject_cells(
                subject=subject,
                merged_table=alt_kept,
                cell_specs=_confirmatory_cell_specs(
                    rois=alt_rois,
                    bands=coupling_cfg.eeg.bands,
                    model_terms=coupling_cfg.covariates.model_terms,
                    use_outcome_variance=stats_cfg.use_outcome_variance,
                    family="sensitivity_anatomical_specificity",
                ),
                stats_cfg=stats_cfg,
            )
            out_dir = _resolve_sensitivity_output_dir(
                subject_output_dir=subject_output_dir,
                sensitivity_name=item.name,
            )
            _write_analysis_bundle(
                output_dir=out_dir,
                merged_table=alt_qc_table,
                analysis_cells=_apply_qc_status_to_cells(
                    analysis_cells=alt_cells,
                    qc_result=fmri_qc,
                ),
                extra_tables={
                    "trialwise_eeg": alt_eeg_table,
                    "trialwise_bold": alt_bold_table,
                },
            )

    if sensitivity_cfg.artifact_models.enabled:
        for item in sensitivity_cfg.artifact_models.items:
            alt_nuisance_cfg = _build_overridden_nuisance_config(
                base_nuisance_cfg=nuisance_cfg,
                nuisance_overrides=item.nuisance_overrides,
                active_window=coupling_cfg.eeg.active_window,
                epoch_window=(float(epochs.tmin), float(epochs.tmax)),
                default_hrf_model=coupling_cfg.fmri.hrf_model,
            )
            alt_eeg_table = compute_eeg_artifact_table(
                epochs=epochs,
                clean_events=clean_events,
                eeg_table=eeg_table_base,
                nuisance_cfg=alt_nuisance_cfg,
            )
            alt_merged = _merge_trialwise_tables(
                eeg_table=alt_eeg_table,
                trial_table=trial_table,
                bold_table=primary_bold_table,
            )
            _validate_model_terms_present(
                table=alt_merged,
                terms=coupling_cfg.covariates.model_terms,
            )
            alt_kept, alt_qc_table, _alt_qc_summary = apply_trial_censoring(
                merged_table=alt_merged,
                model_terms=coupling_cfg.covariates.model_terms,
                nuisance_cfg=alt_nuisance_cfg,
            )
            alt_cells = summarize_subject_cells(
                subject=subject,
                merged_table=alt_kept,
                cell_specs=_build_sensitivity_cell_specs(
                    base_cells=primary_cells,
                    model_terms=coupling_cfg.covariates.model_terms,
                    available_columns=alt_kept.columns,
                    family="sensitivity_artifact_model",
                ),
                stats_cfg=stats_cfg,
            )
            out_dir = _resolve_sensitivity_output_dir(
                subject_output_dir=subject_output_dir,
                sensitivity_name=item.name,
            )
            _write_analysis_bundle(
                output_dir=out_dir,
                merged_table=alt_qc_table,
                analysis_cells=_apply_qc_status_to_cells(
                    analysis_cells=alt_cells,
                    qc_result=fmri_qc,
                ),
                extra_tables={"trialwise_eeg": alt_eeg_table},
                extra_json={"nuisance_config": asdict(alt_nuisance_cfg)},
            )

    if sensitivity_cfg.within_between.enabled:
        within_between_merged = _add_within_between_predictors(
            merged_table=merged_kept,
            base_cells=primary_cells,
        )
        within_between_specs: List[CellSpec] = []
        for cell in primary_cells:
            within_column, between_column = _within_between_predictor_columns(
                predictor_column=cell.predictor_column
            )
            within_between_specs.append(
                CellSpec(
                    analysis_id=cell.analysis_id,
                    family="sensitivity_within_between",
                    roi=cell.roi,
                    band=cell.band,
                    predictor_column=within_column,
                    outcome_column=cell.outcome_column,
                    outcome_variance_column=cell.outcome_variance_column,
                    model_terms=tuple([*coupling_cfg.covariates.model_terms, between_column]),
                    categorical_terms=tuple(),
                    nonstandardized_terms=(between_column,),
                )
            )
        within_between_cells = summarize_subject_cells(
            subject=subject,
            merged_table=within_between_merged,
            cell_specs=within_between_specs,
            stats_cfg=stats_cfg,
        )
        out_dir = _resolve_sensitivity_output_dir(
            subject_output_dir=subject_output_dir,
            sensitivity_name=sensitivity_cfg.within_between.output_name,
        )
        _write_analysis_bundle(
            output_dir=out_dir,
            merged_table=within_between_merged,
            analysis_cells=_apply_qc_status_to_cells(
                analysis_cells=within_between_cells,
                qc_result=fmri_qc,
            ),
        )

    if sensitivity_cfg.painful_only.enabled:
        painful_merged = filter_painful_trials(
            merged_table=merged_kept,
            cfg=sensitivity_cfg.painful_only,
        )
        painful_cells = summarize_subject_cells(
            subject=subject,
            merged_table=painful_merged,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=coupling_cfg.covariates.model_terms,
                available_columns=painful_merged.columns,
                family="sensitivity_painful_only",
            ),
            stats_cfg=stats_cfg,
        )
        out_dir = _resolve_sensitivity_output_dir(
            subject_output_dir=subject_output_dir,
            sensitivity_name=sensitivity_cfg.painful_only.output_name,
        )
        _write_analysis_bundle(
            output_dir=out_dir,
            merged_table=painful_merged,
            analysis_cells=_apply_qc_status_to_cells(
                analysis_cells=painful_cells,
                qc_result=fmri_qc,
            ),
        )

    if sensitivity_cfg.alternative_fmri.enabled:
        beta_root = resolve_sensitivity_beta_dir(
            template=str(sensitivity_cfg.alternative_fmri.beta_dir_template),
            deriv_root=deriv_root,
            subject=subject,
            task=task,
            contrast_name=coupling_cfg.fmri.contrast_name,
        )
        matched_trial_table = trial_table.loc[
            trial_table["trial_key"].isin(set(eeg_table["trial_key"]))
        ].reset_index(drop=True)
        alt_bold_table = _extract_trialwise_bold_features(
            subject=subject,
            task=task,
            beta_root=beta_root,
            trial_table=matched_trial_table,
            rois=rois,
            subjects_dir=coupling_cfg.eeg.subjects_dir,
            require_variance=coupling_cfg.statistics.use_outcome_variance,
            signature_weights=signature_weights,
            expression_cfg=expression_cfg if expression_cfg.enabled else None,
        )
        alt_merged = _merge_trialwise_tables(
            eeg_table=eeg_table,
            trial_table=matched_trial_table,
            bold_table=alt_bold_table,
        )
        _validate_model_terms_present(
            table=alt_merged,
            terms=coupling_cfg.covariates.model_terms,
        )
        alt_merged_kept, alt_qc_table, _alt_qc_summary = apply_trial_censoring(
            merged_table=alt_merged,
            model_terms=coupling_cfg.covariates.model_terms,
            nuisance_cfg=nuisance_cfg,
        )
        alt_cells = summarize_subject_cells(
            subject=subject,
            merged_table=alt_merged_kept,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=coupling_cfg.covariates.model_terms,
                available_columns=alt_merged_kept.columns,
                family="sensitivity_alternative_fmri",
            ),
            stats_cfg=stats_cfg,
        )
        concordance = _bold_concordance_table(
            primary_bold_table=primary_bold_table,
            alternative_bold_table=alt_bold_table,
        )
        out_dir = _resolve_sensitivity_output_dir(
            subject_output_dir=subject_output_dir,
            sensitivity_name=sensitivity_cfg.alternative_fmri.name,
        )
        _write_analysis_bundle(
            output_dir=out_dir,
            merged_table=alt_qc_table,
            analysis_cells=_apply_qc_status_to_cells(
                analysis_cells=alt_cells,
                qc_result=fmri_qc,
            ),
            extra_tables={
                "trialwise_bold": alt_bold_table,
                "lss_vs_alternative_bold_concordance": concordance,
            },
        )

    if sensitivity_cfg.delta_temperature.enabled:
        delta_terms = tuple(sensitivity_cfg.delta_temperature.model_terms)
        _validate_model_terms_present(table=merged_kept, terms=delta_terms)
        delta_cells = summarize_subject_cells(
            subject=subject,
            merged_table=merged_kept,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=delta_terms,
                available_columns=merged_kept.columns,
                family="sensitivity_delta_temperature",
            ),
            stats_cfg=stats_cfg,
        )
        out_dir = _resolve_sensitivity_output_dir(
            subject_output_dir=subject_output_dir,
            sensitivity_name=sensitivity_cfg.delta_temperature.output_name,
        )
        _write_analysis_bundle(
            output_dir=out_dir,
            merged_table=merged_kept,
            analysis_cells=_apply_qc_status_to_cells(
                analysis_cells=delta_cells,
                qc_result=fmri_qc,
            ),
        )

    if sensitivity_cfg.temperature_categorical.enabled:
        categorical_merged = _make_temperature_factor(
            merged_table=merged_kept,
            temperature_column=sensitivity_cfg.temperature_categorical.temperature_column,
            factor_column=sensitivity_cfg.temperature_categorical.factor_column,
            max_levels=sensitivity_cfg.temperature_categorical.max_levels,
        )
        cat_terms = tuple(sensitivity_cfg.temperature_categorical.model_terms)
        _validate_model_terms_present(table=categorical_merged, terms=cat_terms)
        cat_cells = summarize_subject_cells(
            subject=subject,
            merged_table=categorical_merged,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=cat_terms,
                categorical_terms=(sensitivity_cfg.temperature_categorical.factor_column,),
                available_columns=categorical_merged.columns,
                family="sensitivity_temperature_categorical",
            ),
            stats_cfg=stats_cfg,
        )
        out_dir = _resolve_sensitivity_output_dir(
            subject_output_dir=subject_output_dir,
            sensitivity_name=sensitivity_cfg.temperature_categorical.output_name,
        )
        _write_analysis_bundle(
            output_dir=out_dir,
            merged_table=categorical_merged,
            analysis_cells=_apply_qc_status_to_cells(
                analysis_cells=cat_cells,
                qc_result=fmri_qc,
            ),
        )

    if sensitivity_cfg.residualized_correlation.enabled:
        residual_cells = summarize_subject_cells(
            subject=subject,
            merged_table=merged_kept,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=coupling_cfg.covariates.model_terms,
                available_columns=merged_kept.columns,
                family="sensitivity_residualized_correlation",
            ),
            stats_cfg=stats_cfg,
        )
        residual_cells = _apply_qc_status_to_cells(
            analysis_cells=residual_cells,
            qc_result=fmri_qc,
        )
        subject_effects = fit_subject_residualized_correlations(
            subject=subject,
            merged_table=merged_kept,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=coupling_cfg.covariates.model_terms,
                available_columns=merged_kept.columns,
                family="sensitivity_residualized_correlation",
            ),
            analysis_cells=residual_cells,
            include_run_fixed_effect=stats_cfg.include_run_fixed_effect,
        )
        out_dir = _resolve_sensitivity_output_dir(
            subject_output_dir=subject_output_dir,
            sensitivity_name=sensitivity_cfg.residualized_correlation.output_name,
        )
        _write_analysis_bundle(
            output_dir=out_dir,
            merged_table=merged_kept,
            analysis_cells=residual_cells,
            extra_tables={"subject_effects": subject_effects},
        )

    if sensitivity_cfg.primary_permutation.enabled:
        permutation_cells = summarize_subject_cells(
            subject=subject,
            merged_table=merged_kept,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=coupling_cfg.covariates.model_terms,
                available_columns=merged_kept.columns,
                family="sensitivity_primary_permutation",
            ),
            stats_cfg=stats_cfg,
        )
        permutation_cells = _apply_qc_status_to_cells(
            analysis_cells=permutation_cells,
            qc_result=fmri_qc,
        )
        permutation_effects = fit_subject_primary_permutation_effects(
            subject=subject,
            merged_table=merged_kept,
            cell_specs=_build_sensitivity_cell_specs(
                base_cells=primary_cells,
                model_terms=coupling_cfg.covariates.model_terms,
                available_columns=merged_kept.columns,
                family="sensitivity_primary_permutation",
            ),
            analysis_cells=permutation_cells,
            include_run_fixed_effect=stats_cfg.include_run_fixed_effect,
        )
        out_dir = _resolve_sensitivity_output_dir(
            subject_output_dir=subject_output_dir,
            sensitivity_name=sensitivity_cfg.primary_permutation.output_name,
        )
        _write_analysis_bundle(
            output_dir=out_dir,
            merged_table=merged_kept,
            analysis_cells=permutation_cells,
            extra_tables={"subject_effects": permutation_effects},
        )

    if sensitivity_cfg.source_methods.enabled:
        frequency_bands = get_config_value(config, "frequency_bands", {})
        epochs, clean_events = load_epochs_for_analysis(
            subject=subject,
            task=task,
            align="strict",
            preload=True,
            deriv_root=deriv_root,
            logger=LOGGER,
            config=config,
        )
        if epochs is None or clean_events is None:
            raise ValueError(f"Could not reload epochs/events for source-method sensitivity: sub-{subject}.")
        fwd, _src = _resolve_subject_source_model(
            subject=subject,
            epochs=epochs,
            config=config,
            coupling_cfg=coupling_cfg,
            logger=LOGGER,
        )
        times = np.asarray(epochs.times, dtype=float)
        active_mask = _build_window_mask(
            times=times,
            window=coupling_cfg.eeg.active_window,
            label="EEG active",
        )
        baseline_mask = _build_window_mask(
            times=times,
            window=coupling_cfg.eeg.baseline_window,
            label="EEG baseline",
        )
        for item in sensitivity_cfg.source_methods.items:
            if item.method == "lcmv":
                eeg_features = pd.concat(
                    [
                        _extract_trialwise_band_features_from_stc_stream(
                            stcs=iterate_band_specific_lcmv_estimates(
                                epochs=epochs,
                                fwd=fwd,
                                band=band,
                                frequency_bands=frequency_bands,
                                baseline_window=coupling_cfg.eeg.baseline_window,
                                reg=coupling_cfg.eeg.reg,
                                logger=LOGGER,
                            ),
                            rois=rois,
                            band=band,
                            active_mask=active_mask,
                            baseline_mask=baseline_mask,
                        )
                        for band in item.bands
                    ],
                    axis=1,
                )
            else:
                stcs = compute_source_estimates(
                    epochs=epochs,
                    fwd=fwd,
                    method=item.method,
                    baseline_window=coupling_cfg.eeg.baseline_window,
                    reg=coupling_cfg.eeg.reg,
                    loose=coupling_cfg.eeg.loose,
                    depth=coupling_cfg.eeg.depth,
                    snr=coupling_cfg.eeg.snr,
                    logger=LOGGER,
                )
                eeg_features = _extract_trialwise_eeg_features(
                    stcs=stcs,
                    rois=rois,
                    bands=item.bands,
                    frequency_bands=frequency_bands,
                    active_mask=active_mask,
                    baseline_mask=baseline_mask,
                    sfreq=float(epochs.info["sfreq"]),
                    feature_batch_size=coupling_cfg.eeg.feature_batch_size,
                )
            eeg_method_table = _prepare_eeg_trial_table(
                events_df=clean_events,
                eeg_features=eeg_features,
                alignment_cfg=coupling_cfg.alignment,
            )
            eeg_method_table = compute_eeg_artifact_table(
                epochs=epochs,
                clean_events=clean_events,
                eeg_table=eeg_method_table,
                nuisance_cfg=nuisance_cfg,
            )
            method_merged = _merge_trialwise_tables(
                eeg_table=eeg_method_table,
                trial_table=trial_table,
                bold_table=primary_bold_table,
            )
            _validate_model_terms_present(
                table=method_merged,
                terms=coupling_cfg.covariates.model_terms,
            )
            method_kept, method_qc, _method_summary = apply_trial_censoring(
                merged_table=method_merged,
                model_terms=coupling_cfg.covariates.model_terms,
                nuisance_cfg=nuisance_cfg,
            )
            method_cells = summarize_subject_cells(
                subject=subject,
                merged_table=method_kept,
                cell_specs=_confirmatory_cell_specs(
                    rois=rois,
                    bands=item.bands,
                    model_terms=coupling_cfg.covariates.model_terms,
                    use_outcome_variance=stats_cfg.use_outcome_variance,
                    family="sensitivity_source_method",
                ),
                stats_cfg=stats_cfg,
            )
            out_dir = _resolve_sensitivity_output_dir(
                subject_output_dir=subject_output_dir,
                sensitivity_name=item.name,
            )
            _write_analysis_bundle(
                output_dir=out_dir,
                merged_table=method_qc,
                analysis_cells=_apply_qc_status_to_cells(
                    analysis_cells=method_cells,
                    qc_result=fmri_qc,
                ),
            )


def _aggregate_analysis_dir(
    *,
    subjects: Sequence[str],
    task: str,
    deriv_root: Path,
    cfg: EEGBOLDCouplingConfig,
    relative_dir: Tuple[str, ...],
    output_root: Path,
    skip_analysis_names: Sequence[str] = (),
) -> None:
    skipped = {str(name).strip() for name in skip_analysis_names if str(name).strip()}
    analysis_names: set[str] = set()
    for subject in subjects:
        subject_dir = _resolve_subject_output_dir(
            deriv_root=deriv_root,
            subject=subject,
            task=task,
            cfg=cfg,
        )
        root_dir = subject_dir
        for part in relative_dir:
            root_dir = root_dir / part
        if not root_dir.exists():
            continue
        for path in root_dir.iterdir():
            if path.is_dir() and path.name not in skipped:
                analysis_names.add(path.name)

    if not analysis_names:
        return

    ensure_dir(output_root)
    for analysis_name in sorted(analysis_names):
        cell_frames: List[pd.DataFrame] = []
        pooled_frames: List[pd.DataFrame] = []
        for subject in subjects:
            subject_dir = _resolve_subject_output_dir(
                deriv_root=deriv_root,
                subject=subject,
                task=task,
                cfg=cfg,
            )
            bundle_dir = subject_dir
            for part in relative_dir:
                bundle_dir = bundle_dir / part
            bundle_dir = bundle_dir / analysis_name
            cells_path = bundle_dir / "analysis_cells.tsv"
            merged_path = bundle_dir / "trialwise_merged.tsv"
            if not cells_path.exists() or not merged_path.exists():
                continue
            cell_frame = pd.read_csv(cells_path, sep="\t", dtype={"subject": str})
            merged_frame = pd.read_csv(merged_path, sep="\t")
            if cell_frame.empty or merged_frame.empty:
                continue
            cell_frames.append(cell_frame)
            merged_frame = merged_frame.loc[_keep_trial_mask(merged_frame)].reset_index(drop=True)
            if merged_frame.empty:
                continue
            merged_frame["subject"] = str(subject).replace("sub-", "", 1)
            pooled_frames.append(merged_frame)
        if not cell_frames or not pooled_frames:
            continue
        analysis_cells = pd.concat(cell_frames, axis=0, ignore_index=True)
        pooled = pd.concat(pooled_frames, axis=0, ignore_index=True)
        rows: List[Dict[str, Any]] = []
        grouped_cells = analysis_cells.drop_duplicates(
            subset=[
                "analysis_id",
                "family",
                "roi",
                "band",
                "predictor_column",
                "outcome_column",
                "outcome_variance_column",
                "nonstandardized_terms",
            ]
        )
        for cell_row in grouped_cells.itertuples(index=False):
            ok_subjects = analysis_cells.loc[
                (analysis_cells["analysis_id"].astype(str) == str(cell_row.analysis_id))
                & (analysis_cells["status"].astype(str) == "ok"),
                "subject",
            ].astype(str)
            pooled_subset = pooled.loc[
                pooled["subject"].astype(str).isin(ok_subjects.tolist())
            ].reset_index(drop=True)
            if pooled_subset.empty:
                continue
            model_terms = tuple(json.loads(str(cell_row.model_terms)))
            categorical_terms = tuple(json.loads(str(cell_row.categorical_terms)))
            row = fit_mixedlm_cell(
                pooled_table=pooled_subset,
                cell=CellSpec(
                    analysis_id=str(cell_row.analysis_id),
                    family=str(cell_row.family),
                    roi=str(cell_row.roi),
                    band=str(cell_row.band),
                    predictor_column=str(cell_row.predictor_column),
                    outcome_column=str(cell_row.outcome_column),
                    outcome_variance_column=(
                        None
                        if str(getattr(cell_row, "outcome_variance_column", "")).strip() == ""
                        else str(getattr(cell_row, "outcome_variance_column"))
                    ),
                    model_terms=model_terms,
                    categorical_terms=categorical_terms,
                    nonstandardized_terms=tuple(
                        json.loads(str(getattr(cell_row, "nonstandardized_terms", "[]")))
                    ),
                ),
                stats_cfg=cfg.statistics,
            )
            row["model_terms"] = list(model_terms)
            row["categorical_terms"] = list(categorical_terms)
            row["nonstandardized_terms"] = list(
                json.loads(str(getattr(cell_row, "nonstandardized_terms", "[]")))
            )
            rows.append(row)
        group_results = finalize_group_results(rows=rows, alpha=cfg.statistics.alpha)
        analysis_group_dir = output_root / analysis_name
        ensure_dir(analysis_group_dir)
        analysis_cells.to_csv(
            analysis_group_dir / "analysis_cells_all.tsv",
            sep="\t",
            index=False,
        )
        if not group_results.empty:
            group_results.to_csv(
                analysis_group_dir / "group_results.tsv",
                sep="\t",
                index=False,
            )


def _aggregate_residualized_sensitivity_dir(
    *,
    subjects: Sequence[str],
    task: str,
    deriv_root: Path,
    cfg: EEGBOLDCouplingConfig,
    sensitivity_name: str,
    output_root: Path,
    config: Any,
) -> None:
    analysis_name = _safe_slug(sensitivity_name)
    cell_frames: List[pd.DataFrame] = []
    effect_frames: List[pd.DataFrame] = []
    for subject in subjects:
        subject_dir = _resolve_subject_output_dir(
            deriv_root=deriv_root,
            subject=subject,
            task=task,
            cfg=cfg,
        )
        bundle_dir = subject_dir / "sensitivities" / analysis_name
        cells_path = bundle_dir / "analysis_cells.tsv"
        effects_path = bundle_dir / "subject_effects.tsv"
        if not cells_path.exists() or not effects_path.exists():
            continue
        cell_frame = pd.read_csv(cells_path, sep="\t", dtype={"subject": str})
        effect_frame = pd.read_csv(effects_path, sep="\t", dtype={"subject": str})
        if cell_frame.empty or effect_frame.empty:
            continue
        cell_frames.append(cell_frame)
        effect_frames.append(effect_frame)
    if not effect_frames:
        return
    sensitivity_cfg = CouplingSensitivityConfig.from_config(config)
    random_state = int(get_config_value(config, "project.random_state", 42))
    group_results = aggregate_residualized_correlations(
        subject_effects=pd.concat(effect_frames, axis=0, ignore_index=True),
        cfg=sensitivity_cfg.residualized_correlation,
        alpha=cfg.statistics.alpha,
        random_state=random_state,
    )
    analysis_group_dir = output_root / analysis_name
    ensure_dir(analysis_group_dir)
    if cell_frames:
        pd.concat(cell_frames, axis=0, ignore_index=True).to_csv(
            analysis_group_dir / "analysis_cells_all.tsv",
            sep="\t",
            index=False,
        )
    pd.concat(effect_frames, axis=0, ignore_index=True).to_csv(
        analysis_group_dir / "subject_effects_all.tsv",
        sep="\t",
        index=False,
    )
    if not group_results.empty:
        group_results.to_csv(
            analysis_group_dir / "group_results.tsv",
            sep="\t",
            index=False,
        )


def _aggregate_primary_permutation_sensitivity_dir(
    *,
    subjects: Sequence[str],
    task: str,
    deriv_root: Path,
    cfg: EEGBOLDCouplingConfig,
    sensitivity_name: str,
    output_root: Path,
    config: Any,
) -> None:
    analysis_name = _safe_slug(sensitivity_name)
    cell_frames: List[pd.DataFrame] = []
    effect_frames: List[pd.DataFrame] = []
    for subject in subjects:
        subject_dir = _resolve_subject_output_dir(
            deriv_root=deriv_root,
            subject=subject,
            task=task,
            cfg=cfg,
        )
        bundle_dir = subject_dir / "sensitivities" / analysis_name
        cells_path = bundle_dir / "analysis_cells.tsv"
        effects_path = bundle_dir / "subject_effects.tsv"
        if not cells_path.exists() or not effects_path.exists():
            continue
        cell_frame = pd.read_csv(cells_path, sep="\t", dtype={"subject": str})
        effect_frame = pd.read_csv(effects_path, sep="\t", dtype={"subject": str})
        if cell_frame.empty or effect_frame.empty:
            continue
        cell_frames.append(cell_frame)
        effect_frames.append(effect_frame)
    if not effect_frames:
        return
    sensitivity_cfg = CouplingSensitivityConfig.from_config(config)
    random_state = int(get_config_value(config, "project.random_state", 42))
    group_results = aggregate_primary_permutation_effects(
        subject_effects=pd.concat(effect_frames, axis=0, ignore_index=True),
        cfg=sensitivity_cfg.primary_permutation,
        alpha=cfg.statistics.alpha,
        random_state=random_state,
    )
    analysis_group_dir = output_root / analysis_name
    ensure_dir(analysis_group_dir)
    if cell_frames:
        pd.concat(cell_frames, axis=0, ignore_index=True).to_csv(
            analysis_group_dir / "analysis_cells_all.tsv",
            sep="\t",
            index=False,
        )
    pd.concat(effect_frames, axis=0, ignore_index=True).to_csv(
        analysis_group_dir / "subject_effects_all.tsv",
        sep="\t",
        index=False,
    )
    if not group_results.empty:
        group_results.to_csv(
            analysis_group_dir / "group_results.tsv",
            sep="\t",
            index=False,
        )


def _cellspec_from_analysis_row(cell_row: Any) -> CellSpec:
    return CellSpec(
        analysis_id=str(cell_row.analysis_id),
        family=str(cell_row.family),
        roi=str(cell_row.roi),
        band=str(cell_row.band),
        predictor_column=str(cell_row.predictor_column),
        outcome_column=str(cell_row.outcome_column),
        outcome_variance_column=(
            None
            if str(getattr(cell_row, "outcome_variance_column", "")).strip() == ""
            else str(getattr(cell_row, "outcome_variance_column"))
        ),
        model_terms=tuple(json.loads(str(cell_row.model_terms))),
        categorical_terms=tuple(json.loads(str(cell_row.categorical_terms))),
        nonstandardized_terms=tuple(
            json.loads(str(getattr(cell_row, "nonstandardized_terms", "[]")))
        ),
    )


def _group_confirmatory_cell_rows(analysis_cells: pd.DataFrame) -> pd.DataFrame:
    return (
        analysis_cells.loc[
            analysis_cells["family"].astype(str) == "confirmatory"
        ]
        .drop_duplicates(
            subset=[
                "analysis_id",
                "family",
                "roi",
                "band",
                "predictor_column",
                "outcome_column",
                "outcome_variance_column",
                "nonstandardized_terms",
            ]
        )
        .reset_index(drop=True)
    )


def _confirmatory_pooled_subset(
    *,
    pooled: pd.DataFrame,
    analysis_cells: pd.DataFrame,
    analysis_id: str,
) -> pd.DataFrame:
    ok_subjects = analysis_cells.loc[
        (analysis_cells["analysis_id"].astype(str) == str(analysis_id))
        & (analysis_cells["status"].astype(str) == "ok"),
        "subject",
    ].astype(str)
    return pooled.loc[
        pooled["subject"].astype(str).isin(ok_subjects.tolist())
    ].reset_index(drop=True)


def _leave_one_out_refits(
    *,
    pooled_subset: pd.DataFrame,
    cell: CellSpec,
    stats_cfg: CouplingStatisticsConfig,
    refit_type: str,
    holdout_values: Sequence[Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for holdout_value in holdout_values:
        if refit_type == "subject":
            refit_table = pooled_subset.loc[
                pooled_subset["subject"].astype(str) != str(holdout_value)
            ].reset_index(drop=True)
        elif refit_type == "run":
            refit_table = pooled_subset.loc[
                pd.to_numeric(pooled_subset["run_num"], errors="coerce")
                != float(holdout_value)
            ].reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported refit_type {refit_type!r}.")
        result = fit_mixedlm_cell(
            pooled_table=refit_table,
            cell=cell,
            stats_cfg=stats_cfg,
        )
        rows.append(
            {
                "refit_type": refit_type,
                "holdout": str(holdout_value),
                **result,
            }
        )
    return pd.DataFrame(rows)


def _summarize_leave_one_out_refits(
    *,
    detail: pd.DataFrame,
    reference_results: pd.DataFrame,
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    reference_map = {
        str(row.analysis_id): row
        for row in reference_results.itertuples(index=False)
        if str(row.family) == "confirmatory"
    }
    rows: List[Dict[str, Any]] = []
    grouped = detail.groupby(["analysis_id", "refit_type"], sort=True)
    for (analysis_id, refit_type), group in grouped:
        reference = reference_map.get(str(analysis_id))
        if reference is None:
            continue
        beta_values = pd.to_numeric(group["beta"], errors="coerce").to_numpy(dtype=float)
        interpretable_mask = group["interpretable"].astype(bool).to_numpy(dtype=bool)
        interpretable_betas = beta_values[interpretable_mask]
        reference_beta = float(getattr(reference, "beta"))
        if np.isfinite(reference_beta) and reference_beta != 0.0 and interpretable_betas.size > 0:
            sign_flips = int(
                np.sum(np.sign(interpretable_betas) != np.sign(reference_beta))
            )
        else:
            sign_flips = 0
        max_abs_beta_delta = np.nan
        if np.isfinite(reference_beta) and interpretable_betas.size > 0:
            max_abs_beta_delta = float(
                np.max(np.abs(interpretable_betas - reference_beta))
            )
        rows.append(
            {
                "analysis_id": str(analysis_id),
                "refit_type": str(refit_type),
                "reference_beta": reference_beta,
                "reference_p_value": float(getattr(reference, "p_value")),
                "n_refits": int(len(group)),
                "n_interpretable_refits": int(np.sum(interpretable_mask)),
                "min_beta": float(np.min(interpretable_betas)) if interpretable_betas.size > 0 else np.nan,
                "max_beta": float(np.max(interpretable_betas)) if interpretable_betas.size > 0 else np.nan,
                "max_abs_beta_delta": max_abs_beta_delta,
                "n_sign_flips": sign_flips,
                "sign_stable": bool(sign_flips == 0 and interpretable_betas.size > 0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_id", "refit_type"]
    ).reset_index(drop=True)


def _write_group_robustness_refits(
    *,
    group_dir: Path,
    analysis_cells: pd.DataFrame,
    pooled: pd.DataFrame,
    group_results: pd.DataFrame,
    stats_cfg: CouplingStatisticsConfig,
) -> None:
    grouped_cells = _group_confirmatory_cell_rows(analysis_cells)
    if grouped_cells.empty or group_results.empty:
        return
    robustness_dir = group_dir / "robustness"
    ensure_dir(robustness_dir)
    detail_frames: List[pd.DataFrame] = []
    for cell_row in grouped_cells.itertuples(index=False):
        pooled_subset = _confirmatory_pooled_subset(
            pooled=pooled,
            analysis_cells=analysis_cells,
            analysis_id=str(cell_row.analysis_id),
        )
        if pooled_subset.empty:
            continue
        cell = _cellspec_from_analysis_row(cell_row)
        subject_holdouts = sorted(pooled_subset["subject"].astype(str).unique().tolist())
        run_holdouts = sorted(
            int(value)
            for value in pd.to_numeric(pooled_subset["run_num"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        subject_refits = _leave_one_out_refits(
            pooled_subset=pooled_subset,
            cell=cell,
            stats_cfg=stats_cfg,
            refit_type="subject",
            holdout_values=subject_holdouts,
        )
        run_refits = _leave_one_out_refits(
            pooled_subset=pooled_subset,
            cell=cell,
            stats_cfg=stats_cfg,
            refit_type="run",
            holdout_values=run_holdouts,
        )
        detail_frames.extend([subject_refits, run_refits])
    if not detail_frames:
        return
    detail = pd.concat(detail_frames, axis=0, ignore_index=True)
    summary = _summarize_leave_one_out_refits(
        detail=detail,
        reference_results=group_results,
    )
    detail.to_csv(
        robustness_dir / "leave_one_out_refits.tsv",
        sep="\t",
        index=False,
    )
    if not summary.empty:
        summary.to_csv(
            robustness_dir / "leave_one_out_summary.tsv",
            sep="\t",
            index=False,
        )


def _load_optional_group_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path, sep="\t")
    if table.empty:
        return pd.DataFrame()
    return table


def _sign_match(
    *,
    reference_beta: Any,
    candidate_beta: Any,
) -> bool:
    ref = float(pd.to_numeric(pd.Series([reference_beta]), errors="coerce").iloc[0])
    cand = float(pd.to_numeric(pd.Series([candidate_beta]), errors="coerce").iloc[0])
    if not np.isfinite(ref) or not np.isfinite(cand):
        return False
    if ref == 0.0 or cand == 0.0:
        return False
    return bool(np.sign(ref) == np.sign(cand))


def _result_row_by_analysis_id(table: pd.DataFrame) -> Dict[str, Any]:
    if table.empty:
        return {}
    return {
        str(row.analysis_id): row
        for row in table.itertuples(index=False)
    }


def _negative_control_passes(
    *,
    row: Any,
    alpha: float,
) -> bool:
    if not bool(getattr(row, "interpretable", False)):
        return True
    p_value = float(
        pd.to_numeric(pd.Series([getattr(row, "p_value", np.nan)]), errors="coerce").iloc[0]
    )
    return bool(np.isfinite(p_value) and p_value >= float(alpha))


def _criterion_columns_for_names(
    *,
    prefix: str,
    names: Sequence[str],
) -> List[str]:
    return [f"{prefix}_{_safe_slug(name)}" for name in names]


def _write_group_adjudication_summary(
    *,
    group_dir: Path,
    group_results: pd.DataFrame,
    config: Any,
    alpha: float,
) -> None:
    if group_results.empty:
        return
    confirmatory = group_results.loc[
        group_results["family"].astype(str) == "confirmatory"
    ].reset_index(drop=True)
    if confirmatory.empty:
        return

    sensitivity_cfg = CouplingSensitivityConfig.from_config(config)
    negative_control_cfg = TrialShuffleNegativeControlConfig.from_config(config)

    source_tables = {
        item.name: _load_optional_group_results(
            group_dir / "sensitivities" / _safe_slug(item.name) / "group_results.tsv"
        )
        for item in sensitivity_cfg.source_methods.items
    }
    artifact_tables = {
        item.name: _load_optional_group_results(
            group_dir / "sensitivities" / _safe_slug(item.name) / "group_results.tsv"
        )
        for item in sensitivity_cfg.artifact_models.items
    }
    within_between_table = _load_optional_group_results(
        group_dir
        / "sensitivities"
        / _safe_slug(sensitivity_cfg.within_between.output_name)
        / "group_results.tsv"
    ) if sensitivity_cfg.within_between.enabled else pd.DataFrame()
    negative_control_table = _load_optional_group_results(
        group_dir
        / "negative_controls"
        / _safe_slug(negative_control_cfg.output_name)
        / "group_results.tsv"
    ) if negative_control_cfg.enabled else pd.DataFrame()
    loo_summary = _load_optional_group_results(
        group_dir / "robustness" / "leave_one_out_summary.tsv"
    )

    source_maps = {name: _result_row_by_analysis_id(table) for name, table in source_tables.items()}
    artifact_maps = {name: _result_row_by_analysis_id(table) for name, table in artifact_tables.items()}
    within_between_map = _result_row_by_analysis_id(within_between_table)
    negative_control_map = _result_row_by_analysis_id(negative_control_table)

    loo_subject_map: Dict[str, Any] = {}
    loo_run_map: Dict[str, Any] = {}
    if not loo_summary.empty:
        for row in loo_summary.itertuples(index=False):
            if str(row.refit_type) == "subject":
                loo_subject_map[str(row.analysis_id)] = row
            elif str(row.refit_type) == "run":
                loo_run_map[str(row.analysis_id)] = row

    rows: List[Dict[str, Any]] = []
    for ref_row in confirmatory.itertuples(index=False):
        analysis_id = str(ref_row.analysis_id)
        row: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "roi": str(ref_row.roi),
            "band": str(ref_row.band),
            "reference_beta": float(ref_row.beta),
            "reference_p_value": float(ref_row.p_value),
            "reference_p_holm": float(
                pd.to_numeric(pd.Series([getattr(ref_row, "p_holm", np.nan)]), errors="coerce").iloc[0]
            ),
            "confirmatory_interpretable": bool(ref_row.interpretable),
            "confirmatory_significant_holm": bool(getattr(ref_row, "significant_holm", False)),
        }

        source_passes: List[bool] = []
        for name in source_tables:
            candidate = source_maps[name].get(analysis_id)
            passed = (
                candidate is not None
                and bool(getattr(candidate, "interpretable", False))
                and _sign_match(reference_beta=ref_row.beta, candidate_beta=getattr(candidate, "beta", np.nan))
            )
            row[f"source_method_{_safe_slug(name)}"] = passed
            source_passes.append(bool(passed))

        artifact_passes: List[bool] = []
        for name in artifact_tables:
            candidate = artifact_maps[name].get(analysis_id)
            passed = (
                candidate is not None
                and bool(getattr(candidate, "interpretable", False))
                and _sign_match(reference_beta=ref_row.beta, candidate_beta=getattr(candidate, "beta", np.nan))
            )
            row[f"artifact_model_{_safe_slug(name)}"] = passed
            artifact_passes.append(bool(passed))

        if sensitivity_cfg.within_between.enabled:
            candidate = within_between_map.get(analysis_id)
            within_between_pass = (
                candidate is not None
                and bool(getattr(candidate, "interpretable", False))
                and _sign_match(reference_beta=ref_row.beta, candidate_beta=getattr(candidate, "beta", np.nan))
            )
            row["within_between_pass"] = within_between_pass
        else:
            within_between_pass = True

        subject_loo = loo_subject_map.get(analysis_id)
        run_loo = loo_run_map.get(analysis_id)
        subject_loo_pass = bool(getattr(subject_loo, "sign_stable", False)) if subject_loo is not None else False
        run_loo_pass = bool(getattr(run_loo, "sign_stable", False)) if run_loo is not None else False
        row["leave_one_subject_pass"] = subject_loo_pass
        row["leave_one_run_pass"] = run_loo_pass

        if negative_control_cfg.enabled:
            candidate = negative_control_map.get(analysis_id)
            negative_control_pass = (
                candidate is not None
                and _negative_control_passes(row=candidate, alpha=alpha)
            )
            row["negative_control_pass"] = negative_control_pass
        else:
            negative_control_pass = True

        all_source_pass = all(source_passes) if source_passes else True
        all_artifact_pass = all(artifact_passes) if artifact_passes else True
        row["all_source_methods_pass"] = all_source_pass
        row["all_artifact_models_pass"] = all_artifact_pass
        row["overall_robust_pass"] = bool(
            row["confirmatory_significant_holm"]
            and all_source_pass
            and all_artifact_pass
            and within_between_pass
            and subject_loo_pass
            and run_loo_pass
            and negative_control_pass
        )
        rows.append(row)

    adjudication = pd.DataFrame(rows).sort_values(
        ["roi", "band", "analysis_id"]
    ).reset_index(drop=True)
    adjudication.to_csv(
        group_dir / "robustness" / "adjudication_summary.tsv",
        sep="\t",
        index=False,
    )


def run_subject_eeg_bold_coupling(
    *,
    subject: str,
    task: str,
    config: Any,
    logger: Optional[logging.Logger] = None,
) -> SubjectCouplingOutputs:
    subject_logger = logger or LOGGER
    subject_raw = _subject_raw(subject)
    subject_bids = _subject_bids(subject)
    deriv_root = resolve_deriv_root(config=config)
    _materialize_runtime_roi_specs(config)
    coupling_cfg = EEGBOLDCouplingConfig.from_config(config)
    output_dir = _resolve_subject_output_dir(
        deriv_root=deriv_root,
        subject=subject_raw,
        task=task,
        cfg=coupling_cfg,
    )
    ensure_dir(output_dir)
    profiler = SubjectStageProfiler(
        enabled=coupling_cfg.runtime.profile_stages,
        output_dir=output_dir,
        subject=subject_raw,
        task=task,
    )
    try:
        if coupling_cfg.runtime.preflight:
            with profiler.stage("preflight"):
                _run_subject_preflight(
                    subject=subject_raw,
                    task=task,
                    config=config,
                    coupling_cfg=coupling_cfg,
                    output_dir=output_dir,
                )

        with profiler.stage("load_epochs"):
            epochs, clean_events = load_epochs_for_analysis(
                subject=subject,
                task=task,
                align="strict",
                preload=True,
                deriv_root=deriv_root,
                logger=subject_logger,
                config=config,
            )
            if epochs is None or clean_events is None:
                raise ValueError(f"Could not load epochs/events for sub-{subject}.")

        expression_cfg = LocalSignatureExpressionConfig.from_config(config)
        nuisance_cfg = CouplingNuisanceConfig.from_config(
            config,
            active_window=coupling_cfg.eeg.active_window,
            epoch_window=(float(epochs.tmin), float(epochs.tmax)),
            default_hrf_model=coupling_cfg.fmri.hrf_model,
        )
        with profiler.stage("validate_clean_events"):
            _validate_clean_event_qc_columns(
                clean_events=clean_events,
                nuisance_cfg=nuisance_cfg,
            )

        with profiler.stage("resolve_source_model"):
            fwd, _src = _resolve_subject_source_model(
                subject=subject,
                epochs=epochs,
                config=config,
                coupling_cfg=coupling_cfg,
                logger=subject_logger,
            )
            labels = _build_subject_labels(
                subject=subject_bids,
                cfg=coupling_cfg,
                logger=subject_logger,
            )
            surface_info = _subject_surface_info(
                subject=subject_raw,
                subjects_dir=coupling_cfg.eeg.subjects_dir,
            )
            rois = _map_rois_to_source_rows(
                source_vertices=_source_vertices_from_forward(fwd),
                labels=labels,
                surface_info=surface_info,
            )

        with profiler.stage("extract_eeg_features"):
            times = np.asarray(epochs.times, dtype=float)
            active_mask = _build_window_mask(
                times=times,
                window=coupling_cfg.eeg.active_window,
                label="EEG active",
            )
            baseline_mask = _build_window_mask(
                times=times,
                window=coupling_cfg.eeg.baseline_window,
                label="EEG baseline",
            )
            frequency_bands = get_config_value(config, "frequency_bands", {})
            subject_logger.info(
                "Extracting trial-wise EEG source power with method %s",
                coupling_cfg.eeg.method,
            )
            eeg_table = _extract_trialwise_eeg_table(
                epochs=epochs,
                clean_events=clean_events,
                fwd=fwd,
                rois=rois,
                cfg=coupling_cfg,
                frequency_bands=frequency_bands,
                active_mask=active_mask,
                baseline_mask=baseline_mask,
                logger=subject_logger,
            )
            eeg_table_base = eeg_table.copy()
            eeg_table = compute_eeg_artifact_table(
                epochs=epochs,
                clean_events=clean_events,
                eeg_table=eeg_table,
                nuisance_cfg=nuisance_cfg,
            )
            subject_logger.info(
                "Prepared EEG trial table with nuisance metrics for %d trials",
                len(eeg_table),
            )

        with profiler.stage("run_lss"):
            subject_logger.info("Running subject-level LSS trial beta extraction")
            lss_dir = _run_subject_lss(
                subject=subject_raw,
                task=task,
                deriv_root=deriv_root,
                output_dir=output_dir,
                config=config,
                coupling_cfg=coupling_cfg,
                progress_callback=profiler.touch,
            )
            trials_path = lss_dir / "trials.tsv"
            if not trials_path.exists():
                raise FileNotFoundError(f"Missing LSS trials table: {trials_path}")
            trials_df = pd.read_csv(trials_path, sep="\t")
            trial_table = _prepare_trial_table(
                trials_df=trials_df,
                coupling_cfg=coupling_cfg,
                config=config,
            )
            trial_table = compute_fd_table(
                subject=subject_raw,
                task=task,
                deriv_root=deriv_root,
                trial_table=trial_table,
                nuisance_cfg=nuisance_cfg,
                fmri_space=coupling_cfg.fmri.fmriprep_space,
            )
            trial_table = compute_dvars_table(
                subject=subject_raw,
                task=task,
                deriv_root=deriv_root,
                trial_table=trial_table,
                nuisance_cfg=nuisance_cfg,
                fmri_space=coupling_cfg.fmri.fmriprep_space,
            )

        with profiler.stage("match_trials_and_signatures"):
            matched_trial_table = trial_table.loc[
                trial_table["trial_key"].isin(set(eeg_table["trial_key"]))
            ].reset_index(drop=True)
            if matched_trial_table.empty:
                raise ValueError(
                    f"No EEG trials matched selected LSS trials for {subject_bids}."
                )
            signature_weights = None
            if expression_cfg.enabled:
                signature_paths = resolve_signature_paths(config)
                if not signature_paths:
                    raise ValueError(
                        "Local signature expression was enabled, but no signature maps were configured."
                    )
                if expression_cfg.signatures:
                    missing = [
                        name
                        for name in expression_cfg.signatures
                        if name not in signature_paths
                    ]
                    if missing:
                        raise ValueError(
                            f"Missing configured local-expression signatures: {missing}"
                        )
                    signature_paths = {
                        name: signature_paths[name]
                        for name in expression_cfg.signatures
                    }
                signature_weights = sample_signatures_to_subject_surface(
                    signature_paths=signature_paths,
                    subject_surface_paths=_subject_surface_paths(
                        subject=subject_raw,
                        subjects_dir=coupling_cfg.eeg.subjects_dir,
                    ),
                )

        with profiler.stage("extract_bold_features"):
            bold_table = _extract_trialwise_bold_features_surface_glm(
                subject=subject_raw,
                task=task,
                trial_table=matched_trial_table,
                rois=rois,
                surface_info=surface_info,
                config=config,
                coupling_cfg=coupling_cfg,
                require_variance=coupling_cfg.statistics.use_outcome_variance,
                signature_weights=signature_weights,
                expression_cfg=expression_cfg if expression_cfg.enabled else None,
                logger=subject_logger,
                progress_callback=profiler.touch,
            )

        with profiler.stage("fmri_qc"):
            fmri_qc = run_subject_fmri_qc(
                subject=subject_raw,
                task=task,
                deriv_root=deriv_root,
                config=config,
                lss_cfg=_build_lss_config(task=task, cfg=coupling_cfg),
                bold_table=bold_table,
            )

        with profiler.stage("merge_and_censor"):
            merged = _merge_trialwise_tables(
                eeg_table=eeg_table,
                trial_table=matched_trial_table,
                bold_table=bold_table,
            )
            _validate_model_terms_present(
                table=merged,
                terms=coupling_cfg.covariates.model_terms,
            )
            merged_kept, qc_table, qc_summary = apply_trial_censoring(
                merged_table=merged,
                model_terms=coupling_cfg.covariates.model_terms,
                nuisance_cfg=nuisance_cfg,
            )
            qc_summary["subject"] = subject_raw
            qc_summary["task"] = task

        with profiler.stage("summarize_subject_cells"):
            primary_cells = _confirmatory_cell_specs(
                rois=rois,
                bands=coupling_cfg.eeg.bands,
                model_terms=coupling_cfg.covariates.model_terms,
                use_outcome_variance=coupling_cfg.statistics.use_outcome_variance,
            )
            analysis_cells = summarize_subject_cells(
                subject=subject_raw,
                merged_table=merged_kept,
                cell_specs=primary_cells,
                stats_cfg=coupling_cfg.statistics,
            )
            analysis_cells = _apply_qc_status_to_cells(
                analysis_cells=analysis_cells,
                qc_result=fmri_qc,
            )
            if expression_cfg.enabled:
                local_cells = summarize_subject_cells(
                    subject=subject_raw,
                    merged_table=merged_kept,
                    cell_specs=_local_expression_cell_specs(
                        rois=rois,
                        bands=coupling_cfg.eeg.bands,
                        bold_table=bold_table,
                        model_terms=coupling_cfg.covariates.model_terms,
                    ),
                    stats_cfg=coupling_cfg.statistics,
                )
                local_cells = _apply_qc_status_to_cells(
                    analysis_cells=local_cells,
                    qc_result=fmri_qc,
                )
            else:
                local_cells = pd.DataFrame()

        with profiler.stage("write_outputs"):
            eeg_table.to_csv(output_dir / "trialwise_eeg.tsv", sep="\t", index=False)
            matched_trial_table.to_csv(
                output_dir / "trialwise_trials.tsv",
                sep="\t",
                index=False,
            )
            bold_table.to_csv(output_dir / "trialwise_bold.tsv", sep="\t", index=False)
            merged_path = output_dir / "trialwise_merged.tsv"
            qc_table.to_csv(merged_path, sep="\t", index=False)
            qc_table.to_csv(output_dir / "trialwise_qc.tsv", sep="\t", index=False)
            analysis_cells_path = output_dir / "analysis_cells.tsv"
            analysis_cells.to_csv(analysis_cells_path, sep="\t", index=False)
            roi_manifest_path = output_dir / "roi_manifest.tsv"
            _write_roi_manifest(rois=rois, path=roi_manifest_path)
            write_qc_summary(output_dir / "qc_summary.json", qc_summary)
            write_subject_fmri_qc(output_dir=output_dir, qc_result=fmri_qc)
            if not local_cells.empty:
                _write_analysis_bundle(
                    output_dir=_resolve_secondary_output_dir(
                        subject_output_dir=output_dir,
                        secondary_name="local_signature_expression",
                    ),
                    merged_table=qc_table,
                    analysis_cells=local_cells,
                )
            (output_dir / "coupling_config.json").write_text(
                json.dumps(asdict(coupling_cfg), indent=2, default=str),
                encoding="utf-8",
            )

        with profiler.stage("subject_sensitivities"):
            _run_subject_sensitivity_analyses(
                subject=subject_raw,
                task=task,
                deriv_root=deriv_root,
                config=config,
                subject_output_dir=output_dir,
                epochs=epochs,
                clean_events=clean_events,
                fwd=fwd,
                surface_info=surface_info,
                rois=rois,
                trial_table=matched_trial_table,
                eeg_table_base=eeg_table_base,
                eeg_table=eeg_table,
                merged_kept=merged_kept,
                primary_bold_table=bold_table,
                primary_cells=primary_cells,
                coupling_cfg=coupling_cfg,
                nuisance_cfg=nuisance_cfg,
                signature_weights=signature_weights,
                expression_cfg=expression_cfg,
                fmri_qc=fmri_qc,
            )
        with profiler.stage("subject_negative_controls"):
            _run_subject_negative_controls(
                subject=subject_raw,
                config=config,
                subject_output_dir=output_dir,
                trial_table=matched_trial_table,
                eeg_table=eeg_table,
                bold_table=bold_table,
                primary_cells=primary_cells,
                coupling_cfg=coupling_cfg,
                nuisance_cfg=nuisance_cfg,
                fmri_qc=fmri_qc,
            )
        profiler.mark_completed()
        return SubjectCouplingOutputs(
            subject=subject_raw,
            output_dir=output_dir,
            analysis_cells_path=analysis_cells_path,
            merged_trials_path=merged_path,
            roi_manifest_path=roi_manifest_path,
        )
    except Exception as exc:
        profiler.mark_failed(str(exc))
        raise


def run_group_eeg_bold_coupling(
    *,
    subjects: Sequence[str],
    task: str,
    config: Any,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    group_logger = logger or LOGGER
    deriv_root = _as_path(get_config_value(config, "paths.deriv_root", None))
    if deriv_root is None:
        deriv_root = resolve_deriv_root(config=config)
    _materialize_runtime_roi_specs(config)
    coupling_cfg = EEGBOLDCouplingConfig.from_config(config)
    if coupling_cfg.runtime.preflight:
        _run_group_preflight(config=config, coupling_cfg=coupling_cfg)
    sensitivity_cfg = CouplingSensitivityConfig.from_config(config)
    skipped_sensitivity_names: List[str] = []
    if sensitivity_cfg.residualized_correlation.enabled:
        skipped_sensitivity_names.append(
            _safe_slug(sensitivity_cfg.residualized_correlation.output_name)
        )
    if sensitivity_cfg.primary_permutation.enabled:
        skipped_sensitivity_names.append(
            _safe_slug(sensitivity_cfg.primary_permutation.output_name)
        )
    cell_frames: List[pd.DataFrame] = []
    pooled_frames: List[pd.DataFrame] = []
    for subject in subjects:
        subject_dir = _resolve_subject_output_dir(
            deriv_root=deriv_root,
            subject=subject,
            task=task,
            cfg=coupling_cfg,
        )
        analysis_cells_path = subject_dir / "analysis_cells.tsv"
        merged_path = subject_dir / "trialwise_merged.tsv"
        if not analysis_cells_path.exists() or not merged_path.exists():
            continue
        cells = pd.read_csv(analysis_cells_path, sep="\t", dtype={"subject": str})
        merged = pd.read_csv(merged_path, sep="\t")
        if cells.empty or merged.empty:
            continue
        merged = merged.loc[_keep_trial_mask(merged)].reset_index(drop=True)
        if merged.empty:
            continue
        merged["subject"] = str(subject).replace("sub-", "", 1)
        cell_frames.append(cells)
        pooled_frames.append(merged)

    group_dir = _resolve_group_output_dir(
        deriv_root=deriv_root,
        task=task,
        cfg=coupling_cfg,
    )
    ensure_dir(group_dir)

    if not cell_frames or not pooled_frames:
        if sensitivity_cfg.residualized_correlation.enabled:
            _aggregate_residualized_sensitivity_dir(
                subjects=subjects,
                task=task,
                deriv_root=deriv_root,
                cfg=coupling_cfg,
                sensitivity_name=sensitivity_cfg.residualized_correlation.output_name,
                output_root=group_dir / "sensitivities",
                config=config,
            )
        if sensitivity_cfg.primary_permutation.enabled:
            _aggregate_primary_permutation_sensitivity_dir(
                subjects=subjects,
                task=task,
                deriv_root=deriv_root,
                cfg=coupling_cfg,
                sensitivity_name=sensitivity_cfg.primary_permutation.output_name,
                output_root=group_dir / "sensitivities",
                config=config,
            )
        _aggregate_analysis_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            relative_dir=("sensitivities",),
            output_root=group_dir / "sensitivities",
            skip_analysis_names=tuple(skipped_sensitivity_names),
        )
        _aggregate_analysis_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            relative_dir=("secondary",),
            output_root=group_dir / "secondary",
        )
        _aggregate_analysis_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            relative_dir=("negative_controls",),
            output_root=group_dir / "negative_controls",
        )
        group_logger.warning("No subject-level coupling results found for group aggregation.")
        return None

    analysis_cells = pd.concat(cell_frames, axis=0, ignore_index=True)
    pooled = pd.concat(pooled_frames, axis=0, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    grouped_cells = analysis_cells.drop_duplicates(
        subset=[
            "analysis_id",
            "family",
            "roi",
            "band",
            "predictor_column",
            "outcome_column",
            "outcome_variance_column",
            "nonstandardized_terms",
        ]
    )
    for cell_row in grouped_cells.itertuples(index=False):
        ok_subjects = analysis_cells.loc[
            (analysis_cells["analysis_id"].astype(str) == str(cell_row.analysis_id))
            & (analysis_cells["status"].astype(str) == "ok"),
            "subject",
        ].astype(str)
        pooled_subset = pooled.loc[
            pooled["subject"].astype(str).isin(ok_subjects.tolist())
        ].reset_index(drop=True)
        if pooled_subset.empty:
            continue
        row = fit_mixedlm_cell(
            pooled_table=pooled_subset,
            cell=CellSpec(
                analysis_id=str(cell_row.analysis_id),
                family=str(cell_row.family),
                roi=str(cell_row.roi),
                band=str(cell_row.band),
                predictor_column=str(cell_row.predictor_column),
                outcome_column=str(cell_row.outcome_column),
                outcome_variance_column=(
                    None
                    if str(getattr(cell_row, "outcome_variance_column", "")).strip() == ""
                    else str(getattr(cell_row, "outcome_variance_column"))
                ),
                model_terms=tuple(json.loads(str(cell_row.model_terms))),
                categorical_terms=tuple(json.loads(str(cell_row.categorical_terms))),
                nonstandardized_terms=tuple(
                    json.loads(str(getattr(cell_row, "nonstandardized_terms", "[]")))
                ),
            ),
            stats_cfg=coupling_cfg.statistics,
        )
        rows.append(row)
    group_results = finalize_group_results(
        rows=rows,
        alpha=coupling_cfg.statistics.alpha,
    )
    if group_results.empty:
        if sensitivity_cfg.residualized_correlation.enabled:
            _aggregate_residualized_sensitivity_dir(
                subjects=subjects,
                task=task,
                deriv_root=deriv_root,
                cfg=coupling_cfg,
                sensitivity_name=sensitivity_cfg.residualized_correlation.output_name,
                output_root=group_dir / "sensitivities",
                config=config,
            )
        if sensitivity_cfg.primary_permutation.enabled:
            _aggregate_primary_permutation_sensitivity_dir(
                subjects=subjects,
                task=task,
                deriv_root=deriv_root,
                cfg=coupling_cfg,
                sensitivity_name=sensitivity_cfg.primary_permutation.output_name,
                output_root=group_dir / "sensitivities",
                config=config,
            )
        _aggregate_analysis_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            relative_dir=("sensitivities",),
            output_root=group_dir / "sensitivities",
            skip_analysis_names=tuple(skipped_sensitivity_names),
        )
        _aggregate_analysis_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            relative_dir=("secondary",),
            output_root=group_dir / "secondary",
        )
        _aggregate_analysis_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            relative_dir=("negative_controls",),
            output_root=group_dir / "negative_controls",
        )
        group_logger.warning("Group aggregation produced no valid results.")
        return None

    analysis_cells.to_csv(
        group_dir / "analysis_cells_all.tsv",
        sep="\t",
        index=False,
    )
    group_path = group_dir / "group_results.tsv"
    group_results.to_csv(group_path, sep="\t", index=False)
    _write_group_robustness_refits(
        group_dir=group_dir,
        analysis_cells=analysis_cells,
        pooled=pooled,
        group_results=group_results,
        stats_cfg=coupling_cfg.statistics,
    )
    if sensitivity_cfg.residualized_correlation.enabled:
        _aggregate_residualized_sensitivity_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            sensitivity_name=sensitivity_cfg.residualized_correlation.output_name,
            output_root=group_dir / "sensitivities",
            config=config,
        )
    if sensitivity_cfg.primary_permutation.enabled:
        _aggregate_primary_permutation_sensitivity_dir(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            cfg=coupling_cfg,
            sensitivity_name=sensitivity_cfg.primary_permutation.output_name,
            output_root=group_dir / "sensitivities",
            config=config,
        )
    _aggregate_analysis_dir(
        subjects=subjects,
        task=task,
        deriv_root=deriv_root,
        cfg=coupling_cfg,
        relative_dir=("sensitivities",),
        output_root=group_dir / "sensitivities",
        skip_analysis_names=tuple(skipped_sensitivity_names),
    )
    _aggregate_analysis_dir(
        subjects=subjects,
        task=task,
        deriv_root=deriv_root,
        cfg=coupling_cfg,
        relative_dir=("secondary",),
        output_root=group_dir / "secondary",
    )
    _aggregate_analysis_dir(
        subjects=subjects,
        task=task,
        deriv_root=deriv_root,
        cfg=coupling_cfg,
        relative_dir=("negative_controls",),
        output_root=group_dir / "negative_controls",
    )
    _write_group_adjudication_summary(
        group_dir=group_dir,
        group_results=group_results,
        config=config,
        alpha=coupling_cfg.statistics.alpha,
    )
    return group_path


__all__ = [
    "EEGBOLDCouplingConfig",
    "run_subject_eeg_bold_coupling",
    "run_group_eeg_bold_coupling",
]
