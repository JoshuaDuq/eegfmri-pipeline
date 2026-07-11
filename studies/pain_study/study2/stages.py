"""Study 2 stage implementations wired to disk artifacts.

Each stage reads its declared input artifacts, calls the existing pure Study 2
analysis functions, and writes its own artifacts. The orchestration lives here
so the analysis modules stay free of filesystem concerns.
"""

from __future__ import annotations

import copy
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from studies.pain_study.study2 import paths
from studies.pain_study.study2.artifact_controls import (
    evaluate_artifact_controls,
    evaluate_robustness_summary,
)
from studies.pain_study.study2.behavioral_convergence import compute_behavioral_convergence
from studies.pain_study.study2.directional_consistency import evaluate_directional_consistency
from studies.pain_study.study2.gates import (
    evaluate_study1_confirmatory_criteria,
    load_study1_confirmatory_row,
)
from studies.pain_study.study2.point_spread import compute_point_spread_fwhm
from studies.pain_study.study2.source_family import (
    compute_source_family_inference,
    summarize_source_family,
)
from studies.pain_study.study2.source_maps import (
    compute_band_unique_cohort_source_association_maps,
    compute_cohort_source_association_maps,
)
from studies.pain_study.study2.source_model_qc import evaluate_source_model_qc
from studies.pain_study.study2.sensor_patterns import compute_sensor_pattern_summary
from studies.pain_study.study2.source_stage_design import contribution_bands
from studies.pain_study.study2.source_power import (
    apply_sloreta_inverse,
    build_surface_forward_model,
    compute_baseline_noise_covariance,
    compute_morphed_sloreta_hilbert_logratio_power,
    make_surface_source_morph,
    make_sloreta_inverse_operator,
)
from studies.pain_study.study2.source_vertex_manifest import ensure_common_source_vertices
from studies.pain_study.study2.study1_context import (
    load_study1_model_context,
    study1_model_comparison_path,
)
from studies.pain_study.study2.spatial_comparison import compute_spatial_correspondence
from studies.pain_study.study2.table_io import (
    format_mapping,
    metric_mapping,
    parse_bool,
    require_columns,
)
from studies.pain_study.study2.target_retrained_null import build_target_retrained_null_maps
from studies.pain_study.study2.validation import (
    require_config_float,
    require_config_int,
    require_config_string,
    require_config_value,
)

if TYPE_CHECKING:
    from studies.pain_study.study2.runner import Study2StageContext


BandFrequencyRanges = tuple[tuple[float, float], ...]


def _band_frequency_ranges(config: Any) -> dict[str, BandFrequencyRanges]:
    bands = require_config_value(config, "study2.source_modeling.frequency_bands")
    if not isinstance(bands, Mapping):
        raise ValueError("study2.source_modeling.frequency_bands must be a mapping.")
    ranges: dict[str, BandFrequencyRanges] = {}
    for band in contribution_bands(config):
        bounds = bands.get(band)
        ranges[band] = _parse_band_frequency_ranges(
            bounds,
            field_name=f"study2.source_modeling.frequency_bands.{band}",
        )
    return ranges


def _parse_band_frequency_ranges(value: Any, *, field_name: str) -> BandFrequencyRanges:
    if _is_frequency_range(value):
        return (_frequency_range(value, field_name=field_name),)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must be [low, high] or a non-empty list of ranges.")
    ranges = tuple(
        _frequency_range(bounds, field_name=f"{field_name}[{index}]")
        for index, bounds in enumerate(value)
    )
    _require_non_overlapping_ranges(ranges, field_name=field_name)
    return ranges


def _is_frequency_range(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and not isinstance(value[0], (list, tuple))
        and not isinstance(value[1], (list, tuple))
    )


def _frequency_range(value: Any, *, field_name: str) -> tuple[float, float]:
    if not _is_frequency_range(value):
        raise ValueError(f"{field_name} must be [low, high].")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{field_name} must contain finite frequencies.")
    if high <= low:
        raise ValueError(f"{field_name} must have high > low.")
    return low, high


def _require_non_overlapping_ranges(
    ranges: BandFrequencyRanges,
    *,
    field_name: str,
) -> None:
    ordered = sorted(ranges)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"{field_name} ranges must not overlap.")


def _time_window(config: Any, key: str) -> tuple[float, float]:
    window = require_config_value(config, key)
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f"{key} must be a [start, stop] window.")
    return float(window[0]), float(window[1])


@dataclass(frozen=True)
class _AnatomyPaths:
    subjects_dir: Path
    trans: Path
    bem: Path


def _resolve_anatomy(config: Any, *, subject_id: str) -> _AnatomyPaths:
    subjects_dir = require_config_string(
        config,
        "study2.source_modeling.anatomy.subjects_dir",
    )
    trans_template = require_config_string(
        config,
        "study2.source_modeling.anatomy.trans_path_template",
    )
    bem_template = require_config_string(
        config,
        "study2.source_modeling.anatomy.bem_path_template",
    )
    substitutions = {"subjects_dir": str(subjects_dir), "subject": subject_id}
    return _AnatomyPaths(
        subjects_dir=Path(str(subjects_dir)),
        trans=Path(str(trans_template).format(**substitutions)),
        bem=Path(str(bem_template).format(**substitutions)),
    )


def _load_subject_epochs(config: Any, *, subject_id: str, task: str, logger: Any) -> Any:
    epochs, _events = load_epochs_for_analysis(
        subject_id,
        task,
        preload=True,
        deriv_root=resolve_eeg_deriv_root(config),
        config=config,
        logger=logger,
    )
    if epochs is None:
        raise FileNotFoundError(
            f"Study 2 source-power found no clean epochs for {subject_id} (task={task})."
        )
    excluded = [
        str(channel)
        for channel in require_config_value(config, "study2.source_modeling.rank_excluded_channels")
    ]
    present = [channel for channel in excluded if channel in epochs.ch_names]
    if present:
        epochs.drop_channels(present)
    return epochs


def source_power_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required: list[Path] = []
    for subject_id in context.subjects:
        anatomy = _resolve_anatomy(config, subject_id=subject_id)
        required.extend([anatomy.subjects_dir, anatomy.trans, anatomy.bem])
    return tuple(required)


def _aggregate_subband_logratio_power(
    extractions: list[Any],
    *,
    frequency_ranges: BandFrequencyRanges,
) -> np.ndarray:
    if len(extractions) != len(frequency_ranges):
        raise ValueError("Study 2 source-power subband extraction count does not match ranges.")
    if len(extractions) == 1:
        return extractions[0].power_logratio

    reference = extractions[0]
    for extraction in extractions[1:]:
        if extraction.power_logratio.shape != reference.power_logratio.shape:
            raise ValueError("Study 2 source-power subband maps must have identical shapes.")
        if extraction.baseline_window_s != reference.baseline_window_s:
            raise ValueError("Study 2 source-power subbands must use the same baseline window.")
        if extraction.active_window_s != reference.active_window_s:
            raise ValueError("Study 2 source-power subbands must use the same active window.")

    weights = np.asarray([high - low for low, high in frequency_ranges], dtype=float)
    maps = np.stack([extraction.power_logratio for extraction in extractions], axis=0)
    return np.average(maps, axis=0, weights=weights)


def _frequency_aggregation_label(frequency_ranges: BandFrequencyRanges) -> str:
    if len(frequency_ranges) == 1:
        return "single_contiguous_band"
    return "bandwidth_weighted_logratio_mean"


def run_source_power(context: "Study2StageContext") -> None:
    """Reconstruct per-subject sLORETA source power per band from cleaned epochs."""
    config = context.config
    band_ranges = _band_frequency_ranges(config)
    baseline_window = _time_window(config, "study2.source_modeling.noise_covariance_baseline_s")
    active_window = _time_window(config, "study2.source_modeling.active_plateau_window_s")
    snr = require_config_float(config, "study2.source_modeling.regularization.snr")
    loose = require_config_float(
        config,
        "study2.source_modeling.regularization.loose_orientation",
    )
    depth = require_config_float(
        config,
        "study2.source_modeling.regularization.depth_weighting",
    )
    spacing = require_config_string(config, "study2.source_modeling.source_space_spacing")
    mindist_mm = require_config_float(config, "study2.source_modeling.forward_mindist_mm")
    common_subject = _required_config_string(
        config,
        "study2.source_modeling.common_subject",
    )
    common_spacing = _required_config_string(
        config,
        "study2.source_modeling.common_source_space_spacing",
    )

    for subject_id in context.subjects:
        anatomy = _resolve_anatomy(config, subject_id=subject_id)
        epochs = _load_subject_epochs(
            config, subject_id=subject_id, task=context.task, logger=context.logger
        )
        forward = build_surface_forward_model(
            epochs.info,
            subject=subject_id,
            subjects_dir=str(anatomy.subjects_dir),
            trans=str(anatomy.trans),
            bem=str(anatomy.bem),
            spacing=spacing,
            mindist_mm=mindist_mm,
        )
        noise_cov = compute_baseline_noise_covariance(epochs, baseline_window_s=baseline_window)
        inverse_operator = make_sloreta_inverse_operator(
            info=epochs.info,
            forward=forward,
            noise_cov=noise_cov,
            loose=loose,
            depth=depth,
        )

        source_morph = None
        for band, frequency_ranges in band_ranges.items():
            subband_extractions = []
            for low, high in frequency_ranges:
                band_epochs = epochs.copy().filter(low, high, verbose=False)
                stcs = apply_sloreta_inverse(
                    epochs=band_epochs,
                    inverse_operator=inverse_operator,
                    snr=snr,
                    pick_ori="normal",
                )
                if source_morph is None:
                    source_morph = make_surface_source_morph(
                        reference_stc=stcs[0],
                        subject_from=subject_id,
                        subject_to=common_subject,
                        subjects_dir=str(anatomy.subjects_dir),
                        spacing=common_spacing,
                    )
                extraction = compute_morphed_sloreta_hilbert_logratio_power(
                    stcs=stcs,
                    times=np.asarray(epochs.times, dtype=float),
                    baseline_window_s=baseline_window,
                    active_window_s=active_window,
                    morph=source_morph,
                )
                ensure_common_source_vertices(
                    array_path=paths.source_vertex_manifest_path(config),
                    metadata_path=paths.source_vertex_metadata_path(config),
                    vertices=extraction.vertices,
                    common_subject=common_subject,
                    spacing=common_spacing,
                )
                subband_extractions.append(extraction)
                del band_epochs, stcs
                gc.collect()
            extraction = subband_extractions[0]
            power_logratio = _aggregate_subband_logratio_power(
                subband_extractions,
                frequency_ranges=frequency_ranges,
            )
            power_path = paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            power_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(power_path, power_logratio)
            metadata_path = paths.subject_source_power_metadata_path(
                config,
                subject_id=subject_id,
                band=band,
            )
            metadata_payload = {
                "subject_id": subject_id,
                "band": band,
                "frequency_hz": [list(frequency_range) for frequency_range in frequency_ranges],
                "frequency_aggregation": _frequency_aggregation_label(frequency_ranges),
                "baseline_window_s": list(extraction.baseline_window_s),
                "active_window_s": list(extraction.active_window_s),
                "source_space_spacing": spacing,
                "common_subject": common_subject,
                "common_source_space_spacing": common_spacing,
                "inverse_method": "sLORETA",
                "snr": float(snr),
                "loose_orientation": float(loose),
                "depth_weighting": float(depth),
                "n_trials": extraction.n_trials,
                "n_vertices": extraction.n_vertices,
            }
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata_payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            context.logger.info(
                "Study 2 source-power %s %s: %d trials x %d vertices.",
                subject_id,
                band,
                extraction.n_trials,
                extraction.n_vertices,
            )
            del subband_extractions, extraction, power_logratio
            gc.collect()
        del forward, noise_cov, inverse_operator, source_morph, epochs
        gc.collect()


def gate_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.study1_report_path(context.config),)


def run_gate(context: "Study2StageContext") -> None:
    """Evaluate Study 1 confirmatory criteria and persist the QC metrics."""
    config = context.config
    metrics = load_study1_confirmatory_row(
        paths.study1_report_path(config),
        config=config,
    )
    qc = evaluate_study1_confirmatory_criteria(metrics, config)

    gate_path = paths.gate_qc_path(config)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "confirmatory_criteria_met": qc.confirmatory_criteria_met,
        "unmet_criteria": list(qc.unmet_criteria),
    }
    with open(gate_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    context.logger.info(
        "Study 2 confirmatory criteria met: %s; unmet criteria: %s",
        qc.confirmatory_criteria_met,
        ",".join(qc.unmet_criteria),
    )


def haufe_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (
        paths.study1_report_path(context.config),
        study1_model_comparison_path(_study1_capable_config(context.config)),
    )


def run_haufe(context: "Study2StageContext") -> None:
    """Compute foldwise sensor patterns from the frozen Study 1 NPS model."""
    config = context.config
    figure_config = require_config_value(config, "study2.figures.haufe_forward_patterns")
    if not isinstance(figure_config, Mapping):
        raise ValueError("study2.figures.haufe_forward_patterns must be a mapping.")
    target = require_config_string(config, "study2.confirmatory.study1_cell.target")
    model = require_config_string(config, "study2.confirmatory.study1_cell.model")
    feature_spec = require_config_string(
        config,
        "study2.confirmatory.study1_cell.frequency_preset",
    )
    if target != str(figure_config.get("target")):
        raise ValueError("Study 2 Haufe figure target must match the confirmatory target.")
    if model != str(figure_config.get("model")):
        raise ValueError("Study 2 Haufe figure model must match the confirmatory model.")
    if feature_spec != str(figure_config.get("feature_spec")):
        raise ValueError("Study 2 Haufe figure feature spec must match the confirmatory preset.")
    band_specs = figure_config.get("bands")
    if not isinstance(band_specs, list) or not band_specs:
        raise ValueError("Study 2 Haufe figure bands must be a non-empty list.")
    bands = tuple(str(spec["name"]) for spec in band_specs)
    study1_config = _study1_capable_config(config)

    model_context = load_study1_model_context(
        subjects=list(context.subjects),
        task=context.task,
        config=study1_config,
        target_name=target,
        feature_spec=feature_spec,
        logger=context.logger,
    )
    fold_metrics = pd.read_csv(study1_model_comparison_path(study1_config), sep="\t")
    summary = compute_sensor_pattern_summary(
        context=model_context,
        fold_metrics=fold_metrics,
        target=target,
        bands=bands,
        refit_tolerance=float(figure_config["refit_r2_tolerance"]),
        minimum_article_subjects=int(figure_config["minimum_article_subjects"]),
    )

    outputs = (
        (summary.fold_patterns, paths.haufe_fold_patterns_path(config)),
        (summary.aggregate_patterns, paths.haufe_aggregate_patterns_path(config)),
        (summary.stability, paths.haufe_stability_path(config)),
    )
    for frame, tsv_path in outputs:
        write_tsv(frame, tsv_path)
        write_parquet(frame, tsv_path.with_suffix(".parquet"))


def source_model_qc_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.source_model_metrics_path(context.config),)


def run_source_model_qc(context: "Study2StageContext") -> None:
    """Evaluate source-model reconstruction metrics for each subject."""
    config = context.config
    metrics = pd.read_csv(paths.source_model_metrics_path(config), sep="\t")
    require_columns(metrics, ("subject_id",), name="Study 2 source-model metrics")

    records = []
    for row in metrics.to_dict("records"):
        subject_id = str(row.pop("subject_id"))
        qc = evaluate_source_model_qc(subject_id=subject_id, metrics=row, config=config)
        records.append(
            {
                "subject_id": qc.subject_id,
                "source_model_criteria_met": qc.source_model_criteria_met,
                "valid_eeg_channel_location_fraction": qc.valid_eeg_channel_location_fraction,
                "mean_coregistration_error_mm": qc.mean_coregistration_error_mm,
                "max_coregistration_error_mm": qc.max_coregistration_error_mm,
                "unmet_criteria": ";".join(qc.unmet_criteria),
            }
        )

    paths.source_model_dir(config).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(
        paths.source_model_qc_path(config),
        sep="\t",
        index=False,
    )


def point_spread_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    return (
        paths.point_spread_resolution_matrix_path(config),
        paths.point_spread_distances_path(config),
    )


def run_point_spread(context: "Study2StageContext") -> None:
    """Summarize source-resolution point-spread FWHM."""
    config = context.config
    report = compute_point_spread_fwhm(
        resolution_matrix=np.load(paths.point_spread_resolution_matrix_path(config)),
        distances_mm=np.load(paths.point_spread_distances_path(config)),
    )
    paths.source_model_dir(config).mkdir(parents=True, exist_ok=True)
    np.save(paths.point_spread_vertex_fwhm_path(config), report.vertex_fwhm_mm)
    pd.DataFrame(
        [
            {
                "median_fwhm_mm": report.median_fwhm_mm,
                "min_fwhm_mm": report.min_fwhm_mm,
                "max_fwhm_mm": report.max_fwhm_mm,
                "q1_fwhm_mm": report.q1_fwhm_mm,
                "q3_fwhm_mm": report.q3_fwhm_mm,
                "n_vertices": report.n_vertices,
            }
        ]
    ).to_csv(paths.point_spread_summary_path(config), sep="\t", index=False)


def _source_map_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required = [paths.source_stage_frame_path(config)]
    for subject_id in context.subjects:
        for band in contribution_bands(config):
            required.append(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
    return tuple(required)


def _run_source_map_family(
    context: "Study2StageContext",
    *,
    compute_maps: Callable[..., Any],
    fisher_z_path: Callable[..., Path],
    partial_r_path: Callable[..., Path],
    qc_path: Callable[..., Path],
    log_label: str,
) -> None:
    config = context.config
    frame = pd.read_csv(paths.source_stage_frame_path(config), sep="\t")

    for band in contribution_bands(config):
        result = compute_maps(
            frame,
            _source_power_by_subject(config, subjects=context.subjects, band=band),
            band=band,
            config=config,
        )
        fisher_path = fisher_z_path(config, band=band)
        partial_path = partial_r_path(config, band=band)
        qc_output_path = qc_path(config, band=band)
        fisher_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        qc_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(fisher_path, result.fisher_z_maps)
        np.save(partial_path, result.partial_r_maps)
        result.qc.to_csv(qc_output_path, sep="\t", index=False)
        context.logger.info(
            "Study 2 %s %s: %d source-valid subjects.",
            log_label,
            band,
            len(result.subject_ids),
        )


def _source_power_by_subject(
    config: Any,
    *,
    subjects: tuple[str, ...],
    band: str,
) -> dict[str, np.ndarray]:
    return {
        subject_id: np.load(
            paths.subject_source_power_path(config, subject_id=subject_id, band=band)
        )
        for subject_id in subjects
    }


def source_stage_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return _source_map_required_inputs(context)


def run_source_stage(context: "Study2StageContext") -> None:
    """Compute per-subject and cohort source-power association maps per band."""
    _run_source_map_family(
        context,
        compute_maps=compute_cohort_source_association_maps,
        fisher_z_path=paths.source_stage_fisher_z_path,
        partial_r_path=lambda config, *, band: paths.source_stage_dir(config)
        / f"partial_r_{band}.npy",
        qc_path=lambda config, *, band: paths.source_stage_dir(config) / f"qc_{band}.tsv",
        log_label="source-stage",
    )


def band_unique_stage_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return _source_map_required_inputs(context)


def run_band_unique_stage(context: "Study2StageContext") -> None:
    """Compute secondary band-unique source-power association maps."""
    _run_source_map_family(
        context,
        compute_maps=compute_band_unique_cohort_source_association_maps,
        fisher_z_path=paths.band_unique_fisher_z_path,
        partial_r_path=paths.band_unique_partial_r_path,
        qc_path=paths.band_unique_qc_path,
        log_label="band-unique source-stage",
    )


def directional_consistency_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required: list[Path] = []
    for band in contribution_bands(config):
        required.extend(
            [
                paths.directional_prediction_map_path(config, band=band),
                paths.directional_target_map_path(config, band=band),
                paths.directional_cluster_mask_path(config, band=band),
            ]
        )
    return tuple(required)


def run_directional_consistency(context: "Study2StageContext") -> None:
    """Evaluate prediction-map consistency with true-target source maps."""
    config = context.config
    records = []
    for band in contribution_bands(config):
        qc = evaluate_directional_consistency(
            prediction_map=np.load(paths.directional_prediction_map_path(config, band=band)),
            target_map=np.load(paths.directional_target_map_path(config, band=band)),
            cluster_mask=np.load(paths.directional_cluster_mask_path(config, band=band)),
            config=config,
        )
        records.append(
            {
                "band": band,
                "spatial_r": qc.spatial_r,
                "same_sign_fraction": qc.same_sign_fraction,
                "directional_criteria_met": qc.directional_criteria_met,
                "unmet_criteria": ";".join(qc.unmet_criteria),
            }
        )
    paths.diagnostics_dir(config).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(
        paths.directional_consistency_summary_path(config),
        sep="\t",
        index=False,
    )


def artifact_controls_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.artifact_metrics_path(context.config),)


def run_artifact_controls(context: "Study2StageContext") -> None:
    """Evaluate artifact-control criteria from precomputed metrics."""
    config = context.config
    metrics = pd.read_csv(paths.artifact_metrics_path(config), sep="\t")
    require_columns(
        metrics,
        (
            "band",
            "metric",
            "sensor_template_abs_r",
            "source_artifact_map_abs_r",
            "expression_p_value",
        ),
        name="Study 2 artifact metrics",
    )

    records = []
    for band, band_frame in metrics.groupby("band", sort=True):
        qc = evaluate_artifact_controls(
            band=str(band),
            sensor_template_abs_r=metric_mapping(
                band_frame,
                value_column="sensor_template_abs_r",
            ),
            source_artifact_map_abs_r=metric_mapping(
                band_frame,
                value_column="source_artifact_map_abs_r",
            ),
            expression_p_values=metric_mapping(
                band_frame,
                value_column="expression_p_value",
            ),
            config=config,
        )
        records.append(
            {
                "band": qc.band,
                "artifact_control_criteria_met": qc.artifact_control_criteria_met,
                "unmet_criteria": ";".join(qc.unmet_criteria),
                "expression_q_values": format_mapping(qc.expression_q_values),
            }
        )

    paths.diagnostics_dir(config).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(
        paths.artifact_controls_summary_path(config),
        sep="\t",
        index=False,
    )


def robustness_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.robustness_metrics_path(context.config),)


def run_robustness(context: "Study2StageContext") -> None:
    """Evaluate robustness summaries from precomputed censoring metrics."""
    config = context.config
    metrics = pd.read_csv(paths.robustness_metrics_path(config), sep="\t")
    require_columns(
        metrics,
        (
            "band",
            "significance_retained",
            "sign_retained",
            "unthresholded_spatial_r",
            "cluster_dice",
            "centroid_displacement_mm",
        ),
        name="Study 2 robustness metrics",
    )

    records = []
    for row in metrics.to_dict("records"):
        qc = evaluate_robustness_summary(
            significance_retained=parse_bool(row["significance_retained"]),
            sign_retained=parse_bool(row["sign_retained"]),
            unthresholded_spatial_r=float(row["unthresholded_spatial_r"]),
            cluster_dice=float(row["cluster_dice"]),
            centroid_displacement_mm=float(row["centroid_displacement_mm"]),
            config=config,
        )
        records.append(
            {
                "band": str(row["band"]),
                "robustness_criteria_met": qc.robustness_criteria_met,
                "unmet_criteria": ";".join(qc.unmet_criteria),
            }
        )

    paths.diagnostics_dir(config).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(
        paths.robustness_summary_path(config),
        sep="\t",
        index=False,
    )


def spatial_correspondence_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required: list[Path] = [paths.spatial_mask_path(config)]
    for band in contribution_bands(config):
        required.extend(
            [
                paths.spatial_eeg_map_path(config, band=band),
                paths.spatial_fmri_map_path(config, band=band),
                paths.spatial_surrogate_maps_path(config, band=band),
            ]
        )
    return tuple(required)


def run_spatial_correspondence(context: "Study2StageContext") -> None:
    """Compute EEG/fMRI spatial-correspondence summaries from prepared maps."""
    config = context.config
    mask = np.load(paths.spatial_mask_path(config))
    records = []
    for band in contribution_bands(config):
        result = compute_spatial_correspondence(
            eeg_map=np.load(paths.spatial_eeg_map_path(config, band=band)),
            fmri_map=np.load(paths.spatial_fmri_map_path(config, band=band)),
            surrogate_maps=np.load(paths.spatial_surrogate_maps_path(config, band=band)),
            mask=mask,
            config=config,
        )
        records.append(
            {
                "band": band,
                "spatial_r": result.spatial_r,
                "p_value": result.p_value,
                "meaningful": result.meaningful,
            }
        )
    paths.spatial_dir(config).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(
        paths.spatial_correspondence_summary_path(config),
        sep="\t",
        index=False,
    )


def behavioral_convergence_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.behavioral_convergence_input_path(context.config),)


def run_behavioral_convergence(context: "Study2StageContext") -> None:
    """Run the secondary source-pattern behavioral convergence analysis."""
    config = context.config
    frame = pd.read_csv(paths.behavioral_convergence_input_path(config), sep="\t")
    expression_column = _required_config_string(
        config,
        "study2.behavioral_convergence.expression_column",
    )
    rating_column = _required_config_string(
        config,
        "study2.behavioral_convergence.rating_column",
    )
    design_columns = _required_config_string_tuple(
        config,
        "study2.behavioral_convergence.design_columns",
    )
    result = compute_behavioral_convergence(
        frame,
        expression_column=expression_column,
        rating_column=rating_column,
        design_columns=design_columns,
        config=config,
        random_state=require_config_int(config, "project.random_state"),
    )
    paths.behavioral_dir(config).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "mean_beta": result.mean_beta,
                "p_value": result.p_value,
                "n_subjects": result.n_subjects,
            }
        ]
    ).to_csv(paths.behavioral_convergence_summary_path(config), sep="\t", index=False)


def band_unique_inference_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required = [paths.source_adjacency_path(config)]
    for band in contribution_bands(config):
        required.append(paths.band_unique_fisher_z_path(config, band=band))
        required.append(paths.band_unique_null_maps_path(config, band=band))
    return tuple(required)


def run_band_unique_inference(context: "Study2StageContext") -> None:
    """Run the secondary band-unique source-family cluster test."""
    config = context.config
    bands = contribution_bands(config)
    adjacency = np.load(paths.source_adjacency_path(config))
    observed_maps = {
        band: np.load(paths.band_unique_fisher_z_path(config, band=band)) for band in bands
    }
    null_maps = {
        band: np.load(paths.band_unique_null_maps_path(config, band=band)) for band in bands
    }
    result = compute_source_family_inference(
        observed_maps_by_band=observed_maps,
        null_maps_by_band=null_maps,
        adjacency=adjacency,
        cluster_forming_p=require_config_float(
            config,
            "study2.source_inference.primary_cluster_forming_p",
        ),
        alpha=require_config_float(config, "study2.source_inference.family_alpha"),
    )

    paths.band_unique_dir(config).mkdir(parents=True, exist_ok=True)
    summarize_source_family(result).to_csv(
        paths.band_unique_family_summary_path(config),
        sep="\t",
        index=False,
    )


def _study1_capable_config(config: Any) -> Any:
    """Return a config carrying Study 1 defaults so the model context can load.

    The Study 2 CLI applies only Study 2 defaults, but the target-retrained null
    rebuilds the Study 1 model. Study 1 defaults are merged onto a copy without
    overwriting the runtime paths (``_merge_non_null`` only fills/merges). The
    Study 2-selected Study 1 output root is then mirrored into the Study 1
    namespace because Study 1 helper functions read ``study1.outputs.root_name``.
    """
    from eeg_pipeline.utils.config.loader import ConfigDict
    from studies.pain_study.study1.config.loader import apply_study1_config_defaults

    merged = ConfigDict(copy.deepcopy(dict(config)))
    apply_study1_config_defaults(merged)
    study1_root_name = _required_config_string(
        merged,
        "study2.inputs.study1_root_name",
    )
    merged.setdefault("study1", {}).setdefault("outputs", {})["root_name"] = study1_root_name
    return merged


def _observed_source_stage_subject_ids(
    frame: pd.DataFrame,
    source_power_by_band: Mapping[str, Mapping[str, np.ndarray]],
    bands: tuple[str, ...],
    config: Any,
) -> tuple[str, ...]:
    """Source-stage subject set from the observed combined score."""
    return compute_cohort_source_association_maps(
        frame,
        source_power_by_band[bands[0]],
        band=bands[0],
        config=config,
    ).subject_ids


def target_permutations_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required = [
        paths.source_stage_frame_path(config),
        study1_model_comparison_path(_study1_capable_config(config)),
    ]
    for subject_id in context.subjects:
        for band in contribution_bands(config):
            required.append(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
    return tuple(required)


def run_target_permutations(context: "Study2StageContext") -> None:
    """Build the target-retrained null source maps per band (README Section 6)."""
    config = context.config
    bands = contribution_bands(config)
    frame = pd.read_csv(paths.source_stage_frame_path(config), sep="\t")
    source_power_by_band = {
        band: {
            subject_id: np.load(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
            for subject_id in context.subjects
        }
        for band in bands
    }
    model_context = load_study1_model_context(
        subjects=list(context.subjects),
        task=context.task,
        config=_study1_capable_config(config),
        logger=context.logger,
    )
    expected_subject_ids = _observed_source_stage_subject_ids(
        frame, source_power_by_band, bands, config
    )

    null_by_band, result = build_target_retrained_null_maps(
        context=model_context,
        source_power_by_band=source_power_by_band,
        score_frame_template=frame,
        bands=bands,
        expected_subject_ids=expected_subject_ids,
        n_valid_draws=require_config_int(
            config,
            "study2.permutations.target_retrained_valid_draws",
        ),
        max_invalid_fraction=require_config_float(
            config,
            "study2.permutations.max_invalid_draw_fraction",
        ),
        random_state=require_config_int(config, "project.random_state"),
    )

    for band in bands:
        null_path = paths.null_source_maps_path(config, band=band)
        null_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(null_path, null_by_band[band])
    context.logger.info(
        "Study 2 target-permutations: %d valid null draws across %d bands (%d invalid).",
        result.n_valid_draws,
        len(bands),
        result.n_invalid_draws,
    )


def inference_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required = [paths.source_adjacency_path(config)]
    for band in contribution_bands(config):
        required.append(paths.source_stage_fisher_z_path(config, band=band))
        required.append(paths.null_source_maps_path(config, band=band))
    return tuple(required)


def run_inference(context: "Study2StageContext") -> None:
    """Run the primary source-family cluster test (README Section 6)."""
    config = context.config
    bands = contribution_bands(config)
    adjacency = np.load(paths.source_adjacency_path(config))
    observed_maps = {
        band: np.load(paths.source_stage_fisher_z_path(config, band=band)) for band in bands
    }
    null_maps = {band: np.load(paths.null_source_maps_path(config, band=band)) for band in bands}
    result = compute_source_family_inference(
        observed_maps_by_band=observed_maps,
        null_maps_by_band=null_maps,
        adjacency=adjacency,
        cluster_forming_p=require_config_float(
            config,
            "study2.source_inference.primary_cluster_forming_p",
        ),
        alpha=require_config_float(config, "study2.source_inference.family_alpha"),
    )

    summary = summarize_source_family(result)
    summary_path = paths.source_family_summary_path(config)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, sep="\t", index=False)
    context.logger.info(
        "Study 2 inference: %d of %d bands significant after Holm correction.",
        int(summary["significant"].sum()),
        len(summary),
    )


def _required_config_string(config: Any, key: str) -> str:
    return require_config_string(config, key)


def _required_config_string_tuple(config: Any, key: str) -> tuple[str, ...]:
    value = require_config_value(config, key)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list of column names.")
    parsed = tuple(str(item).strip() for item in value)
    if any(not item for item in parsed):
        raise ValueError(f"{key} must not contain empty column names.")
    if not parsed:
        raise ValueError(f"{key} must contain at least one column name.")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{key} must not contain duplicate column names.")
    return parsed


__all__ = [
    "artifact_controls_required_inputs",
    "behavioral_convergence_required_inputs",
    "band_unique_inference_required_inputs",
    "band_unique_stage_required_inputs",
    "directional_consistency_required_inputs",
    "gate_required_inputs",
    "haufe_required_inputs",
    "inference_required_inputs",
    "point_spread_required_inputs",
    "robustness_required_inputs",
    "run_artifact_controls",
    "run_behavioral_convergence",
    "run_band_unique_inference",
    "run_band_unique_stage",
    "run_directional_consistency",
    "run_gate",
    "run_haufe",
    "run_inference",
    "run_point_spread",
    "run_robustness",
    "run_source_model_qc",
    "run_source_power",
    "run_source_stage",
    "run_spatial_correspondence",
    "run_target_permutations",
    "source_model_qc_required_inputs",
    "source_power_required_inputs",
    "source_stage_required_inputs",
    "spatial_correspondence_required_inputs",
    "target_permutations_required_inputs",
]
