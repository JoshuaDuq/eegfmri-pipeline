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
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value
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
    evaluate_study1_confirmatory_gates,
    load_study1_confirmatory_row,
)
from studies.pain_study.study2.haufe import compute_haufe_pattern
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
from studies.pain_study.study2.source_power import (
    apply_sloreta_inverse,
    build_surface_forward_model,
    compute_baseline_noise_covariance,
    compute_sloreta_hilbert_logratio_power,
    make_sloreta_inverse_operator,
)
from studies.pain_study.study2.study1_context import (
    load_study1_model_context,
    study1_model_comparison_path,
)
from studies.pain_study.study2.spatial_comparison import compute_spatial_correspondence
from studies.pain_study.study2.target_retrained_null import build_target_retrained_null_maps

if TYPE_CHECKING:
    from studies.pain_study.study2.runner import Study2StageContext


def _contribution_bands(config: Any) -> tuple[str, ...]:
    bands = get_config_value(
        config,
        "study2.confirmatory.study1_cell.contribution_bands",
        ["alpha", "beta", "gamma"],
    )
    if not isinstance(bands, (list, tuple)) or not bands:
        raise ValueError("study2.confirmatory.study1_cell.contribution_bands must be non-empty.")
    return tuple(str(band).strip().lower() for band in bands)


def _band_frequency_ranges(config: Any) -> dict[str, tuple[float, float]]:
    bands = get_config_value(config, "study2.source_modeling.frequency_bands", None)
    if not isinstance(bands, Mapping):
        raise ValueError("study2.source_modeling.frequency_bands must be a mapping.")
    ranges: dict[str, tuple[float, float]] = {}
    for band in _contribution_bands(config):
        bounds = bands.get(band)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"study2.source_modeling.frequency_bands.{band} must be [low, high].")
        low, high = float(bounds[0]), float(bounds[1])
        if high <= low:
            raise ValueError(f"study2.source_modeling.frequency_bands.{band} must have high > low.")
        ranges[band] = (low, high)
    return ranges


def _time_window(config: Any, key: str) -> tuple[float, float]:
    window = get_config_value(config, key, None)
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f"{key} must be a [start, stop] window.")
    return float(window[0]), float(window[1])


@dataclass(frozen=True)
class _AnatomyPaths:
    subjects_dir: Path
    trans: Path
    bem: Path


def _resolve_anatomy(config: Any, *, subject_id: str) -> _AnatomyPaths:
    subjects_dir = get_config_value(config, "study2.source_modeling.anatomy.subjects_dir", None)
    trans_template = get_config_value(
        config, "study2.source_modeling.anatomy.trans_path_template", None
    )
    bem_template = get_config_value(
        config, "study2.source_modeling.anatomy.bem_path_template", None
    )
    if not subjects_dir or not trans_template or not bem_template:
        raise ValueError(
            "Study 2 source-power requires study2.source_modeling.anatomy.subjects_dir, "
            "trans_path_template, and bem_path_template to be configured."
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
        for channel in get_config_value(config, "study2.source_modeling.rank_excluded_channels", [])
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


def run_source_power(context: "Study2StageContext") -> None:
    """Reconstruct per-subject sLORETA source power per band from cleaned epochs."""
    config = context.config
    band_ranges = _band_frequency_ranges(config)
    baseline_window = _time_window(config, "study2.source_modeling.noise_covariance_baseline_s")
    active_window = _time_window(config, "study2.source_modeling.active_plateau_window_s")
    snr = get_config_value(config, "study2.source_modeling.regularization.snr", None)
    loose = get_config_value(config, "study2.source_modeling.regularization.loose_orientation", None)
    depth = get_config_value(config, "study2.source_modeling.regularization.depth_weighting", None)
    spacing = str(get_config_value(config, "study2.source_modeling.source_space_spacing", "oct6"))
    mindist_mm = get_config_value(config, "study2.source_modeling.forward_mindist_mm", None)

    for subject_id in context.subjects:
        anatomy = _resolve_anatomy(config, subject_id=subject_id)
        epochs = _load_subject_epochs(config, subject_id=subject_id, task=context.task, logger=context.logger)
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

        for band, (low, high) in band_ranges.items():
            band_epochs = epochs.copy().filter(low, high, verbose=False)
            stcs = apply_sloreta_inverse(
                epochs=band_epochs,
                inverse_operator=inverse_operator,
                snr=snr,
                pick_ori="normal",
            )
            extraction = compute_sloreta_hilbert_logratio_power(
                stcs=stcs,
                times=np.asarray(epochs.times, dtype=float),
                baseline_window_s=baseline_window,
                active_window_s=active_window,
            )
            power_path = paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            power_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(power_path, extraction.power_logratio)
            context.logger.info(
                "Study 2 source-power %s %s: %d trials x %d vertices.",
                subject_id,
                band,
                extraction.n_trials,
                extraction.n_vertices,
            )
            del band_epochs, stcs, extraction
            gc.collect()
        del forward, noise_cov, inverse_operator, epochs
        gc.collect()


def gate_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.study1_report_path(context.config),)


def run_gate(context: "Study2StageContext") -> None:
    """Evaluate Study 1 confirmatory gates and persist the eligibility decision."""
    config = context.config
    metrics = load_study1_confirmatory_row(
        paths.study1_report_path(config),
        config=config,
    )
    qc = evaluate_study1_confirmatory_gates(metrics, config)

    gate_path = paths.gate_qc_path(config)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "confirmatory_eligible": qc.confirmatory_eligible,
        "failed_gates": list(qc.failed_gates),
        "reason": qc.reason,
    }
    with open(gate_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    context.logger.info(
        "Study 2 gate: confirmatory_eligible=%s (%s)",
        qc.confirmatory_eligible,
        qc.reason,
    )


def haufe_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.haufe_input_path(context.config),)


def run_haufe(context: "Study2StageContext") -> None:
    """Compute the Haufe sensor pattern from a persisted training-fold design."""
    config = context.config
    with np.load(paths.haufe_input_path(config)) as payload:
        _require_npz_keys(payload, ("X_train", "coefficients"))
        result = compute_haufe_pattern(
            payload["X_train"],
            payload["coefficients"],
        )

    sensor_dir = paths.sensor_dir(config)
    sensor_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.haufe_pattern_path(config), result.pattern)
    np.save(paths.haufe_covariance_path(config), result.feature_covariance)
    pd.DataFrame(
        [
            {
                "n_observations": result.n_observations,
                "n_features": result.n_features,
            }
        ]
    ).to_csv(paths.haufe_summary_path(config), sep="\t", index=False)


def source_model_qc_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    return (paths.source_model_metrics_path(context.config),)


def run_source_model_qc(context: "Study2StageContext") -> None:
    """Evaluate source-model reconstruction metrics for each subject."""
    config = context.config
    metrics = pd.read_csv(paths.source_model_metrics_path(config), sep="\t")
    _require_columns(metrics, ("subject_id",), name="Study 2 source-model metrics")

    records = []
    for row in metrics.to_dict("records"):
        subject_id = str(row.pop("subject_id"))
        qc = evaluate_source_model_qc(subject_id=subject_id, metrics=row, config=config)
        records.append(
            {
                "subject_id": qc.subject_id,
                "eligible": qc.eligible,
                "valid_eeg_channel_location_fraction": qc.valid_eeg_channel_location_fraction,
                "mean_coregistration_error_mm": qc.mean_coregistration_error_mm,
                "max_coregistration_error_mm": qc.max_coregistration_error_mm,
                "failed_criteria": ";".join(qc.failed_gates),
                "reason": qc.reason,
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


def source_stage_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required = [paths.source_stage_frame_path(config)]
    for subject_id in context.subjects:
        for band in _contribution_bands(config):
            required.append(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
    return tuple(required)


def run_source_stage(context: "Study2StageContext") -> None:
    """Compute per-subject and cohort source-power association maps per band."""
    config = context.config
    frame = pd.read_csv(paths.source_stage_frame_path(config), sep="\t")
    output_dir = paths.source_stage_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    for band in _contribution_bands(config):
        source_power_by_subject = {
            subject_id: np.load(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
            for subject_id in context.subjects
        }
        result = compute_cohort_source_association_maps(
            frame,
            source_power_by_subject,
            band=band,
            config=config,
        )
        np.save(paths.source_stage_fisher_z_path(config, band=band), result.fisher_z_maps)
        np.save(output_dir / f"partial_r_{band}.npy", result.partial_r_maps)
        result.qc.to_csv(output_dir / f"qc_{band}.tsv", sep="\t", index=False)
        context.logger.info(
            "Study 2 source-stage %s: %d source-valid subjects.",
            band,
            len(result.subject_ids),
        )


def band_unique_stage_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required = [paths.source_stage_frame_path(config)]
    for subject_id in context.subjects:
        for band in _contribution_bands(config):
            required.append(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
    return tuple(required)


def run_band_unique_stage(context: "Study2StageContext") -> None:
    """Compute secondary band-unique source-power association maps."""
    config = context.config
    frame = pd.read_csv(paths.source_stage_frame_path(config), sep="\t")
    output_dir = paths.band_unique_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    for band in _contribution_bands(config):
        source_power_by_subject = {
            subject_id: np.load(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
            for subject_id in context.subjects
        }
        result = compute_band_unique_cohort_source_association_maps(
            frame,
            source_power_by_subject,
            band=band,
            config=config,
        )
        np.save(paths.band_unique_fisher_z_path(config, band=band), result.fisher_z_maps)
        np.save(paths.band_unique_partial_r_path(config, band=band), result.partial_r_maps)
        result.qc.to_csv(paths.band_unique_qc_path(config, band=band), sep="\t", index=False)


def directional_consistency_required_inputs(context: "Study2StageContext") -> tuple[Path, ...]:
    config = context.config
    required: list[Path] = []
    for band in _contribution_bands(config):
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
    for band in _contribution_bands(config):
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
                "passed": qc.passed,
                "failed_criteria": ";".join(qc.failed_gates),
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
    """Evaluate artifact interpretation controls from precomputed metrics."""
    config = context.config
    metrics = pd.read_csv(paths.artifact_metrics_path(config), sep="\t")
    _require_columns(
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
            sensor_template_abs_r=_metric_mapping(
                band_frame,
                value_column="sensor_template_abs_r",
            ),
            source_artifact_map_abs_r=_metric_mapping(
                band_frame,
                value_column="source_artifact_map_abs_r",
            ),
            expression_p_values=_metric_mapping(
                band_frame,
                value_column="expression_p_value",
            ),
            config=config,
        )
        records.append(
            {
                "band": qc.band,
                "contaminated": qc.contaminated,
                "interpretation": qc.interpretation,
                "failed_criteria": ";".join(qc.failed_gates),
                "expression_q_values": _format_mapping(qc.expression_q_values),
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
    _require_columns(
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
            significance_retained=_bool_value(row["significance_retained"]),
            sign_retained=_bool_value(row["sign_retained"]),
            unthresholded_spatial_r=float(row["unthresholded_spatial_r"]),
            cluster_dice=float(row["cluster_dice"]),
            centroid_displacement_mm=float(row["centroid_displacement_mm"]),
            config=config,
        )
        records.append(
            {
                "band": str(row["band"]),
                "passed": qc.passed,
                "failed_criteria": ";".join(qc.failed_gates),
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
    for band in _contribution_bands(config):
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
    for band in _contribution_bands(config):
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
        random_state=int(get_config_value(config, "project.random_state", 42)),
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
    for band in _contribution_bands(config):
        required.append(paths.band_unique_fisher_z_path(config, band=band))
        required.append(paths.band_unique_null_maps_path(config, band=band))
    return tuple(required)


def run_band_unique_inference(context: "Study2StageContext") -> None:
    """Run the secondary band-unique source-family cluster test."""
    config = context.config
    bands = _contribution_bands(config)
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
        cluster_forming_p=get_config_value(
            config, "study2.source_inference.primary_cluster_forming_p", None
        ),
        alpha=get_config_value(config, "study2.source_inference.family_alpha", 0.05),
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
    overwriting the runtime paths (``_merge_non_null`` only fills/merges).
    """
    from eeg_pipeline.utils.config.loader import ConfigDict
    from studies.pain_study.study1.config.loader import apply_study1_config_defaults

    merged = ConfigDict(copy.deepcopy(dict(config)))
    apply_study1_config_defaults(merged)
    return merged


def _observed_eligible_subject_ids(
    frame: pd.DataFrame,
    source_power_by_band: Mapping[str, Mapping[str, np.ndarray]],
    bands: tuple[str, ...],
    config: Any,
) -> tuple[str, ...]:
    """Eligible cohort from the observed combined score (band-independent)."""
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
        study1_model_comparison_path(config),
    ]
    for subject_id in context.subjects:
        for band in _contribution_bands(config):
            required.append(
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            )
    return tuple(required)


def run_target_permutations(context: "Study2StageContext") -> None:
    """Build the target-retrained null source maps per band (README Section 6)."""
    config = context.config
    bands = _contribution_bands(config)
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
    expected_subject_ids = _observed_eligible_subject_ids(
        frame, source_power_by_band, bands, config
    )

    null_by_band, result = build_target_retrained_null_maps(
        context=model_context,
        source_power_by_band=source_power_by_band,
        score_frame_template=frame,
        bands=bands,
        expected_subject_ids=expected_subject_ids,
        n_valid_draws=int(
            get_config_value(config, "study2.permutations.target_retrained_valid_draws", None)
        ),
        max_invalid_fraction=float(
            get_config_value(config, "study2.permutations.max_invalid_draw_fraction", None)
        ),
        random_state=int(get_config_value(config, "project.random_state", 42)),
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
    for band in _contribution_bands(config):
        required.append(paths.source_stage_fisher_z_path(config, band=band))
        required.append(paths.null_source_maps_path(config, band=band))
    return tuple(required)


def run_inference(context: "Study2StageContext") -> None:
    """Run the primary source-family cluster test (README Section 6)."""
    config = context.config
    bands = _contribution_bands(config)
    adjacency = np.load(paths.source_adjacency_path(config))
    observed_maps = {
        band: np.load(paths.source_stage_fisher_z_path(config, band=band)) for band in bands
    }
    null_maps = {
        band: np.load(paths.null_source_maps_path(config, band=band)) for band in bands
    }
    result = compute_source_family_inference(
        observed_maps_by_band=observed_maps,
        null_maps_by_band=null_maps,
        adjacency=adjacency,
        cluster_forming_p=get_config_value(
            config, "study2.source_inference.primary_cluster_forming_p", None
        ),
        alpha=get_config_value(config, "study2.source_inference.family_alpha", 0.05),
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


def _require_npz_keys(payload: Any, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload.files]
    if missing:
        raise ValueError(f"Study 2 NPZ input is missing arrays: {missing}.")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}.")


def _metric_mapping(frame: pd.DataFrame, *, value_column: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in frame.to_dict("records"):
        value = row[value_column]
        if pd.isna(value):
            continue
        values[str(row["metric"])] = float(value)
    return values


def _format_mapping(values: Mapping[str, float]) -> str:
    return ";".join(f"{key}={values[key]:.12g}" for key in sorted(values))


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def _required_config_string(config: Any, key: str) -> str:
    value = get_config_value(config, key, None)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{key} must be configured.")
    return text


def _required_config_string_tuple(config: Any, key: str) -> tuple[str, ...]:
    value = get_config_value(config, key, None)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list of column names.")
    parsed = tuple(str(item).strip() for item in value if str(item).strip())
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
