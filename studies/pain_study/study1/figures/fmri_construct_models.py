"""First- and second-level models for Study 1 fMRI construct validity."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from fmri_pipeline.analysis.trial_signatures import (
    _prepare_confounds_for_first_level_model,
)
from fmri_pipeline.utils.bold_discovery import (
    build_first_level_model,
    get_tr_from_bold,
    select_confounds_for_glm_from_path,
    slice_time_ref_from_bold,
    validate_design_matrices,
)
from studies.pain_study.study1.figures.fmri_construct_data import (
    FirstLevelRunDesign,
    FmriRunInput,
)


@dataclass(frozen=True)
class FirstLevelSettings:
    """Prespecified Study 1 first-level GLM settings."""

    hrf_model: str
    drift_model: str | None
    high_pass_hz: float
    low_pass_hz: float | None
    smoothing_fwhm: float | None
    confounds_strategy: str
    max_condition_number: float
    min_target_efficiency: float | None


@dataclass(frozen=True)
class SubjectEffectResult:
    """Participant effect images and their full design audit."""

    subject_id: str
    effect_images: Mapping[str, nib.Nifti1Image]
    analysis_mask: nib.Nifti1Image
    design_audit: pd.DataFrame


@dataclass(frozen=True)
class GroupInferenceSettings:
    """Participant-level permutation inference settings."""

    n_permutations: int
    two_sided: bool
    alpha: float
    random_state: int


@dataclass(frozen=True)
class GroupMapResult:
    """Effect, corrected probability, significance, and spatial audit."""

    estimand: str
    mean_effect: nib.Nifti1Image
    neg_log10_fwe_p: nib.Nifti1Image
    significance_mask: nib.Nifti1Image
    peaks: pd.DataFrame
    n_subjects: int


def first_level_settings(config: Any) -> FirstLevelSettings:
    """Resolve the fMRI validity GLM from the frozen Study 1 target model."""

    targets = require_config_value(config, "study1.targets")
    if not isinstance(targets, Mapping):
        raise ValueError("study1.targets must be a mapping.")
    required = (
        "hrf_model",
        "drift_model",
        "high_pass_hz",
        "low_pass_hz",
        "smoothing_fwhm",
        "confounds_strategy",
        "max_design_condition_number",
        "min_target_design_efficiency",
    )
    missing = [key for key in required if key not in targets]
    if missing:
        raise ValueError(f"study1.targets is missing fMRI GLM settings: {missing}.")
    return FirstLevelSettings(
        hrf_model=str(targets["hrf_model"]),
        drift_model=_optional_string(targets["drift_model"]),
        high_pass_hz=float(targets["high_pass_hz"]),
        low_pass_hz=_optional_float(targets["low_pass_hz"]),
        smoothing_fwhm=_optional_float(targets["smoothing_fwhm"]),
        confounds_strategy=str(targets["confounds_strategy"]),
        max_condition_number=float(targets["max_design_condition_number"]),
        min_target_efficiency=float(targets["min_target_design_efficiency"]),
    )


def fit_subject_effects(
    *,
    subject_runs: Sequence[FmriRunInput],
    designs: Sequence[FirstLevelRunDesign],
    settings: FirstLevelSettings,
) -> SubjectEffectResult:
    """Fit two explicit multi-run GLMs and return participant effect maps."""

    runs = _validated_subject_runs(subject_runs)
    subject_id = runs[0].subject_id
    design_by_estimand = _designs_by_estimand(designs, runs=runs, subject_id=subject_id)
    mask_image, repetition_time = _common_mask_and_tr(runs)
    slice_time_ref = _common_slice_time_ref(runs, tr=repetition_time)
    run_images = [str(run.bold_path) for run in runs]
    confounds, sample_masks = _load_confounds(runs, settings=settings)

    effects: dict[str, nib.Nifti1Image] = {}
    audit_frames: list[pd.DataFrame] = []
    for estimand in ("temperature", "rating"):
        estimand_designs = design_by_estimand[estimand]
        target_columns = {design.target_column for design in estimand_designs}
        if len(target_columns) != 1:
            raise ValueError(f"The {estimand} designs must use one target column.")
        target_column = next(iter(target_columns))
        model = build_first_level_model(
            tr=repetition_time,
            cfg=settings,
            mask_img=mask_image,
            slice_time_ref=slice_time_ref,
        )
        with warnings.catch_warnings():
            # SciPy 1.17 on macOS emits spurious matmul warnings while returning a
            # finite pseudoinverse; the explicit design and effect checks below
            # remain authoritative.
            for message in (
                "divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul",
            ):
                warnings.filterwarnings(
                    "ignore",
                    message=message,
                    category=RuntimeWarning,
                    module=r"scipy\.linalg\._basic",
                )
            warnings.filterwarnings(
                "ignore",
                message=r"\[MultiNiftiMasker\.fit\] Generation of a mask has been requested",
                category=RuntimeWarning,
            )
            model.fit(
                run_images,
                events=[design.events for design in estimand_designs],
                confounds=confounds,
                sample_masks=sample_masks,
            )
        validate_design_matrices(
            model,
            context=f"Study 1 fMRI {estimand} GLM ({subject_id})",
            min_residual_dof=1,
            max_condition_number=settings.max_condition_number,
            target_columns=(target_column,),
            min_target_efficiency=settings.min_target_efficiency,
        )
        effect_image = model.compute_contrast(target_column, output_type="effect_size")
        _validate_effect_image(effect_image, context=f"{subject_id} {estimand}")
        effects[estimand] = effect_image
        audit_frames.append(
            _design_audit(
                model.design_matrices_,
                designs=estimand_designs,
                runs=runs,
                sample_masks=sample_masks,
                target_column=target_column,
                repetition_time=repetition_time,
            )
        )
    return SubjectEffectResult(
        subject_id=subject_id,
        effect_images=effects,
        analysis_mask=mask_image,
        design_audit=pd.concat(audit_frames, ignore_index=True),
    )


def run_group_inference(
    effect_images: Sequence[nib.Nifti1Image],
    *,
    analysis_mask: nib.Nifti1Image,
    estimand: str,
    settings: GroupInferenceSettings,
) -> GroupMapResult:
    """Estimate the participant mean and two-sided voxelwise max-T FWE map."""

    from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference

    images = tuple(effect_images)
    _validate_group_inputs(images, analysis_mask=analysis_mask, settings=settings)
    design = pd.DataFrame({"intercept": np.ones(len(images), dtype=float)})
    image_list = list(images)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"\[NiftiMasker\.fit\] Generation of a mask has been requested .*"
                r"Given mask will be used\."
            ),
            category=UserWarning,
        )
        model = SecondLevelModel(mask_img=analysis_mask).fit(
            image_list,
            design_matrix=design,
        )
        mean_effect = model.compute_contrast(
            second_level_contrast="intercept",
            output_type="effect_size",
        )
        neg_log10_fwe_p = non_parametric_inference(
            second_level_input=image_list,
            design_matrix=design,
            second_level_contrast="intercept",
            mask=analysis_mask,
            model_intercept=False,
            n_perm=settings.n_permutations,
            two_sided_test=settings.two_sided,
            random_state=settings.random_state,
            n_jobs=1,
        )
    _validate_effect_image(mean_effect, context=f"group {estimand} mean")
    _validate_effect_image(neg_log10_fwe_p, context=f"group {estimand} max-T p")
    significance_mask = _significance_mask(
        neg_log10_fwe_p,
        analysis_mask=analysis_mask,
        alpha=settings.alpha,
    )
    return GroupMapResult(
        estimand=estimand,
        mean_effect=mean_effect,
        neg_log10_fwe_p=neg_log10_fwe_p,
        significance_mask=significance_mask,
        peaks=_spatial_peaks(
            mean_effect,
            significance_mask=significance_mask,
            estimand=estimand,
        ),
        n_subjects=len(images),
    )


def _validated_subject_runs(subject_runs: Sequence[FmriRunInput]) -> tuple[FmriRunInput, ...]:
    if not subject_runs:
        raise ValueError("Participant fMRI fitting requires at least one run.")
    ordered = tuple(sorted(subject_runs, key=lambda run: run.run))
    subjects = {run.subject_id for run in ordered}
    if len(subjects) != 1:
        raise ValueError("Participant fMRI fitting requires exactly one participant.")
    run_numbers = [run.run for run in ordered]
    if len(set(run_numbers)) != len(run_numbers):
        raise ValueError("Participant fMRI fitting contains duplicate runs.")
    return ordered


def _designs_by_estimand(
    designs: Sequence[FirstLevelRunDesign],
    *,
    runs: Sequence[FmriRunInput],
    subject_id: str,
) -> dict[str, tuple[FirstLevelRunDesign, ...]]:
    expected_runs = tuple(run.run for run in runs)
    output: dict[str, tuple[FirstLevelRunDesign, ...]] = {}
    for estimand in ("temperature", "rating"):
        selected = tuple(
            sorted(
                (design for design in designs if design.estimand == estimand),
                key=lambda design: design.run,
            )
        )
        observed_runs = tuple(design.run for design in selected)
        if observed_runs != expected_runs:
            raise ValueError(
                f"The {estimand} design runs do not match participant inputs: "
                f"observed={observed_runs}, expected={expected_runs}."
            )
        if any(design.subject_id != subject_id for design in selected):
            raise ValueError(f"The {estimand} designs contain a different participant.")
        output[estimand] = selected
    return output


def _common_slice_time_ref(runs: Sequence[FmriRunInput], *, tr: float) -> float:
    references = [slice_time_ref_from_bold(run.bold_path, tr=tr) for run in runs]
    if not np.allclose(references, references[0], rtol=0.0, atol=1e-6):
        raise ValueError("Participant fMRI runs must have one slice-timing reference.")
    return float(references[0])


def _common_mask_and_tr(
    runs: Sequence[FmriRunInput],
) -> tuple[nib.Nifti1Image, float]:
    reference_bold = nib.load(str(runs[0].bold_path))
    if len(reference_bold.shape) != 4:
        raise ValueError(f"fMRI BOLD input must be 4D: {runs[0].bold_path}.")
    reference_shape = reference_bold.shape[:3]
    reference_affine = np.asarray(reference_bold.affine, dtype=float)
    mask_values: list[np.ndarray] = []
    repetition_times: list[float] = []
    for run in runs:
        bold = nib.load(str(run.bold_path))
        mask = nib.load(str(run.mask_path))
        if len(bold.shape) != 4 or bold.shape[:3] != reference_shape:
            raise ValueError("Participant fMRI BOLD runs must share one 3D grid.")
        if mask.shape != reference_shape:
            raise ValueError("Participant fMRI masks must match the BOLD grid.")
        if not np.allclose(bold.affine, reference_affine, rtol=0.0, atol=1e-5):
            raise ValueError("Participant fMRI BOLD runs must share one affine.")
        if not np.allclose(mask.affine, reference_affine, rtol=0.0, atol=1e-5):
            raise ValueError("Participant fMRI masks must share the BOLD affine.")
        values = np.asanyarray(mask.dataobj)
        if not np.isfinite(values).all():
            raise ValueError(f"fMRIPrep brain mask contains non-finite values: {run.mask_path}.")
        mask_values.append(values > 0)
        repetition_times.append(get_tr_from_bold(run.bold_path))
    if not np.allclose(repetition_times, repetition_times[0], rtol=0.0, atol=1e-6):
        raise ValueError("Participant fMRI runs must have one repetition time.")
    intersection = np.logical_and.reduce(mask_values)
    if not intersection.any():
        raise ValueError("Participant fMRI brain-mask intersection is empty.")
    mask_image = nib.Nifti1Image(
        intersection.astype(np.uint8),
        reference_affine,
        reference_bold.header,
    )
    return mask_image, float(repetition_times[0])


def _load_confounds(
    runs: Sequence[FmriRunInput],
    *,
    settings: FirstLevelSettings,
) -> tuple[list[pd.DataFrame] | None, list[np.ndarray | None] | None]:
    strategy = settings.confounds_strategy.strip().lower()
    if strategy in {"none", "no", "off"}:
        return None, None
    confounds: list[pd.DataFrame] = []
    sample_masks: list[np.ndarray | None] = []
    for run in runs:
        selected, columns, sample_mask = select_confounds_for_glm_from_path(
            run.confounds_path,
            strategy,
        )
        if selected is None or not columns:
            raise ValueError(
                "Study 1 fMRI construct validity requires selected confounds for every run: "
                f"{run.confounds_path}."
            )
        n_scans = int(nib.load(str(run.bold_path)).shape[3])
        if len(selected) != n_scans:
            raise ValueError(
                f"Confound rows do not match BOLD scans for {run.bold_path}: "
                f"confounds={len(selected)}, scans={n_scans}."
            )
        prepared = _prepare_confounds_for_first_level_model(selected, sample_mask)
        if prepared is None:
            raise ValueError(f"Confound preparation returned no regressors for {run.bold_path}.")
        confounds.append(prepared)
        sample_masks.append(sample_mask)
    return confounds, sample_masks


def _design_audit(
    design_matrices: Sequence[pd.DataFrame],
    *,
    designs: Sequence[FirstLevelRunDesign],
    runs: Sequence[FmriRunInput],
    sample_masks: Sequence[np.ndarray | None] | None,
    target_column: str,
    repetition_time: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    masks = sample_masks or [None] * len(runs)
    for matrix, design, run, sample_mask in zip(
        design_matrices,
        designs,
        runs,
        masks,
        strict=True,
    ):
        values = matrix.to_numpy(dtype=float)
        rank = int(np.linalg.matrix_rank(values))
        condition_number = float(np.linalg.cond(values))
        target_index = list(matrix.columns).index(target_column)
        contrast = np.zeros(values.shape[1], dtype=float)
        contrast[target_index] = 1.0
        _orthogonal, upper = np.linalg.qr(values, mode="reduced")
        normalized_contrast = np.linalg.solve(upper.T, contrast)
        variance_factor = float(normalized_contrast @ normalized_contrast)
        efficiency = math.inf if variance_factor <= 0 else 1.0 / variance_factor
        rows.append(
            {
                "subject_id": run.subject_id,
                "run": run.run,
                "estimand": design.estimand,
                "target_column": target_column,
                "n_scans": int(nib.load(str(run.bold_path)).shape[3]),
                "n_retained_scans": (
                    int(len(sample_mask))
                    if sample_mask is not None
                    else int(nib.load(str(run.bold_path)).shape[3])
                ),
                "n_events": int(design.audit["n_events"]),
                "n_retained_trials": int(design.audit["n_retained_trials"]),
                "n_regressors": values.shape[1],
                "rank": rank,
                "residual_dof": values.shape[0] - rank,
                "condition_number": condition_number,
                "target_efficiency": efficiency,
                "repetition_time_s": repetition_time,
                "slice_time_ref": slice_time_ref_from_bold(run.bold_path, tr=repetition_time),
                "bold_path": str(run.bold_path),
                "mask_path": str(run.mask_path),
                "confounds_path": str(run.confounds_path),
                "events_path": str(run.raw_events_path),
            }
        )
    return pd.DataFrame(rows)


def _validate_effect_image(image: nib.Nifti1Image, *, context: str) -> None:
    if len(image.shape) != 3:
        raise ValueError(f"{context} effect image must be 3D, got {image.shape}.")
    values = np.asanyarray(image.dataobj, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} effect image contains non-finite values.")


def _validate_group_inputs(
    images: Sequence[nib.Nifti1Image],
    *,
    analysis_mask: nib.Nifti1Image,
    settings: GroupInferenceSettings,
) -> None:
    if len(images) < 2:
        raise ValueError("Group fMRI inference requires at least two participants.")
    if settings.n_permutations < 1:
        raise ValueError("Group fMRI inference requires at least one permutation.")
    if not 0.0 < settings.alpha < 1.0:
        raise ValueError("Group fMRI inference alpha must be within (0, 1).")
    if not settings.two_sided:
        raise ValueError("Study 1 fMRI construct validity requires two-sided inference.")
    if len(analysis_mask.shape) != 3:
        raise ValueError("Group fMRI analysis mask must be 3D.")
    reference_shape = analysis_mask.shape
    reference_affine = np.asarray(analysis_mask.affine, dtype=float)
    mask_values = np.asanyarray(analysis_mask.dataobj)
    if not np.isfinite(mask_values).all() or not np.any(mask_values > 0):
        raise ValueError("Group fMRI analysis mask must be finite and non-empty.")
    for image in images:
        if image.shape != reference_shape:
            raise ValueError("Participant fMRI effect images must match the group mask shape.")
        if not np.allclose(image.affine, reference_affine, rtol=0.0, atol=1e-5):
            raise ValueError("Participant fMRI effect images must match the group mask affine.")
        _validate_effect_image(image, context="participant group input")


def _significance_mask(
    neg_log10_fwe_p: nib.Nifti1Image,
    *,
    analysis_mask: nib.Nifti1Image,
    alpha: float,
) -> nib.Nifti1Image:
    corrected = np.asanyarray(neg_log10_fwe_p.dataobj, dtype=float)
    mask = np.asanyarray(analysis_mask.dataobj) > 0
    significant = mask & (corrected >= -math.log10(alpha))
    return nib.Nifti1Image(
        significant.astype(np.uint8),
        analysis_mask.affine,
        analysis_mask.header,
    )


def _spatial_peaks(
    mean_effect: nib.Nifti1Image,
    *,
    significance_mask: nib.Nifti1Image,
    estimand: str,
) -> pd.DataFrame:
    from scipy.ndimage import generate_binary_structure, label

    columns = (
        "estimand",
        "cluster_id",
        "sign",
        "peak_effect",
        "peak_x_mm",
        "peak_y_mm",
        "peak_z_mm",
        "n_voxels",
    )
    effects = np.asanyarray(mean_effect.dataobj, dtype=float)
    significant = np.asanyarray(significance_mask.dataobj) > 0
    structure = generate_binary_structure(rank=3, connectivity=1)
    rows: list[dict[str, object]] = []
    cluster_id = 0
    for sign, signed_mask in (
        ("positive", significant & (effects > 0)),
        ("negative", significant & (effects < 0)),
    ):
        labels, count = label(signed_mask, structure=structure)
        for component in range(1, count + 1):
            voxels = np.argwhere(labels == component)
            if voxels.size == 0:
                raise ValueError("Spatial cluster labeling produced an empty component.")
            component_effects = effects[tuple(voxels.T)]
            peak_voxel = voxels[int(np.argmax(np.abs(component_effects)))]
            peak_world = nib.affines.apply_affine(mean_effect.affine, peak_voxel)
            cluster_id += 1
            rows.append(
                {
                    "estimand": estimand,
                    "cluster_id": cluster_id,
                    "sign": sign,
                    "peak_effect": float(effects[tuple(peak_voxel)]),
                    "peak_x_mm": float(peak_world[0]),
                    "peak_y_mm": float(peak_world[1]),
                    "peak_z_mm": float(peak_world[2]),
                    "n_voxels": int(len(voxels)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Optional fMRI frequency/smoothing values must be positive and finite.")
    return number


__all__ = [
    "FirstLevelSettings",
    "GroupInferenceSettings",
    "GroupMapResult",
    "SubjectEffectResult",
    "first_level_settings",
    "fit_subject_effects",
    "run_group_inference",
]
