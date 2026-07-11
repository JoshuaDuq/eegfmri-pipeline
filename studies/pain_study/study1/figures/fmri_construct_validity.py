"""End-to-end orchestration for Study 1 whole-brain fMRI construct validity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.fmri_construct_data import (
    build_subject_designs,
    load_fmri_run_inputs,
)
from studies.pain_study.study1.figures.fmri_construct_models import (
    GroupInferenceSettings,
    GroupMapResult,
    first_level_settings,
    fit_subject_effects,
    run_group_inference,
)

FIGURE_CONFIG_KEY = "study1.figures.fmri_construct_validity"
ESTIMANDS = ("temperature", "rating")


@dataclass(frozen=True)
class FmriConstructValiditySummary:
    """Plot-ready whole-brain maps and complete participant-level audits."""

    subjects: pd.DataFrame
    design_audit: pd.DataFrame
    subject_effects: Mapping[str, tuple[nib.Nifti1Image, ...]]
    group_maps: Mapping[str, GroupMapResult]
    n_subjects: int
    article_ready: bool


def build_fmri_construct_validity_summary(
    *,
    task: str,
    config: Any,
) -> FmriConstructValiditySummary:
    """Fit both estimands for the retained cohort and run group inference."""

    figure_config = _figure_config(config)
    run_inputs = load_fmri_run_inputs(task=task, config=config)
    subject_ids = tuple(sorted({run.subject_id for run in run_inputs}))
    if len(subject_ids) < 2:
        raise ValueError("Study 1 fMRI construct validity requires at least two participants.")

    settings = first_level_settings(config)
    results = []
    subject_rows = []
    for subject_id in subject_ids:
        subject_runs = tuple(run for run in run_inputs if run.subject_id == subject_id)
        designs = build_subject_designs(subject_runs)
        result = fit_subject_effects(
            subject_runs=subject_runs,
            designs=designs,
            settings=settings,
        )
        if result.subject_id != subject_id:
            raise ValueError("Participant fMRI result order does not match the retained cohort.")
        if set(result.effect_images) != set(ESTIMANDS):
            raise ValueError(f"Participant {subject_id} is missing a construct-validity estimand.")
        results.append(result)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "n_runs": len(subject_runs),
                "temperature_estimable": True,
                "rating_estimable": True,
            }
        )

    common_mask = _common_group_mask([result.analysis_mask for result in results])
    inference = _group_inference_settings(figure_config)
    subject_effects = {
        estimand: tuple(result.effect_images[estimand] for result in results)
        for estimand in ESTIMANDS
    }
    group_maps = {
        estimand: run_group_inference(
            subject_effects[estimand],
            analysis_mask=common_mask,
            estimand=estimand,
            settings=inference,
        )
        for estimand in ESTIMANDS
    }
    observed_group_counts = {result.n_subjects for result in group_maps.values()}
    if observed_group_counts != {len(subject_ids)}:
        raise ValueError("Group fMRI estimands do not contain the identical participant cohort.")

    minimum_article_subjects = int(figure_config["minimum_article_subjects"])
    if minimum_article_subjects < 2:
        raise ValueError("minimum_article_subjects must be at least two.")
    return FmriConstructValiditySummary(
        subjects=pd.DataFrame(subject_rows),
        design_audit=pd.concat([result.design_audit for result in results], ignore_index=True),
        subject_effects=subject_effects,
        group_maps=group_maps,
        n_subjects=len(subject_ids),
        article_ready=len(subject_ids) >= minimum_article_subjects,
    )


def _common_group_mask(masks: list[nib.Nifti1Image]) -> nib.Nifti1Image:
    if not masks:
        raise ValueError("Group fMRI inference requires participant analysis masks.")
    reference = masks[0]
    values = []
    for mask in masks:
        if mask.shape != reference.shape or not np.allclose(
            mask.affine,
            reference.affine,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError("Participant analysis masks must share one MNI grid.")
        data = np.asanyarray(mask.dataobj)
        if not np.isfinite(data).all():
            raise ValueError("Participant analysis masks must contain finite values.")
        values.append(data > 0)
    intersection = np.logical_and.reduce(values)
    if not intersection.any():
        raise ValueError("The retained participant group brain-mask intersection is empty.")
    return nib.Nifti1Image(
        intersection.astype(np.uint8),
        reference.affine,
        reference.header,
    )


def _group_inference_settings(
    figure_config: Mapping[str, object],
) -> GroupInferenceSettings:
    inference = figure_config["inference"]
    if not isinstance(inference, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY}.inference must be a mapping.")
    return GroupInferenceSettings(
        n_permutations=int(inference["n_permutations"]),
        two_sided=bool(inference["two_sided"]),
        alpha=float(inference["alpha"]),
        random_state=int(inference["random_state"]),
    )


def _figure_config(config: Any) -> Mapping[str, Any]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


__all__ = [
    "FmriConstructValiditySummary",
    "build_fmri_construct_validity_summary",
]
