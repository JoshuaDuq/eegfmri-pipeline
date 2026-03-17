from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.contrast_builder import (
    ContrastBuilderConfig,
    _apply_trial_phase_scoping,
    _assert_constraint_mask_requires_z_output,
    _build_intersection_brain_mask,
    _get_constraint_mask_hash,
    _load_constraint_mask_spec,
    _load_matching_brain_mask_for_bold,
    _prepare_events_for_glm,
    _resolve_fmri_stats_artifact,
    _validate_consistent_trs,
    _validate_events_against_bold_run,
    build_contrast_from_runs_detailed,
    compute_contrast_map,
    discover_bold_runs,
    load_contrast_config,
    load_contrast_config_section,
    _remap_events_by_condition_columns,
)
from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
from fmri_pipeline.analysis.reporting import run_fmri_plotting_and_report
from fmri_pipeline.analysis.trial_signatures import (
    TrialInfo,
    TrialSignatureExtractionConfig,
    _combine_effect_images,
    _build_lss_events,
    _discover_runs,
    _extract_trials_for_run,
    run_trial_signature_extraction_for_subject,
)


def test_load_contrast_config_section_reads_top_level_fmri_contrast() -> None:
    config = {
        "fmri_contrast": {
            "name": "yaml-contrast",
            "condition_a": {"column": "trial_type", "value": "pain"},
        }
    }

    section = load_contrast_config_section(config)

    assert section["name"] == "yaml-contrast"
    assert section["condition_a"]["value"] == "pain"


def test_load_contrast_config_section_prefers_nested_source_localization_config() -> None:
    config = {
        "fmri_contrast": {"name": "top-level"},
        "feature_engineering": {
            "sourcelocalization": {
                "fmri": {
                    "contrast": {"name": "nested"}
                }
            }
        },
    }

    section = load_contrast_config_section(config)

    assert section["name"] == "nested"


def test_load_contrast_config_rejects_legacy_nested_condition_keys() -> None:
    config = {
        "feature_engineering": {
            "sourcelocalization": {
                "fmri": {
                    "contrast": {
                        "type": "t-test",
                        "cond_a": {"column": "trial_type", "value": "pain"},
                    }
                }
            }
        }
    }

    with pytest.raises(ValueError, match="Use fmri_contrast.condition_a"):
        load_contrast_config(config)


def test_load_contrast_config_rejects_unsupported_contrast_types() -> None:
    config = {
        "fmri_contrast": {
            "type": "paired-t-test",
            "condition_a": {"column": "trial_type", "value": "pain"},
        }
    }

    with pytest.raises(ValueError, match="Supported values: custom, t-test"):
        load_contrast_config(config)


def test_load_contrast_config_rejects_invalid_input_source() -> None:
    config = {
        "fmri_contrast": {
            "input_source": "mystery",
            "condition_a": {"column": "trial_type", "value": "pain"},
        }
    }

    with pytest.raises(ValueError, match="Unsupported fmri input_source"):
        load_contrast_config(config)


def test_load_contrast_config_rejects_invalid_output_type() -> None:
    config = {
        "fmri_contrast": {
            "output_type": "mystery",
            "condition_a": {"column": "trial_type", "value": "pain"},
        }
    }

    with pytest.raises(ValueError, match="Unsupported fmri output_type"):
        load_contrast_config(config)


def test_load_contrast_config_requires_events_to_model_column() -> None:
    config = {
        "fmri_contrast": {
            "condition_a": {"column": "trial_type", "value": "pain"},
            "events_to_model": ["stimulation"],
            "events_to_model_column": "",
        }
    }

    with pytest.raises(ValueError, match="events_to_model_column"):
        load_contrast_config(config)


def test_load_contrast_config_requires_condition_scope_column() -> None:
    config = {
        "fmri_contrast": {
            "condition_a": {"column": "trial_type", "value": "pain"},
            "condition_scope_trial_types": ["stimulation"],
            "condition_scope_column": "",
        }
    }

    with pytest.raises(ValueError, match="condition_scope_column"):
        load_contrast_config(config)


def test_constraint_mask_requires_z_score_output() -> None:
    with pytest.raises(ValueError, match="requires fmri_contrast.output_type='z-score'"):
        _assert_constraint_mask_requires_z_output(
            output_type="cope",
            constraint_spec={"threshold_mode": "z"},
        )


def test_apply_trial_phase_scoping_requires_phase_column_when_requested() -> None:
    events_df = pd.DataFrame(
        {
            "onset": [0.0],
            "duration": [1.0],
            "trial_type": ["pain"],
        }
    )

    with pytest.raises(ValueError, match="stim_phases_to_model is set"):
        _apply_trial_phase_scoping(
            events_df,
            allowed_phases=["plateau"],
            phase_column="stim_phase",
        )


def test_apply_trial_phase_scoping_requires_scope_column_when_requested() -> None:
    events_df = pd.DataFrame(
        {
            "onset": [0.0],
            "duration": [1.0],
            "trial_type": ["stimulation"],
            "stim_phase": ["plateau"],
        }
    )

    with pytest.raises(ValueError, match="phase_scope_value is set"):
        _apply_trial_phase_scoping(
            events_df,
            allowed_phases=["plateau"],
            phase_column="stim_phase",
            phase_scope_column="phase_scope",
            phase_scope_value="stimulation",
        )


def test_trial_signature_config_requires_scope_columns_when_scoping_requested() -> None:
    cfg = TrialSignatureExtractionConfig(
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        runs=None,
        task="pain",
        name="trial-signatures",
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        smoothing_fwhm=None,
        confounds_strategy="auto",
        method="beta-series",
        condition_scope_trial_type_column="",
        condition_scope_phase_column="",
        condition_scope_trial_types=("stimulation",),
        condition_scope_stim_phases=("plateau",),
    )

    with pytest.raises(ValueError, match="condition_scope_trial_types"):
        cfg.normalized()


def test_trial_signature_config_defaults_do_not_inject_scope_columns() -> None:
    cfg = TrialSignatureExtractionConfig(
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        runs=None,
        task="pain",
        name="trial-signatures",
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        smoothing_fwhm=None,
        confounds_strategy="auto",
        method="beta-series",
        condition_scope_trial_types=("stimulation",),
        condition_scope_stim_phases=("plateau",),
    )

    with pytest.raises(ValueError, match="condition_scope_trial_types"):
        cfg.normalized()


def test_discover_available_conditions_requires_condition_column() -> None:
    from fmri_pipeline.analysis.contrast_builder import discover_available_conditions

    with pytest.raises(ValueError, match="condition_column is required"):
        discover_available_conditions(
            Path("/tmp"),
            subject="0001",
            task="pain",
            condition_column="",
        )


def test_first_level_condition_overlap_raises() -> None:
    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="pain",
        formula=None,
        name="pain",
        runs=None,
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=True,
    )
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 10.0],
            "duration": [1.0, 1.0],
            "trial_type": ["pain", "rest"],
        }
    )

    with pytest.raises(ValueError, match="must be mutually exclusive"):
        _remap_events_by_condition_columns(events_df, cfg, strict=True)


def test_prepare_events_for_glm_keeps_out_of_scope_rows_as_nuisance() -> None:
    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="binary_outcome",
        condition_a_value=1,
        condition_b_column="binary_outcome",
        condition_b_value=0,
        formula=None,
        name="pain",
        runs=None,
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=False,
        events_to_model=["stimulation"],
        events_to_model_column="trial_type",
        stim_phases_to_model=["plateau"],
        phase_column="stim_phase",
        phase_scope_column="trial_type",
        phase_scope_value="stimulation",
    )
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 10.0, 20.0, 30.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["stimulation", "stimulation", "stimulation", "rating"],
            "stim_phase": ["plateau", "plateau", "ramp", ""],
            "binary_outcome": [1, 0, 1, 0],
        }
    )

    prepared_events, eligible_mask = _prepare_events_for_glm(events_df, cfg)
    remap_result = _remap_events_by_condition_columns(
        prepared_events,
        cfg,
        strict=True,
        eligible_mask=eligible_mask,
    )

    assert eligible_mask.tolist() == [True, True, False, False]
    assert remap_result.events_df.loc[0, "trial_type"] == "cond_a_pain"
    assert remap_result.events_df.loc[1, "trial_type"] == "cond_b_pain"
    assert remap_result.events_df.loc[2, "trial_type"].startswith("nuis_stimulation")
    assert remap_result.events_df.loc[3, "trial_type"].startswith("nuis_rating")


def test_condition_scope_respects_existing_eligibility_mask() -> None:
    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="binary_outcome",
        condition_a_value=1,
        condition_b_column="binary_outcome",
        condition_b_value=0,
        formula=None,
        name="pain",
        runs=None,
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=False,
        condition_scope_trial_types=["stimulation"],
        condition_scope_column="trial_type",
        events_to_model=["rating"],
        events_to_model_column="trial_type",
    )
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 10.0],
            "duration": [1.0, 1.0],
            "trial_type": ["stimulation", "rating"],
            "binary_outcome": [1, 0],
        }
    )

    prepared_events, eligible_mask = _prepare_events_for_glm(events_df, cfg)
    remap_result = _remap_events_by_condition_columns(
        prepared_events,
        cfg,
        strict=False,
        eligible_mask=eligible_mask,
    )

    assert eligible_mask.tolist() == [False, True]
    assert remap_result.cond_a_found == False
    assert remap_result.events_df.loc[0, "trial_type"].startswith("nuis_stimulation")


def test_prepare_events_for_glm_uses_configured_events_to_model_column() -> None:
    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="binary_outcome",
        condition_a_value=1,
        condition_b_column="binary_outcome",
        condition_b_value=0,
        formula=None,
        name="pain",
        runs=None,
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=False,
        events_to_model=["stimulus", "rating"],
        events_to_model_column="event_class",
    )
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 10.0, 20.0],
            "duration": [1.0, 1.0, 1.0],
            "trial_type": ["a", "b", "c"],
            "event_class": ["stimulus", "rating", "iti"],
            "binary_outcome": [1, 0, 1],
        }
    )

    prepared_events, eligible_mask = _prepare_events_for_glm(events_df, cfg)

    assert prepared_events.loc[0, "trial_type"] == "a"
    assert prepared_events.loc[1, "trial_type"] == "b"
    assert prepared_events.loc[2, "trial_type"].startswith("nuis_c")
    assert eligible_mask.tolist() == [True, True, False]


def test_prepare_events_for_glm_requires_configured_events_to_model_column() -> None:
    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        formula=None,
        name="pain",
        runs=None,
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=False,
        events_to_model=["stimulus"],
        events_to_model_column="event_class",
    )
    events_df = pd.DataFrame(
        {
            "onset": [0.0],
            "duration": [1.0],
            "trial_type": ["pain"],
        }
    )

    with pytest.raises(ValueError, match="events_to_model is set but events file has no 'event_class' column"):
        _prepare_events_for_glm(events_df, cfg)


def test_validate_events_against_bold_run_rejects_negative_onsets() -> None:
    events_df = pd.DataFrame(
        {
            "onset": [-0.5, 1.0],
            "duration": [1.0, 1.0],
            "trial_type": ["pain", "rest"],
        }
    )

    with patch(
        "fmri_pipeline.analysis.contrast_builder._get_bold_run_duration_seconds",
        return_value=20.0,
    ), patch(
        "fmri_pipeline.analysis.contrast_builder._get_tr_from_bold",
        return_value=2.0,
    ):
        with pytest.raises(ValueError, match="onset must be >= 0"):
            _validate_events_against_bold_run(
                events_df,
                bold_path=Path("run-01_bold.nii.gz"),
                context="unit-test",
            )


def test_validate_events_against_bold_run_rejects_events_past_scan_end() -> None:
    events_df = pd.DataFrame(
        {
            "onset": [0.0, 19.5],
            "duration": [1.0, 2.0],
            "trial_type": ["pain", "rest"],
        }
    )

    with patch(
        "fmri_pipeline.analysis.contrast_builder._get_bold_run_duration_seconds",
        return_value=20.0,
    ), patch(
        "fmri_pipeline.analysis.contrast_builder._get_tr_from_bold",
        return_value=2.0,
    ):
        with pytest.raises(ValueError, match="onset \\+ duration exceeds run duration"):
            _validate_events_against_bold_run(
                events_df,
                bold_path=Path("run-01_bold.nii.gz"),
                context="unit-test",
            )


def test_load_matching_brain_mask_for_bold_surfaces_mask_load_errors(tmp_path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_space-T1w_desc-preproc_bold.nii.gz"
    mask_path = tmp_path / "sub-0001_task-task_run-01_space-T1w_desc-brain_mask.nii.gz"
    bold_path.write_bytes(b"")
    mask_path.write_bytes(b"")

    fake_nib = SimpleNamespace(load=lambda _path: (_ for _ in ()).throw(OSError("bad mask")))

    with patch.dict(sys.modules, {"nibabel": fake_nib}), patch(
        "fmri_pipeline.analysis.contrast_builder._discover_brain_mask_for_bold",
        return_value=mask_path,
    ):
        with pytest.raises(OSError, match="bad mask"):
            _load_matching_brain_mask_for_bold(bold_path)


def test_build_intersection_brain_mask_surfaces_intersection_errors(tmp_path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_space-T1w_desc-preproc_bold.nii.gz"
    mask_path = tmp_path / "sub-0001_task-task_run-01_space-T1w_desc-brain_mask.nii.gz"
    bold_path.write_bytes(b"")
    mask_path.write_bytes(b"")

    fake_nib = SimpleNamespace(load=lambda _path: object())
    fake_masking = SimpleNamespace(
        intersect_masks=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("intersect failed"))
    )

    with patch.dict(
        sys.modules,
        {
            "nibabel": fake_nib,
            "nilearn.masking": fake_masking,
        },
    ), patch(
        "fmri_pipeline.analysis.contrast_builder._discover_brain_mask_for_bold",
        return_value=mask_path,
    ):
        with pytest.raises(RuntimeError, match="intersect failed"):
            _build_intersection_brain_mask([bold_path])


def test_trial_signature_condition_overlap_raises(tmp_path) -> None:
    cfg = TrialSignatureExtractionConfig(
        input_source="fmriprep",
        fmriprep_space="MNI152NLin2009cAsym",
        require_fmriprep=True,
        runs=None,
        task="task",
        name="pain",
        condition_a_column="binary_outcome",
        condition_a_value="1",
        condition_b_column="binary_outcome",
        condition_b_value="1",
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        smoothing_fwhm=None,
        confounds_strategy="none",
        method="beta-series",
    )
    events_path = tmp_path / "events.tsv"
    events_df = pd.DataFrame(
        {
            "onset": [0.0],
            "duration": [1.0],
            "trial_type": ["pain"],
            "binary_outcome": [1],
        }
    )

    with pytest.raises(ValueError, match="must be mutually exclusive"):
        _extract_trials_for_run(
            events_df=events_df,
            cfg=cfg,
            events_path=events_path,
            run_num=1,
        )


def test_combine_effect_images_preserves_missing_support_as_nan(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")

    effect_a = nib.Nifti1Image(np.array([[[1.0]]], dtype=np.float32), np.eye(4))
    effect_b = nib.Nifti1Image(np.array([[[3.0]]], dtype=np.float32), np.eye(4))
    variance_a = nib.Nifti1Image(np.array([[[1.0]]], dtype=np.float32), np.eye(4))
    variance_b = nib.Nifti1Image(np.array([[[np.inf]]], dtype=np.float32), np.eye(4))

    combined = _combine_effect_images(
        effects=[effect_a, effect_b],
        variances=[variance_a, variance_b],
        method="variance",
    )

    combined_data = np.asarray(combined.dataobj, dtype=float)
    assert np.isclose(combined_data[0, 0, 0], 1.0)

    no_support = _combine_effect_images(
        effects=[effect_a],
        variances=[nib.Nifti1Image(np.array([[[np.inf]]], dtype=np.float32), np.eye(4))],
        method="variance",
    )
    no_support_data = np.asarray(no_support.dataobj, dtype=float)
    assert np.isnan(no_support_data[0, 0, 0])


def test_compute_contrast_map_accepts_zero_valued_condition_codes() -> None:
    flm = SimpleNamespace(compute_contrast=lambda *args, **kwargs: "contrast-map", design_matrices_=[object()])
    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=True,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="binary_outcome",
        condition_a_value=1,
        condition_b_column="binary_outcome",
        condition_b_value=0,
        formula=None,
        name="pain",
        runs=None,
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=True,
    )

    contrast_map, contrast_def, output_type = compute_contrast_map(
        flm,
        cfg,
        available_conditions=[0, 1],
    )

    assert contrast_map == "contrast-map"
    assert contrast_def == "1 - 0"
    assert output_type == "z_score"


def test_discover_bold_runs_rejects_mixed_raw_and_fmriprep_inputs(tmp_path) -> None:
    bids_root = tmp_path / "bids"
    func_dir = bids_root / "sub-0001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    deriv_root = tmp_path / "derivatives"
    deriv_func = deriv_root / "fmriprep" / "sub-0001" / "func"
    deriv_func.mkdir(parents=True, exist_ok=True)

    for run_num in (1, 2):
        (func_dir / f"sub-0001_task-task_run-{run_num:02d}_events.tsv").write_text(
            "onset\tduration\ttrial_type\n",
            encoding="utf-8",
        )
        (func_dir / f"sub-0001_task-task_run-{run_num:02d}_bold.nii.gz").write_bytes(b"")

    (deriv_func / "sub-0001_task-task_run-01_space-T1w_desc-preproc_bold.nii.gz").write_bytes(b"")

    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=False,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        formula=None,
        name="pain",
        runs=None,
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=True,
    )

    with pytest.raises(FileNotFoundError, match="inconsistent across runs"):
        discover_bold_runs(
            bids_fmri_root=bids_root,
            bids_derivatives=deriv_root,
            subject="0001",
            task="task",
            runs=None,
            cfg=cfg,
        )


def test_discover_bold_runs_rejects_missing_requested_runs(tmp_path) -> None:
    bids_root = tmp_path / "bids"
    func_dir = bids_root / "sub-0001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    (func_dir / "sub-0001_task-task_run-01_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0\t1\tpain\n",
        encoding="utf-8",
    )
    (func_dir / "sub-0001_task-task_run-01_bold.nii.gz").write_bytes(b"")

    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="bids_raw",
        fmriprep_space="T1w",
        require_fmriprep=False,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        formula=None,
        name="pain",
        runs=[1, 2],
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=False,
    )

    with pytest.raises(FileNotFoundError, match="Some requested runs could not be resolved"):
        discover_bold_runs(
            bids_fmri_root=bids_root,
            bids_derivatives=None,
            subject="0001",
            task="task",
            runs=[1, 2],
            cfg=cfg,
        )


def test_validate_consistent_trs_rejects_mixed_values(tmp_path) -> None:
    run_1 = tmp_path / "run-01_bold.nii.gz"
    run_2 = tmp_path / "run-02_bold.nii.gz"
    run_1.write_bytes(b"")
    run_2.write_bytes(b"")

    with patch(
        "fmri_pipeline.analysis.contrast_builder._get_tr_from_bold",
        side_effect=[1.0, 2.0],
    ):
        with pytest.raises(ValueError, match="share the same TR"):
            _validate_consistent_trs([run_1, run_2])


def test_resolve_fmri_stats_artifact_prefers_matching_constraint_mask(tmp_path) -> None:
    contrast_path = tmp_path / "sub-0001_pain_z-score_deadbeef.nii.gz"
    contrast_path.write_bytes(b"")
    constraint_path = tmp_path / "sub-0001_pain_constraint-mask_deadbeef_cafefeed.nii.gz"
    constraint_path.write_bytes(b"")

    config = {
        "feature_engineering": {
            "sourcelocalization": {
                "fmri": {
                    "enabled": True,
                    "threshold": 3.1,
                    "tail": "pos",
                    "thresholding": {"mode": "z", "fdr_q": 0.05},
                    "cluster_min_voxels": 50,
                    "contrast": {
                        "enabled": True,
                        "name": "pain",
                        "input_source": "fmriprep",
                        "fmriprep_space": "T1w",
                        "require_fmriprep": True,
                        "contrast_type": "t-test",
                        "condition_a": {"column": "trial_type", "value": "pain"},
                        "condition_b": {"column": "trial_type", "value": "rest"},
                        "hrf_model": "spm",
                        "drift_model": "cosine",
                        "high_pass_hz": 0.008,
                        "output_type": "z-score",
                        "resample_to_freesurfer": False,
                    },
                }
            }
        }
    }

    constraint_spec = _load_constraint_mask_spec(config)
    assert constraint_spec is not None
    constraint_hash = _get_constraint_mask_hash(
        constraint_spec,
        resample_to_freesurfer=False,
    )

    contrast_path.with_suffix("").with_suffix(".json").write_text(
        json.dumps(
            {
                "constraint_mask": {
                    "hash": constraint_hash,
                    "path": str(constraint_path),
                    "resample_to_freesurfer": False,
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_fmri_stats_artifact(contrast_path, config)

    assert resolved == constraint_path


def test_trial_signature_extraction_requires_confounds_for_included_runs(tmp_path) -> None:
    cfg = TrialSignatureExtractionConfig(
        input_source="fmriprep",
        fmriprep_space="MNI152NLin2009cAsym",
        require_fmriprep=True,
        runs=None,
        task="task",
        name="pain",
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        smoothing_fwhm=None,
        confounds_strategy="auto",
        method="beta-series",
    )
    bold_path = tmp_path / "sub-0001_task-task_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    events_path = tmp_path / "sub-0001_task-task_run-01_events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\n0\t1\tpain\n10\t1\trest\n",
        encoding="utf-8",
    )

    with patch(
        "fmri_pipeline.analysis.trial_signatures._discover_runs",
        return_value=[(1, bold_path, events_path, None)],
    ), patch(
        "fmri_pipeline.analysis.trial_signatures._get_tr_from_bold",
        return_value=2.0,
    ):
        with pytest.raises(ValueError, match="require confounds for every included run"):
            run_trial_signature_extraction_for_subject(
                bids_fmri_root=tmp_path,
                bids_derivatives=tmp_path,
                deriv_root=tmp_path,
                subject="0001",
                cfg=cfg,
                signature_root=None,
                signature_specs=None,
            )


def test_trial_signature_config_rejects_invalid_method() -> None:
    cfg = TrialSignatureExtractionConfig(
        input_source="fmriprep",
        fmriprep_space="MNI152NLin2009cAsym",
        require_fmriprep=True,
        runs=None,
        task="task",
        name="pain",
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        smoothing_fwhm=None,
        confounds_strategy="auto",
        method="not-a-real-mode",
    )

    with pytest.raises(ValueError, match="method must be 'beta-series' or 'lss'"):
        cfg.normalized()


def test_trial_signature_discover_runs_rejects_missing_requested_runs(tmp_path) -> None:
    bids_root = tmp_path / "bids"
    func_dir = bids_root / "sub-0001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    (func_dir / "sub-0001_task-task_run-01_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0\t1\tpain\n",
        encoding="utf-8",
    )
    (func_dir / "sub-0001_task-task_run-01_bold.nii.gz").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Some requested runs could not be resolved"):
        _discover_runs(
            bids_fmri_root=bids_root,
            bids_derivatives=None,
            subject="0001",
            task="task",
            runs=[1, 2],
            input_source="bids_raw",
            fmriprep_space=None,
            require_fmriprep=False,
        )


def test_build_lss_events_keeps_group_specific_other_regressors() -> None:
    cfg = TrialSignatureExtractionConfig(
        input_source="fmriprep",
        fmriprep_space="MNI152NLin2009cAsym",
        require_fmriprep=True,
        runs=None,
        task="task",
        name="pain",
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        smoothing_fwhm=None,
        confounds_strategy="none",
        method="lss",
        include_other_events=False,
        lss_other_regressors="per_condition",
        signature_group_column="temperature",
        signature_group_values=("44.3", "45.3"),
    )
    trials = [
        TrialInfo(
            run=1,
            run_label="run-01",
            trial_index=1,
            condition="44.3",
            regressor="trial_run_01_001_44_3",
            onset=0.0,
            duration=1.0,
            original_trial_type="stimulation",
            source_events_path=Path("events.tsv"),
            source_row=0,
            extra={},
        ),
        TrialInfo(
            run=1,
            run_label="run-01",
            trial_index=2,
            condition="44.3",
            regressor="trial_run_01_002_44_3",
            onset=10.0,
            duration=1.0,
            original_trial_type="stimulation",
            source_events_path=Path("events.tsv"),
            source_row=1,
            extra={},
        ),
        TrialInfo(
            run=1,
            run_label="run-01",
            trial_index=3,
            condition="45.3",
            regressor="trial_run_01_003_45_3",
            onset=20.0,
            duration=1.0,
            original_trial_type="stimulation",
            source_events_path=Path("events.tsv"),
            source_row=2,
            extra={},
        ),
    ]

    lss_events = _build_lss_events(
        trial=trials[0],
        all_trials=trials,
        original_events_df=pd.DataFrame(),
        cfg=cfg,
    )

    assert set(lss_events["trial_type"].tolist()) == {
        "target",
        "other_group_44_3",
        "other_group_45_3",
    }


def test_build_contrast_from_runs_detailed_raises_when_requested_design_qc_write_fails(tmp_path: Path) -> None:
    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=False,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        formula=None,
        name="pain-vs-rest",
        runs=[1],
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=False,
        write_design_matrix=True,
    )
    glm_result = SimpleNamespace(
        flm="flm",
        included_bold_paths=[tmp_path / "run-01_bold.nii.gz"],
        included_events_paths=[tmp_path / "run-01_events.tsv"],
        included_confounds_paths=[None],
        skipped_runs=[],
        total_cond_a_events=1,
        total_cond_b_events=1,
        confound_columns=[],
        all_conditions=["cond_a_pain", "cond_b_pain"],
        synthetic_labels=[],
    )

    with patch(
        "fmri_pipeline.analysis.contrast_builder.discover_bold_runs",
        return_value=[(tmp_path / "run-01_bold.nii.gz", tmp_path / "run-01_events.tsv", 1)],
    ), patch(
        "fmri_pipeline.analysis.contrast_builder.discover_confounds",
        return_value=None,
    ), patch(
        "fmri_pipeline.analysis.contrast_builder.fit_first_level_glm_multi_run",
        return_value=glm_result,
    ), patch(
        "fmri_pipeline.analysis.contrast_builder._write_design_matrices",
        side_effect=OSError("disk full"),
    ), patch(
        "fmri_pipeline.analysis.contrast_builder.compute_contrast_map",
        return_value=("contrast-map", "cond_a_pain - cond_b_pain", "z_score"),
    ):
        with pytest.raises(OSError, match="disk full"):
            build_contrast_from_runs_detailed(
                bids_fmri_root=tmp_path,
                bids_derivatives=tmp_path,
                subject="0001",
                task="pain",
                cfg=cfg,
                output_dir=tmp_path / "contrast",
            )


def test_run_fmri_plotting_and_report_raises_when_provenance_write_fails(tmp_path: Path) -> None:
    cfg = FmriPlottingConfig(
        enabled=True,
        html_report=False,
        formats=("png",),
        space="both",
        include_motion_qc=False,
        include_carpet_qc=False,
        include_tsnr_qc=False,
        include_design_qc=False,
        include_signatures=False,
    )

    with patch(
        "fmri_pipeline.analysis.reporting.generate_signature_tables",
        return_value=[],
    ), patch(
        "pathlib.Path.write_text",
        side_effect=OSError("provenance write failed"),
    ):
        with pytest.raises(OSError, match="provenance write failed"):
            run_fmri_plotting_and_report(
                contrast_dir=tmp_path,
                subject="0001",
                task="pain",
                contrast_name="pain-vs-rest",
                cfg=cfg,
            )
