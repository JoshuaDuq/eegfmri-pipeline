from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from fmri_pipeline.analysis.contrast_builder import (
    ContrastBuilderConfig,
    _get_constraint_mask_hash,
    _load_constraint_mask_spec,
    _resolve_fmri_stats_artifact,
    _validate_consistent_trs,
    compute_contrast_map,
    discover_bold_runs,
    load_contrast_config_section,
    _remap_events_by_condition_columns,
)
from fmri_pipeline.analysis.trial_signatures import (
    TrialSignatureExtractionConfig,
    _extract_trials_for_run,
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
        cluster_correction=True,
        cluster_p_threshold=0.001,
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
        cluster_correction=True,
        cluster_p_threshold=0.001,
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
        cluster_correction=True,
        cluster_p_threshold=0.001,
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
                        "cluster_correction": True,
                        "cluster_p_threshold": 0.001,
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
