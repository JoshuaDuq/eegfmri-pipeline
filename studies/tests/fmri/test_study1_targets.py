from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from studies.tests.test_support import DotConfig


NPS_MASK_HASH = "0" * 64
SIIPS1_MASK_HASH = "1" * 64


def _base_config(root: Path) -> DotConfig:
    (root / "maps").mkdir(parents=True, exist_ok=True)
    manifest_entries = {}
    try:
        import nibabel as nib

        for rel_path in (
            "NPS/weights_NSF_grouppred_cvpcr.nii.gz",
            "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
        ):
            image_path = root / "maps" / rel_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                nib.save(
                    nib.Nifti1Image(np.ones((2, 2, 2), dtype=float), np.eye(4)),
                    image_path,
                )
            data = np.asanyarray(nib.load(str(image_path)).dataobj, dtype=float)
            finite = np.isfinite(data)
            positive = finite & (data > 0.0)
            negative = finite & (data < 0.0)
            manifest_entries["NPS" if rel_path.startswith("NPS/") else "SIIPS1"] = {
                "path": rel_path,
                "space": "MNI152NLin2009cAsym",
                "source_publication": "test fixture",
                "source_repository_or_access_record": "test fixture",
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "shape": [2, 2, 2],
                "affine": np.eye(4).tolist(),
                "support": {
                    "nonzero_voxels": int(np.count_nonzero(finite & (np.abs(data) > 0.0))),
                    "positive_voxels": int(np.count_nonzero(positive)),
                    "negative_voxels": int(np.count_nonzero(negative)),
                    "positive_abs_weight_mass": float(np.sum(np.abs(data[positive]))),
                    "negative_abs_weight_mass": float(np.sum(np.abs(data[negative]))),
                },
            }
        (root / "maps" / "signature_manifest.yaml").write_text(
            json.dumps({"signatures": manifest_entries}),
            encoding="utf-8",
        )
    except ImportError:
        pass
    return DotConfig(
        {
            "paths": {
                "bids_fmri_root": str(root / "bids_fmri"),
                "deriv_root": str(root / "derivatives"),
                "signature_dir": str(root / "maps"),
                "signature_maps": [
                    {"name": "NPS", "path": "NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
                    {
                        "name": "SIIPS1",
                        "path": "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
                    },
                ],
            },
            "study1": {
                "targets": {
                    "method": "lss",
                    "metric": "dot",
                    "normalization": "none",
                    "round_decimals": 3,
                    "contrast_name": "pain_vs_nonpain",
                    "signature_manifest_path": "signature_manifest.yaml",
                    "fmriprep_space": "MNI152NLin2009cAsym",
                    "input_source": "fmriprep",
                    "require_fmriprep": True,
                    "condition_a_column": "pain_binary_coded",
                    "condition_a_value": "1",
                    "condition_b_column": "pain_binary_coded",
                    "condition_b_value": "0",
                    "condition_scope_trial_type_column": "trial_type",
                    "condition_scope_trial_types": ["stimulation"],
                    "condition_scope_phase_column": "stim_phase",
                    "condition_scope_stim_phases": ["plateau"],
                    "hrf_model": "spm",
                    "drift_model": "cosine",
                    "high_pass_hz": 0.008,
                    "low_pass_hz": None,
                    "smoothing_fwhm": 0.0,
                    "confounds_strategy": "auto",
                    "lss_other_regressors": "all",
                    "names": ["NPS", "SIIPS1"],
                    "signature_provenance": {
                        "NPS": {
                            "path": "NPS/weights_NSF_grouppred_cvpcr.nii.gz",
                            "space": "MNI152NLin2009cAsym",
                        },
                        "SIIPS1": {
                            "path": "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
                            "space": "MNI152NLin2009cAsym",
                        },
                    },
                }
            },
        }
    )


def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block": [1, 1],
            "run_id": [1, 1],
            "trial_number": [1, 2],
            "pain_binary_coded": [1, 0],
            "onset": [22.150, 65.084],
            "duration": [0.001, 0.001],
        }
    )


def _write_signature_outputs(
    root: Path,
    *,
    contrast_name: str = "pain_vs_nonpain",
    include_siips1: bool = True,
    siips1_second_dot=2.2,
) -> None:
    sig_dir = (
        root
        / "derivatives"
        / "sub-0001"
        / "fmri"
        / "lss"
        / "task-pain"
        / f"contrast-{contrast_name}"
        / "signatures"
    )
    sig_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run": "run-01",
            "run_num": 1,
            "trial_index": 1,
            "signature": "NPS",
            "dot": 1.1,
            "n_voxels": 1000,
            "scoring_mask_sha256": NPS_MASK_HASH,
            "onset": 21.532,
            "duration": 7.5,
        },
        {
            "run": "run-01",
            "run_num": 1,
            "trial_index": 2,
            "signature": "NPS",
            "dot": 1.2,
            "n_voxels": 1000,
            "scoring_mask_sha256": NPS_MASK_HASH,
            "onset": 64.465,
            "duration": 7.5,
        },
    ]
    if include_siips1:
        rows.extend(
            [
                {
                    "run": "run-01",
                    "run_num": 1,
                    "trial_index": 1,
                    "signature": "SIIPS1",
                    "dot": 2.1,
                    "n_voxels": 800,
                    "scoring_mask_sha256": SIIPS1_MASK_HASH,
                    "onset": 21.532,
                    "duration": 7.5,
                },
                {
                    "run": "run-01",
                    "run_num": 1,
                    "trial_index": 2,
                    "signature": "SIIPS1",
                    "dot": siips1_second_dot,
                    "n_voxels": 800,
                    "scoring_mask_sha256": SIIPS1_MASK_HASH,
                    "onset": 64.465,
                    "duration": 7.5,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(sig_dir / "trial_signature_expression.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "run": ["run-01", "run-01"],
            "trial_index": [1, 2],
            "events_trial_number": [1, 2],
        }
    ).to_csv(sig_dir.parent / "trials.tsv", sep="\t", index=False)


def test_prepare_primary_targets_requires_mni_space() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        cfg["study1"]["targets"]["fmriprep_space"] = "T1w"

        with pytest.raises(ValueError, match="MNI"):
            prepare_primary_targets(
                subjects=["0001"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )


def test_prepare_primary_targets_requires_both_primary_signatures() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        _write_signature_outputs(root, include_siips1=False)

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=_events_frame(),
            ),
        ):
            with pytest.raises(ValueError, match="SIIPS1"):
                prepare_primary_targets(
                    subjects=["0001"],
                    task="pain",
                    config=cfg,
                    logger=logging.getLogger(__name__),
                )


def test_prepare_primary_targets_requires_original_trial_indices() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        _write_signature_outputs(root)
        events = _events_frame().drop(columns=["trial_number"])

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=events,
            ),
        ):
            with pytest.raises(ValueError, match="trial_number.*trial_index"):
                prepare_primary_targets(
                    subjects=["0001"],
                    task="pain",
                    config=cfg,
                    logger=logging.getLogger(__name__),
                )


def test_validate_signature_space_checks_configured_provenance_and_maps(tmp_path) -> None:
    import nibabel as nib

    from studies.pain_study.study1.targets import _validate_signature_space

    maps = tmp_path / "maps"
    (maps / "NPS").mkdir(parents=True)
    (maps / "SIIPS1").mkdir(parents=True)
    for rel_path in (
        "NPS/weights_NSF_grouppred_cvpcr.nii.gz",
        "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
    ):
        image_path = maps / rel_path
        image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=float), np.eye(4))
        nib.save(image, image_path)

    cfg = _base_config(tmp_path)
    cfg["paths"]["signature_dir"] = str(maps)
    cfg["study1"]["targets"]["signature_provenance"] = {
        "NPS": {
            "path": "NPS/weights_NSF_grouppred_cvpcr.nii.gz",
            "space": "MNI152NLin2009cAsym",
        },
        "SIIPS1": {
            "path": "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
            "space": "MNI152NLin2009cAsym",
        },
    }

    _validate_signature_space(
        cfg,
        [
            {"name": "NPS", "path": "NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
            {"name": "SIIPS1", "path": "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"},
        ],
    )

    with pytest.raises(ValueError, match="provenance"):
        _validate_signature_space(
            cfg,
            [
                {"name": "NPS", "path": "NPS/thresholded.nii.gz"},
                {"name": "SIIPS1", "path": "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"},
            ],
        )


def test_validate_signature_space_requires_frozen_signature_manifest(tmp_path) -> None:
    from studies.pain_study.study1.targets import _validate_signature_space

    cfg = _base_config(tmp_path)
    cfg["study1"]["targets"].pop("signature_manifest_path")

    with pytest.raises(ValueError, match="signature_manifest_path"):
        _validate_signature_space(
            cfg,
            [
                {"name": "NPS", "path": "NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
                {
                    "name": "SIIPS1",
                    "path": "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
                },
            ],
        )


def test_prepare_primary_targets_rejects_non_finite_primary_values() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        _write_signature_outputs(root, include_siips1=True, siips1_second_dot=float("nan"))

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=_events_frame(),
            ),
        ):
            with pytest.raises(ValueError, match="finite values"):
                prepare_primary_targets(
                    subjects=["0001"],
                    task="pain",
                    config=cfg,
                    logger=logging.getLogger(__name__),
                )


def test_prepare_primary_targets_requires_explicit_task_block_column() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        _write_signature_outputs(root)
        events = _events_frame().drop(columns=["block"])

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=events,
            ),
        ):
            with pytest.raises(ValueError, match="task block"):
                prepare_primary_targets(
                    subjects=["0001"],
                    task="pain",
                    config=cfg,
                    logger=logging.getLogger(__name__),
                )


def test_prepare_primary_targets_writes_wide_primary_table() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        _write_signature_outputs(root)

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=_events_frame(),
            ),
        ):
            out_path = prepare_primary_targets(
                subjects=["0001"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )

        assert out_path.name == "primary_targets.parquet"
        frame = pd.read_parquet(out_path)
        assert list(frame["subject_id"]) == ["sub-0001", "sub-0001"]
        assert list(frame["task"]) == ["pain", "pain"]
        assert list(frame["NPS"]) == [1.1, 1.2]
        assert list(frame["SIIPS1"]) == [2.1, 2.2]
        assert list(frame["NPS_fmri_scoring_mask_sha256"]) == [NPS_MASK_HASH, NPS_MASK_HASH]
        assert list(frame["SIIPS1_fmri_scoring_mask_sha256"]) == [
            SIIPS1_MASK_HASH,
            SIIPS1_MASK_HASH,
        ]


def test_prepare_primary_targets_records_nuisance_columns_without_residual_targets() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        cfg["study1"]["targets"]["nuisance_regression"] = {
            "enabled": True,
            "continuous_columns": ["pain_binary_coded"],
            "categorical_columns": [],
        }

        events = pd.DataFrame(
            {
                "block": [1, 1, 1, 1],
                "run_id": [1, 1, 1, 1],
                "trial_number": [1, 2, 3, 4],
                "pain_binary_coded": [0, 0, 1, 1],
                "onset": [10.0, 20.0, 30.0, 40.0],
                "duration": [0.5, 0.5, 0.5, 0.5],
            }
        )

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=events,
            ),
            patch(
                "studies.pain_study.study1.targets.load_fmri_signature_target_for_subject",
                side_effect=[
                    (
                        pd.Series([1.0, 3.0, 11.0, 13.0]),
                        "NPS",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [1000, 1000, 1000, 1000],
                                "fmri_scoring_mask_sha256": [NPS_MASK_HASH] * 4,
                            }
                        ),
                    ),
                    (
                        pd.Series([2.0, 4.0, 12.0, 14.0]),
                        "SIIPS1",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [800, 800, 800, 800],
                                "fmri_scoring_mask_sha256": [SIIPS1_MASK_HASH] * 4,
                            }
                        ),
                    ),
                ],
            ),
        ):
            out_path = prepare_primary_targets(
                subjects=["0001"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )

        frame = pd.read_parquet(out_path)
        assert "pain_binary_coded" in frame.columns
        assert "NPS_nuisance_residual" not in frame.columns
        assert "SIIPS1_nuisance_residual" not in frame.columns
        assert list(frame["pain_binary_coded"]) == [0, 0, 1, 1]


def test_prepare_primary_targets_expands_categorical_temperature_nuisance() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        cfg["study1"]["targets"]["nuisance_regression"] = {
            "enabled": True,
            "continuous_columns": ["block", "onset"],
            "categorical_columns": ["stimulus_temp"],
        }

        events = pd.DataFrame(
            {
                "block": [1, 1, 1, 1],
                "run_id": [1, 1, 1, 1],
                "trial_number": [1, 2, 3, 4],
                "stimulus_temp": [44.0, 46.0, 44.0, 47.0],
                "onset": [10.0, 20.0, 30.0, 40.0],
                "duration": [0.5, 0.5, 0.5, 0.5],
            }
        )

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=events,
            ),
            patch(
                "studies.pain_study.study1.targets.load_fmri_signature_target_for_subject",
                side_effect=[
                    (
                        pd.Series([1.0, 3.0, 11.0, 13.0]),
                        "NPS",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [1000, 1000, 1000, 1000],
                                "fmri_scoring_mask_sha256": [NPS_MASK_HASH] * 4,
                            }
                        ),
                    ),
                    (
                        pd.Series([2.0, 4.0, 12.0, 14.0]),
                        "SIIPS1",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [800, 800, 800, 800],
                                "fmri_scoring_mask_sha256": [SIIPS1_MASK_HASH] * 4,
                            }
                        ),
                    ),
                ],
            ),
        ):
            out_path = prepare_primary_targets(
                subjects=["0001"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )

        frame = pd.read_parquet(out_path)
        assert list(frame["stimulus_temp_level_46_0"]) == [0.0, 1.0, 0.0, 0.0]
        assert list(frame["stimulus_temp_level_47_0"]) == [0.0, 0.0, 0.0, 1.0]
        assert "stimulus_temp_level_44_0" not in frame.columns
        assert "NPS_nuisance_residual" not in frame.columns


def test_prepare_primary_targets_rejects_variable_signature_voxel_counts() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=_events_frame(),
            ),
            patch(
                "studies.pain_study.study1.targets.load_fmri_signature_target_for_subject",
                side_effect=[
                    (
                        pd.Series([1.0, 3.0]),
                        "NPS",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [1000, 1001],
                                "fmri_scoring_mask_sha256": [NPS_MASK_HASH] * 2,
                            }
                        ),
                    ),
                    (
                        pd.Series([2.0, 4.0]),
                        "SIIPS1",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [800, 800],
                                "fmri_scoring_mask_sha256": [SIIPS1_MASK_HASH] * 2,
                            }
                        ),
                    ),
                ],
            ),
        ):
            with pytest.raises(ValueError, match="identical voxel count"):
                prepare_primary_targets(
                    subjects=["0001"],
                    task="pain",
                    config=cfg,
                    logger=logging.getLogger(__name__),
                )


def test_prepare_primary_targets_rejects_variable_signature_mask_extent() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)

        with (
            patch(
                "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
                return_value={"output_dir": "ignored"},
            ),
            patch(
                "studies.pain_study.study1.targets.load_events_df",
                return_value=_events_frame(),
            ),
            patch(
                "studies.pain_study.study1.targets.load_fmri_signature_target_for_subject",
                side_effect=[
                    (
                        pd.Series([1.0, 3.0]),
                        "NPS",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [1000, 1000],
                                "fmri_scoring_mask_sha256": [NPS_MASK_HASH, "2" * 64],
                            }
                        ),
                    ),
                    (
                        pd.Series([2.0, 4.0]),
                        "SIIPS1",
                        pd.DataFrame(
                            {
                                "fmri_n_voxels": [800, 800],
                                "fmri_scoring_mask_sha256": [SIIPS1_MASK_HASH] * 2,
                            }
                        ),
                    ),
                ],
            ),
        ):
            with pytest.raises(ValueError, match="identical scoring-mask extent"):
                prepare_primary_targets(
                    subjects=["0001"],
                    task="pain",
                    config=cfg,
                    logger=logging.getLogger(__name__),
                )


def test_build_trial_signature_config_propagates_scope_fields() -> None:
    from studies.pain_study.study1.targets import _build_trial_signature_config

    with tempfile.TemporaryDirectory() as td:
        cfg = _base_config(Path(td))

        trial_cfg = _build_trial_signature_config(cfg, task="thermalactive")

    assert trial_cfg.condition_a_column == "pain_binary_coded"
    assert trial_cfg.condition_a_value == "1"
    assert trial_cfg.condition_b_column == "pain_binary_coded"
    assert trial_cfg.condition_b_value == "0"
    assert trial_cfg.condition_scope_trial_type_column == "trial_type"
    assert trial_cfg.condition_scope_trial_types == ("stimulation",)
    assert trial_cfg.condition_scope_phase_column == "stim_phase"
    assert trial_cfg.condition_scope_stim_phases == ("plateau",)
