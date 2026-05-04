from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from studies.tests.test_support import DotConfig


def _base_config(root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {
                "bids_fmri_root": str(root / "bids_fmri"),
                "deriv_root": str(root / "derivatives"),
                "signature_maps": [
                    {"name": "NPS", "path": str(root / "maps" / "nps.nii.gz")},
                    {"name": "SIIPS1", "path": str(root / "maps" / "siips1.nii.gz")},
                ],
            },
            "study1": {
                "targets": {
                    "method": "lss",
                    "metric": "dot",
                    "normalization": "none",
                    "round_decimals": 3,
                    "contrast_name": "pain_vs_nonpain",
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
                }
            },
        }
    )


def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [1, 1],
            "trial_number": [1, 2],
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
            "onset": 21.532,
            "duration": 7.5,
        },
        {
            "run": "run-01",
            "run_num": 1,
            "trial_index": 2,
            "signature": "NPS",
            "dot": 1.2,
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
                    "onset": 21.532,
                    "duration": 7.5,
                },
                {
                    "run": "run-01",
                    "run_num": 1,
                    "trial_index": 2,
                    "signature": "SIIPS1",
                    "dot": siips1_second_dot,
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

        with patch(
            "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
            return_value={"output_dir": "ignored"},
        ), patch(
            "studies.pain_study.study1.targets.load_events_df",
            return_value=_events_frame(),
        ):
            with pytest.raises(ValueError, match="SIIPS1"):
                prepare_primary_targets(
                    subjects=["0001"],
                    task="pain",
                    config=cfg,
                    logger=logging.getLogger(__name__),
                )


def test_prepare_primary_targets_rejects_non_finite_primary_values() -> None:
    from studies.pain_study.study1.targets import prepare_primary_targets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _base_config(root)
        _write_signature_outputs(root, include_siips1=True, siips1_second_dot=float("nan"))

        with patch(
            "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
            return_value={"output_dir": "ignored"},
        ), patch(
            "studies.pain_study.study1.targets.load_events_df",
            return_value=_events_frame(),
        ):
            with pytest.raises(ValueError, match="finite values"):
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

        with patch(
            "studies.pain_study.study1.targets.run_trial_signature_extraction_for_subject",
            return_value={"output_dir": "ignored"},
        ), patch(
            "studies.pain_study.study1.targets.load_events_df",
            return_value=_events_frame(),
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
