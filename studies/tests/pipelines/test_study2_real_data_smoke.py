from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.tests.pipelines.test_study2_source_stage import _source_stage_frame


def test_real_data_smoke_requires_explicit_gate_override(tmp_path: Path) -> None:
    from studies.pain_study.study2.real_data_smoke import run_real_data_smoke

    derivatives_root = _write_smoke_derivatives(tmp_path)

    with pytest.raises(RuntimeError, match="unmet Study 1 criteria"):
        run_real_data_smoke(
            derivatives_root=derivatives_root,
            output_dir=tmp_path / "outputs",
            force_gate_override=False,
            n_permutations=4,
        )


def test_real_data_smoke_runs_with_recorded_gate_override(tmp_path: Path) -> None:
    from studies.pain_study.study2.real_data_smoke import run_real_data_smoke

    derivatives_root = _write_smoke_derivatives(tmp_path)

    result = run_real_data_smoke(
        derivatives_root=derivatives_root,
        output_dir=tmp_path / "outputs",
        force_gate_override=True,
        n_permutations=4,
        random_seed=11,
    )

    assert result.subject_ids == ("sub-0001", "sub-0002")
    assert result.vertex_labels == ("C3", "C4")
    assert result.summary_path.exists()
    assert result.family_summary_path.exists()

    summary = pd.read_json(result.summary_path, typ="series")
    assert summary["gate_override_applied"] is True
    assert "significant_positive_delta_r2" in summary["study1_unmet_criteria"]
    assert "practical_effect_delta_r2" in summary["study1_unmet_criteria"]
    assert "practical_effect_lower_ci" in summary["study1_unmet_criteria"]

    family_summary = pd.read_csv(result.family_summary_path, sep="\t")
    assert family_summary["band"].tolist() == ["alpha", "beta", "gamma"]
    assert family_summary["n_subjects"].tolist() == [2, 2, 2]


def test_real_data_smoke_requires_feature_file_for_each_subject(tmp_path: Path) -> None:
    from studies.pain_study.study2.real_data_smoke import run_real_data_smoke

    derivatives_root = _write_smoke_derivatives(tmp_path)
    missing_file = (
        derivatives_root
        / "sub-0002"
        / "eeg"
        / "features"
        / "power"
        / "features_power_plateau.csv"
    )
    missing_file.unlink()

    with pytest.raises(FileNotFoundError, match="sub-0002"):
        run_real_data_smoke(
            derivatives_root=derivatives_root,
            output_dir=tmp_path / "outputs",
            force_gate_override=True,
            n_permutations=4,
        )


def _write_smoke_derivatives(tmp_path: Path) -> Path:
    derivatives_root = tmp_path / "derivatives"
    targets_dir = derivatives_root / "group" / "multimodal" / "study1" / "targets"
    reports_dir = derivatives_root / "group" / "multimodal" / "study1" / "reports"
    targets_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    target_frames: list[pd.DataFrame] = []
    for subject_id in ("sub-0001", "sub-0002"):
        frame = _source_stage_frame().assign(subject_id=subject_id)
        target_frames.append(frame.drop(columns=["eta_combined_z", "eta_alpha_z", "eta_beta_z", "eta_gamma_z"]))
        _write_subject_features(
            derivatives_root,
            subject_id=subject_id,
            n_trials=len(frame),
            seed=1 if subject_id == "sub-0001" else 2,
        )

    targets = pd.concat(target_frames, ignore_index=True)
    targets["within_block_trial"] = targets["trial_index_within_block"]
    targets.to_csv(targets_dir / "primary_targets.tsv", sep="\t", index=False)

    report = pd.DataFrame(
        [
            {
                "analysis_partition": "primary",
                "target": "NPS",
                "feature_spec": "alpha_beta_gamma",
                "model": "elasticnet",
                "mean_delta_r2": 0.011,
                "ci_low_delta_r2": -0.112,
                "p_value_delta_r2_holm": 0.090909,
                "temporal_negative_controls_passed": True,
            }
        ]
    )
    report.to_csv(reports_dir / "study1_report.tsv", sep="\t", index=False)
    return derivatives_root


def _write_subject_features(
    derivatives_root: Path,
    *,
    subject_id: str,
    n_trials: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    feature_dir = derivatives_root / subject_id / "eeg" / "features" / "power"
    feature_dir.mkdir(parents=True)
    features: dict[str, np.ndarray] = {
        "trial_id": np.arange(1, n_trials + 1, dtype=int),
    }
    for band in ("alpha", "beta", "gamma"):
        for channel in ("Fp1", "C3", "C4"):
            features[f"power_plateau_{band}_ch_{channel}_logratio"] = rng.normal(
                size=n_trials,
            )
    pd.DataFrame(features).to_csv(
        feature_dir / "features_power_plateau.csv",
        index=False,
    )
