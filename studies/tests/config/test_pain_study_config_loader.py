from __future__ import annotations

from pathlib import Path

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.eeg_coupling.config.eeg_bold_coupling_loader import (
    load_eeg_bold_coupling_config,
)

from studies.tests.test_support import REPO_ROOT

LEVEL2_CONTINUOUS_COLUMNS = [
    "block",
    "onset",
    "within_block_trial",
    "hrf_weighted_framewise_displacement",
    "hrf_weighted_std_dvars",
    "hrf_weighted_fp1_fp2_high_frequency_power",
    "residual_ecg_coupling",
]
LEVEL2_CATEGORICAL_COLUMNS = ["stimulus_temp", "selected_surface"]


def test_eeg_coupling_config_resolves_roi_assets_from_study_package() -> None:
    config = load_eeg_bold_coupling_config(
        config_path=REPO_ROOT
        / "studies/pain_study/eeg_coupling/config/eeg_bold_coupling_production.yaml",
    )

    label_file = Path(config["eeg_bold_coupling"]["rois"]["items"][0]["label_files"][0])
    assert "studies/pain_study/eeg_coupling/config/roi_library" in str(label_file)
    assert label_file.exists()


def test_smoke_configs_do_not_override_derivatives_root_output_dir() -> None:
    smoke_paths = (
        REPO_ROOT / "studies/pain_study/eeg_coupling/config/eeg_bold_coupling_smoketest.yaml",
        REPO_ROOT
        / "studies/pain_study/eeg_coupling/config/eeg_bold_coupling_smoke_robustness.yaml",
    )
    for config_path in smoke_paths:
        config = load_eeg_bold_coupling_config(config_path=config_path)
        output_dir = config["eeg_bold_coupling"]["output_dir"]
        assert output_dir is None, (
            f"{config_path.name} must keep eeg_bold_coupling.output_dir null "
            "so runs write under configured paths.deriv_root."
        )


def test_study1_production_loso_requires_inferential_cohort_size() -> None:
    config = load_study1_config(
        config_path=REPO_ROOT / "studies/pain_study/study1/config/study1_config.yaml",
    )

    assert int(config["study1"]["cohort"]["min_subjects"]) >= 30


def test_study1_production_target_and_inference_settings_are_prespecified() -> None:
    config = load_study1_config(
        config_path=REPO_ROOT / "studies/pain_study/study1/config/study1_config.yaml",
    )

    targets = config["study1"]["targets"]
    assert targets["metric"] == "dot"
    assert "signature_provenance" in targets
    assert targets["confounds_strategy"] == "motion24"
    assert targets["smoothing_fwhm"] == 6.0
    assert int(config["study1"]["feature_benchmark"]["n_perm"]) > 0
    assert targets["nuisance_regression"]["continuous_columns"] == LEVEL2_CONTINUOUS_COLUMNS
    assert targets["nuisance_regression"]["categorical_columns"] == LEVEL2_CATEGORICAL_COLUMNS
