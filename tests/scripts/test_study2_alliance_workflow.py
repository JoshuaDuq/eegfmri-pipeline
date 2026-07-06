from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from eeg_pipeline.utils.config.overrides import apply_set_overrides
from studies.pain_study.scanner_contamination import SCANNER_CLEAN_GAMMA_RANGES_HZ


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLIANCE_ROOT = REPO_ROOT / "studies" / "pain_study" / "study2" / "alliance"


def test_study2_runtime_overrides_emit_parseable_path_templates():
    script = f"""
        set -euo pipefail
        source "{ALLIANCE_ROOT / "lib" / "study2_alliance_common.sh"}"
        STUDY2_SUBJECTS_DIR=/scratch/test/study2_freesurfer_subjects
        STUDY1_OUTPUT_ROOT_NAME=study1
        STUDY2_OUTPUT_ROOT_NAME=study2
        study2_runtime_overrides
    """
    result = subprocess.run(
        ["bash", "-lc", script],
        check=True,
        text=True,
        capture_output=True,
    )

    lines = result.stdout.splitlines()
    overrides = [
        lines[index + 1]
        for index, value in enumerate(lines)
        if value == "--set"
    ]

    config: dict[str, object] = {}
    apply_set_overrides(config, overrides)

    anatomy = config["study2"]["source_modeling"]["anatomy"]  # type: ignore[index]
    assert anatomy["trans_path_template"] == (  # type: ignore[index]
        "{subjects_dir}/{subject}/bem/{subject}-trans.fif"
    )
    assert anatomy["bem_path_template"] == (  # type: ignore[index]
        "{subjects_dir}/{subject}/bem/{subject}-5120-5120-5120-bem-sol.fif"
    )


def test_study2_runtime_overrides_require_subjects_dir():
    script = f"""
        set -euo pipefail
        source "{ALLIANCE_ROOT / "lib" / "study2_alliance_common.sh"}"
        STUDY1_OUTPUT_ROOT_NAME=study1
        STUDY2_OUTPUT_ROOT_NAME=study2
        study2_runtime_overrides
    """
    result = subprocess.run(
        ["bash", "-lc", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert result.stderr == "Missing required environment variable: STUDY2_SUBJECTS_DIR\n"


def test_study2_alliance_defaults_match_bem_valid_oct6_source_space():
    alliance_env = REPO_ROOT / "local_workflows" / "alliance_canada" / "alliance_env.sh"
    env_example = ALLIANCE_ROOT / "study2_alliance.env.example"

    assert 'export STUDY2_ADJACENCY_SUBJECT="sub-0001"' in alliance_env.read_text()
    assert 'export STUDY2_EXPECTED_VERTICES="8196"' in alliance_env.read_text()
    assert "STUDY2_ADJACENCY_SUBJECT=sub-0001" in env_example.read_text()
    assert "STUDY2_EXPECTED_VERTICES=8196" in env_example.read_text()


def test_study2_source_power_preserves_bids_subject_label():
    source_power_script = ALLIANCE_ROOT / "sbatch" / "04_source_power.sbatch"

    assert 'subject_label="${subject}"' in source_power_script.read_text()


def test_study2_subject_cli_args_preserve_bids_subject_labels(tmp_path):
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("sub-0001\nsub-0003\n")
    script = f"""
        set -euo pipefail
        source "{ALLIANCE_ROOT / "lib" / "study2_alliance_common.sh"}"
        STUDY2_SUBJECTS_FILE="{subjects_file}"
        study2_subject_cli_args
    """
    result = subprocess.run(
        ["bash", "-lc", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.splitlines() == [
        "--subject",
        "sub-0001",
        "--subject",
        "sub-0003",
    ]


def test_study2_subject_cli_args_require_subjects_file():
    script = f"""
        set -euo pipefail
        source "{ALLIANCE_ROOT / "lib" / "study2_alliance_common.sh"}"
        study2_subject_cli_args
    """
    result = subprocess.run(
        ["bash", "-lc", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert result.stderr == "Missing required environment variable: STUDY2_SUBJECTS_FILE\n"


def test_study2_adjacency_writes_to_runtime_output_root():
    adjacency_script = ALLIANCE_ROOT / "sbatch" / "03_prepare_adjacency.sbatch"

    assert "--output" in adjacency_script.read_text()
    assert "${STUDY2_OUTPUT_ROOT_NAME}/inference/adjacency.npy" in adjacency_script.read_text()


def test_study2_smoketest_has_required_source_modeling_fields():
    config_path = REPO_ROOT / "studies" / "pain_study" / "study2" / "config" / "study2_smoketest.yaml"
    config = yaml.safe_load(config_path.read_text())
    source_modeling = config["study2"]["source_modeling"]
    scanner_clean_gamma_ranges = [
        list(frequency_range) for frequency_range in SCANNER_CLEAN_GAMMA_RANGES_HZ.values()
    ]

    assert source_modeling["source_space_spacing"] == "oct6"
    assert source_modeling["forward_mindist_mm"] == 5.0
    assert source_modeling["frequency_bands"] == {
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": scanner_clean_gamma_ranges,
    }
