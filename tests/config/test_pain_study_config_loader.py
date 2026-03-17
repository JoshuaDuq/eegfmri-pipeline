from __future__ import annotations

from pathlib import Path

from studies.pain_study.config.eeg_bold_coupling_loader import (
    load_eeg_bold_coupling_config,
)

from tests import REPO_ROOT


def test_study2_config_resolves_roi_assets_from_study_package() -> None:
    config = load_eeg_bold_coupling_config(
        config_path=REPO_ROOT / "studies/pain_study/config/eeg_bold_coupling_study2.yaml",
    )

    label_file = Path(config["eeg_bold_coupling"]["rois"]["items"][0]["label_files"][0])
    assert "studies/pain_study/config/roi_library" in str(label_file)
    assert label_file.exists()
