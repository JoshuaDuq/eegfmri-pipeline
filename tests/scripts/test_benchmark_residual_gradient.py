from __future__ import annotations

from pathlib import Path

import pytest

from studies.pain_study.scripts.benchmark_residual_gradient import (
    discover_pilot_files,
    load_benchmark_config,
)


def test_default_config_has_fixed_pilot_and_component_grid() -> None:
    path = Path("studies/pain_study/scripts/config/residual_gradient_benchmark.yaml")

    config = load_benchmark_config(path)

    assert config.pilot_subject == "0006"
    assert config.component_counts == (0, 1, 2, 3, 4)
    assert config.obs.volume_marker == "Volume/V  1"


def test_discovery_returns_only_corrected_pilot_headers(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-0006" / "eeg"
    eeg_dir.mkdir(parents=True)
    accepted = eeg_dir / "run1_sub0006_scannerpulse_corrected.vhdr"
    accepted.touch()
    (eeg_dir / "run1_sub0006.vhdr").touch()
    (eeg_dir / "._run1_sub0006_scannerpulse_corrected.vhdr").touch()

    assert discover_pilot_files(tmp_path, "0006") == [accepted]


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("pilot_subject: '0006'\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown benchmark configuration keys"):
        load_benchmark_config(path)
