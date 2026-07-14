from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.scripts.benchmark_residual_gradient import (
    OutputPolicy,
    build_parser,
    discover_pilot_files,
    load_benchmark_config,
    run_pilot_benchmark,
    write_benchmark_reports,
)


def test_cli_requires_source_output_and_config() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source-root",
            "/data/source_data",
            "--output-root",
            "/data/derivatives/qc/residual_gradient_obs_benchmark",
            "--config",
            "config.yaml",
            "--output-policy",
            "error",
        ]
    )

    assert args.output_policy == "error"


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


def test_reports_are_complete_and_provenance_is_machine_readable(tmp_path: Path) -> None:
    run_rows = [{"run": 1, "n_components": count} for count in range(5)]
    component_rows = [{"run": 1, "channel": "Fz", "fold": 0, "n_components": 1}]
    preservation_rows = [{"n_components": 1, "minimum_sinusoid_amplitude_ratio": 0.99}]
    provenance = {"subject": "0006", "inputs": ["run1.vhdr"], "config_sha256": "abc"}

    paths = write_benchmark_reports(
        output_root=tmp_path,
        run_rows=run_rows,
        component_rows=component_rows,
        preservation_rows=preservation_rows,
        provenance=provenance,
        decision={"status": "accepted", "selected_components": 1},
        policy=OutputPolicy.ERROR,
    )

    assert pd.read_csv(paths.run_audit, sep="\t").shape[0] == 5
    assert json.loads(paths.provenance.read_text())["subject"] == "0006"


def test_report_writer_refuses_existing_files(tmp_path: Path) -> None:
    (tmp_path / "residual_obs_run_audit.tsv").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_benchmark_reports(
            output_root=tmp_path,
            run_rows=[],
            component_rows=[],
            preservation_rows=[],
            provenance={},
            decision={},
            policy=OutputPolicy.ERROR,
        )


def test_report_writer_validates_json_before_writing_any_report(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        write_benchmark_reports(
            output_root=tmp_path,
            run_rows=[{"run": 1}],
            component_rows=[{"run": 1}],
            preservation_rows=[{"run": 1}],
            provenance={"excluded_samples": np.int64(1)},
            decision={"status": "accepted"},
            policy=OutputPolicy.ERROR,
        )

    assert list(tmp_path.iterdir()) == []


def test_pilot_run_count_fails_before_loading_data(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    eeg_dir = source_root / "sub-0006" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "run1_sub0006_scannerpulse_corrected.vhdr").touch()
    config = load_benchmark_config(
        Path("studies/pain_study/scripts/config/residual_gradient_benchmark.yaml")
    )

    with pytest.raises(ValueError, match="Expected 6 pilot runs.*found 1"):
        run_pilot_benchmark(
            source_root=source_root,
            output_root=tmp_path / "output",
            config=config,
            output_policy=OutputPolicy.ERROR,
        )
