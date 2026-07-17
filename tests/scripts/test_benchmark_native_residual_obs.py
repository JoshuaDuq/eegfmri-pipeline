from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from studies.pain_study.scripts.benchmark_native_residual_obs import (
    BasisScope,
    build_parser,
    load_benchmark_config,
    select_benchmark_recordings,
    write_benchmark_reports,
)
from studies.pain_study.scripts.run_native_eeg_fmri_artifact_correction import (
    InputRecording,
)


def _recording(tmp_path: Path, subject: str, run: int) -> InputRecording:
    return InputRecording(
        subject=subject,
        run=run,
        vhdr_path=tmp_path / f"{subject}_run-{run}.vhdr",
        bold_json_path=tmp_path / f"{subject}_run-{run}_bold.json",
        bold_json_sha256="bold",
        source_vhdr_sha256="vhdr",
        source_vmrk_sha256="vmrk",
        source_eeg_size=1,
        source_eeg_mtime_ns=2,
    )


def test_default_config_prespecifies_first_run_without_a_cohort_count() -> None:
    config = load_benchmark_config(
        Path("studies/pain_study/scripts/config/native_residual_obs_benchmark.yaml")
    )

    assert config.run_indices == (1,)
    assert config.component_counts == (0, 1, 2, 3, 4)
    assert config.minimum_participants == 3
    assert config.residual_line_frequencies_hz == (
        37.17,
        38.39,
        51.57,
        52.80,
        57.19,
        61.10,
        83.98,
    )


def test_cli_accepts_an_explicit_whole_volume_basis_scope() -> None:
    args = build_parser().parse_args(["--basis-scope", "whole_volume"])

    assert args.basis_scope is BasisScope.WHOLE_VOLUME


def test_recording_selection_includes_new_participants_dynamically(tmp_path: Path) -> None:
    recordings = [
        _recording(tmp_path, subject, run)
        for subject in ("sub-0001", "sub-0002", "sub-0003", "sub-0004")
        for run in (1, 2)
    ]

    selected = select_benchmark_recordings(
        recordings,
        run_indices=(1,),
        minimum_participants=3,
    )

    assert [(recording.subject, recording.run) for recording in selected] == [
        ("sub-0001", 1),
        ("sub-0002", 1),
        ("sub-0003", 1),
        ("sub-0004", 1),
    ]


def test_recording_selection_fails_when_a_participant_lacks_a_selected_run(
    tmp_path: Path,
) -> None:
    recordings = [
        _recording(tmp_path, "sub-0001", 1),
        _recording(tmp_path, "sub-0002", 2),
        _recording(tmp_path, "sub-0003", 1),
    ]

    with pytest.raises(ValueError, match="missing configured benchmark runs.*sub-0002"):
        select_benchmark_recordings(
            recordings,
            run_indices=(1,),
            minimum_participants=3,
        )


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("version: 1\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid benchmark configuration keys"):
        load_benchmark_config(path)


def test_report_writer_publishes_all_machine_readable_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark"
    paths = write_benchmark_reports(
        output_root=output_root,
        line_rows=[{"recording_id": "sub-0001_run-1", "n_components": 0}],
        run_rows=[{"recording_id": "sub-0001_run-1", "n_components": 0}],
        preservation_rows=[{"recording_id": "sub-0001_run-1", "n_components": 1}],
        cohort_rows=[{"n_components": 1, "reference_frequency_hz": 61.10}],
        provenance={"recordings": ["sub-0001_run-1"]},
        decision={"status": "rejected", "selected_components": None},
    )

    assert pd.read_csv(paths.line_audit, sep="\t").shape[0] == 1
    assert json.loads(paths.decision.read_text(encoding="utf-8"))["status"] == "rejected"
    assert len(paths.all_paths()) == 6


def test_report_writer_refuses_an_existing_output_root(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="already exists"):
        write_benchmark_reports(
            output_root=tmp_path,
            line_rows=[],
            run_rows=[],
            preservation_rows=[],
            cohort_rows=[],
            provenance={},
            decision={},
        )
