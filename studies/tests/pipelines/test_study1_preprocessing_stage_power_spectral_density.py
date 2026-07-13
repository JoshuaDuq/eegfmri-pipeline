from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from studies.pain_study.study1.config.loader import load_study1_config


@pytest.mark.parametrize(
    ("stage", "label", "sampling_frequency_hz", "n_fft", "n_overlap"),
    (
        ("raw", "Original BrainVision", 5000.0, 81_920, 40_960),
        ("processed", "BrainVision processed", 1000.0, 16_384, 8_192),
        ("mne", "Final MNE processed", 500.0, 8_192, 4_096),
    ),
)
def test_preprocessing_stage_psd_specification_uses_equal_time_windows(
    stage: str,
    label: str,
    sampling_frequency_hz: float,
    n_fft: int,
    n_overlap: int,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
        preprocessing_stage_psd_specification,
    )

    specification = preprocessing_stage_psd_specification(load_study1_config(), stage)

    assert specification.stage.identifier == stage
    assert specification.stage.label == label
    assert specification.spectrum.sampling_frequency_hz == sampling_frequency_hz
    assert specification.spectrum.n_fft == n_fft
    assert specification.spectrum.n_overlap == n_overlap
    assert specification.segment_duration_s == 16.384
    assert specification.overlap_fraction == 0.5
    expected_corrections = 1 if stage in {"raw", "processed"} else 0
    assert len(specification.source_corrections) == expected_corrections
    expected_exclusions = 1 if stage == "raw" else 0
    assert len(specification.source_exclusions) == expected_exclusions


def test_preprocessing_stage_psd_specification_rejects_unknown_stage() -> None:
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
        preprocessing_stage_psd_specification,
    )

    with pytest.raises(ValueError, match="Unknown preprocessing PSD stage"):
        preprocessing_stage_psd_specification(load_study1_config(), "corrected")


def test_label_preprocessing_stage_summary_adds_stage_and_source_provenance() -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import FifRunSource
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
        label_preprocessing_stage_summary,
        preprocessing_stage_psd_specification,
    )

    summary = _summary()
    original_columns = tuple(summary.run_audit.columns)
    source_path = Path(summary.run_audit.loc[0, "source_file"])
    source = FifRunSource(
        subject_id="sub-0001",
        run_id="1",
        source_path=str(source_path),
        path=source_path,
    )
    specification = preprocessing_stage_psd_specification(load_study1_config(), "mne")

    labeled = label_preprocessing_stage_summary(summary, (source,), specification)

    assert tuple(summary.run_audit.columns) == original_columns
    assert labeled.run_audit.columns[:3].tolist() == [
        "stage",
        "source_representation",
        "source_correction",
    ]
    assert labeled.participant_spectra.columns[0] == "stage"
    assert labeled.cohort_spectrum.columns[0] == "stage"
    assert labeled.run_audit.loc[0, "stage"] == "mne"
    assert labeled.run_audit.loc[0, "source_representation"] == "fif"
    assert pd.isna(labeled.run_audit.loc[0, "source_correction"])
    assert pd.api.types.is_integer_dtype(labeled.run_audit["run"])
    assert labeled.run_audit.loc[0, "run"] == 1
    assert labeled.run_audit.loc[0, "segment_duration_s"] == 16.384
    assert labeled.run_audit.loc[0, "overlap_fraction"] == 0.5


def test_build_preprocessing_stage_psd_figure_labels_checkpoint() -> None:
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
        preprocessing_stage_psd_specification,
    )
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density_plot import (
        build_preprocessing_stage_psd_figure,
    )

    config = load_study1_config()
    specification = preprocessing_stage_psd_specification(config, "raw")

    figure = build_preprocessing_stage_psd_figure(_summary(), specification, config)

    labels = [text.get_text() for text in figure.axes[0].texts]
    assert "Original BrainVision · 5000 Hz" in labels
    plt.close(figure)


def test_preprocessing_stage_writer_creates_exact_artifact_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density as module
    from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
        preprocessing_stage_psd_specification,
    )

    summary = _summary(stage="raw")
    monkeypatch.setattr(module, "_build_stage_summary", lambda **kwargs: summary)
    specification = preprocessing_stage_psd_specification(load_study1_config(), "raw")

    first = module.write_preprocessing_stage_psd(
        sources=(),
        specification=specification,
        config=load_study1_config(),
        output_dir=tmp_path / "first",
    )
    second = module.write_preprocessing_stage_psd(
        sources=(),
        specification=specification,
        config=load_study1_config(),
        output_dir=tmp_path / "second",
    )

    assert first.svg.name == "cohort_power_spectral_density_raw.svg"
    assert first.run_tsv.name == "cohort_power_spectral_density_raw_by_run.tsv"
    assert first.participant_tsv.name == ("cohort_power_spectral_density_raw_by_subject.tsv")
    assert first.summary_tsv.name == "cohort_power_spectral_density_raw_summary.tsv"
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(183.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(92.0, abs=0.01)
    _assert_table_parity(first.run_tsv, first.run_parquet)
    _assert_table_parity(first.participant_tsv, first.participant_parquet)
    _assert_table_parity(first.summary_tsv, first.summary_parquet)
    assert sorted(path.name for path in first.svg.parent.iterdir()) == [
        "cohort_power_spectral_density_raw.svg",
        "cohort_power_spectral_density_raw_by_run.parquet",
        "cohort_power_spectral_density_raw_by_run.tsv",
        "cohort_power_spectral_density_raw_by_subject.parquet",
        "cohort_power_spectral_density_raw_by_subject.tsv",
        "cohort_power_spectral_density_raw_summary.parquet",
        "cohort_power_spectral_density_raw_summary.tsv",
    ]


def test_multi_stage_writer_validates_every_source_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density as module

    discovered = []
    written = []

    def discover(**kwargs):
        stage = kwargs["stage_identifier"]
        discovered.append(stage)
        if stage == "mne":
            raise FileNotFoundError("missing MNE stage")
        return ()

    monkeypatch.setattr(module, "_discover_stage_sources", discover)
    monkeypatch.setattr(
        module,
        "write_preprocessing_stage_psd",
        lambda **kwargs: written.append(kwargs),
    )

    with pytest.raises(FileNotFoundError, match="missing MNE stage"):
        module.write_preprocessing_stage_psds(
            stage_identifiers=("raw", "processed", "mne"),
            kingston_root=tmp_path / "kingston",
            derivative_root=tmp_path / "derivatives",
            task="thermalactive",
            config=load_study1_config(),
            output_dir=tmp_path / "reports",
        )

    assert discovered == ["raw", "processed", "mne"]
    assert written == []


def test_multi_stage_writer_preserves_requested_stage_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density as module

    written = []
    monkeypatch.setattr(module, "_discover_stage_sources", lambda **kwargs: ())

    def write(**kwargs):
        stage = kwargs["specification"].stage.identifier
        written.append(stage)
        path = tmp_path / f"{stage}.svg"
        return module.PreprocessingStagePsdPaths(
            svg=path,
            run_tsv=path.with_suffix(".run.tsv"),
            run_parquet=path.with_suffix(".run.parquet"),
            participant_tsv=path.with_suffix(".participant.tsv"),
            participant_parquet=path.with_suffix(".participant.parquet"),
            summary_tsv=path.with_suffix(".summary.tsv"),
            summary_parquet=path.with_suffix(".summary.parquet"),
        )

    monkeypatch.setattr(module, "write_preprocessing_stage_psd", write)

    paths = module.write_preprocessing_stage_psds(
        stage_identifiers=("raw", "processed", "mne"),
        kingston_root=tmp_path / "kingston",
        derivative_root=tmp_path / "derivatives",
        task="thermalactive",
        config=load_study1_config(),
        output_dir=tmp_path / "reports",
    )

    assert written == ["raw", "processed", "mne"]
    assert [path.svg.name for path in paths] == ["raw.svg", "processed.svg", "mne.svg"]


def test_multi_stage_writer_rejects_duplicate_stages(tmp_path: Path) -> None:
    import studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density as module

    with pytest.raises(ValueError, match="Duplicate preprocessing PSD stage"):
        module.write_preprocessing_stage_psds(
            stage_identifiers=("raw", "raw"),
            kingston_root=tmp_path,
            derivative_root=tmp_path,
            task="thermalactive",
            config=load_study1_config(),
        )


def test_preprocessing_stage_psd_main_passes_explicit_roots_and_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density as module

    output_paths = tuple(
        module.PreprocessingStagePsdPaths(
            svg=tmp_path / f"{stage}.svg",
            run_tsv=tmp_path / f"{stage}.run.tsv",
            run_parquet=tmp_path / f"{stage}.run.parquet",
            participant_tsv=tmp_path / f"{stage}.participant.tsv",
            participant_parquet=tmp_path / f"{stage}.participant.parquet",
            summary_tsv=tmp_path / f"{stage}.summary.tsv",
            summary_parquet=tmp_path / f"{stage}.summary.parquet",
        )
        for stage in ("raw", "processed", "mne")
    )
    captured = {}
    monkeypatch.setattr(module, "load_config", lambda path: load_study1_config())
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *args, **kwargs: None)

    def write(**kwargs):
        captured.update(kwargs)
        return output_paths

    monkeypatch.setattr(module, "write_preprocessing_stage_psds", write)

    result = module.main(
        [
            "--config",
            "pipeline.yaml",
            "--task",
            "thermalactive",
            "--kingston-root",
            str(tmp_path / "kingston"),
            "--derivative-root",
            str(tmp_path / "derivatives"),
            "--stage",
            "raw",
            "--stage",
            "processed",
            "--stage",
            "mne",
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert result == output_paths
    assert captured["stage_identifiers"] == ("raw", "processed", "mne")
    assert captured["kingston_root"] == tmp_path / "kingston"
    assert captured["derivative_root"] == tmp_path / "derivatives"
    assert capsys.readouterr().out.splitlines() == [str(path.svg) for path in output_paths]


def test_preprocessing_stage_psd_cli_help_has_no_runtime_warning(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MNE_DONTWRITE_HOME"] = "true"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def _assert_table_parity(tsv_path: Path, parquet_path: Path) -> None:
    assert_frame_equal(
        pd.read_csv(tsv_path, sep="\t"),
        pd.read_parquet(parquet_path),
        check_dtype=False,
    )


def _summary(stage: str | None = None):
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        CohortPsdSummary,
    )

    frequencies = np.asarray([1.0, 10.0, 90.0])
    participant = pd.DataFrame(
        {
            "subject_id": ["sub-0001"] * 3,
            "frequency_hz": frequencies,
            "psd_db_uv2_hz": [10.0, 5.0, 0.0],
            "n_runs": [1] * 3,
        }
    )
    cohort = pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "median_psd_db_uv2_hz": [10.0, 5.0, 0.0],
            "ci_low_psd_db_uv2_hz": [9.0, 4.0, -1.0],
            "ci_high_psd_db_uv2_hz": [11.0, 6.0, 1.0],
            "n_subjects": [1] * 3,
        }
    )
    run = pd.DataFrame(
        {
            "subject_id": ["sub-0001"],
            "run": ["1"],
            "source_file": ["sub-0001_task-thermalactive_run-1_proc-clean_raw.fif"],
            "n_channels": [63],
            "sampling_frequency_hz": [500.0],
            "n_samples": [50_000],
            "recording_duration_s": [100.0],
            "bad_annotation_duration_s": [2.0],
            "analyzed_duration_s": [98.0],
            "frequency_min_hz": [1.0],
            "frequency_max_hz": [90.0],
            "n_fft": [8_192],
            "n_overlap": [4_096],
            "frequency_resolution_hz": [500.0 / 8_192],
        }
    )
    if stage is not None:
        run["run"] = run["run"].astype(int)
        participant.insert(0, "stage", stage)
        cohort.insert(0, "stage", stage)
        run.insert(0, "stage", stage)
        run.insert(1, "source_representation", "brainvision_zip")
        run["segment_duration_s"] = 16.384
        run["overlap_fraction"] = 0.5
    return CohortPsdSummary(
        participant_spectra=participant,
        cohort_spectrum=cohort,
        run_audit=run,
    )
