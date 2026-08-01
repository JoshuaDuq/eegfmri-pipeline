from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

RAW_STEM = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
PROCESSED_STEM = f"{RAW_STEM}_scannerpulse_corrected"
CORRECTED_STEM = (
    "ThermalPainEEGFMRI_run3 (1)_sub0003_2026-03-23_11h30.23.962_scannerpulse_corrected"
)
RAW_SUB3_RUN1_STEM = "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h11.33.962"
RAW_SUB3_ABORTED_STEM = "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h10.39.899"
RAW_SUB3_RUN3_STEM = "ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h30.23.962"


def test_discover_raw_brainvision_runs_rejects_archive_representation(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_raw_brainvision_runs,
    )

    _write_raw_archive(tmp_path)

    with pytest.raises(FileNotFoundError, match="No eligible Study 1 raw EEG runs"):
        discover_raw_brainvision_runs(tmp_path, excluded_subjects=())


def test_discover_raw_brainvision_runs_requires_complete_triplet(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_raw_brainvision_runs,
    )

    _write_raw_archive(tmp_path, include_data=False)

    with pytest.raises(FileNotFoundError, match="No eligible Study 1 raw EEG runs"):
        discover_raw_brainvision_runs(tmp_path, excluded_subjects=())


def test_discover_raw_brainvision_runs_reads_directory_triplet(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        BrainVisionFileRunSource,
        discover_raw_brainvision_runs,
    )

    header = _write_raw_triplet(tmp_path, "sub_0001_2026_03_02", RAW_STEM)

    sources = discover_raw_brainvision_runs(tmp_path, excluded_subjects=())

    assert len(sources) == 1
    assert isinstance(sources[0], BrainVisionFileRunSource)
    assert sources[0].header_path == header
    assert sources[0].data_path == header.with_suffix(".eeg")
    assert sources[0].marker_path == header.with_suffix(".vmrk")


def test_discovery_reads_labeled_source_data_directories(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_processed_brainvision_runs,
        discover_raw_brainvision_runs,
    )

    raw_header = _write_raw_triplet(tmp_path, "sub-0001", RAW_STEM)
    processed_header = _write_processed_triplet(
        tmp_path,
        participant_directory="sub-0001",
    )

    raw_sources = discover_raw_brainvision_runs(tmp_path, excluded_subjects=())
    processed_sources = discover_processed_brainvision_runs(tmp_path, excluded_subjects=())

    assert [source.header_path for source in raw_sources] == [raw_header]
    assert [source.header_path for source in processed_sources] == [processed_header]


def test_processed_discovery_reads_the_gap_recovery_export_suffix(tmp_path: Path) -> None:
    """The export that recovers Analyzer's missed beats ends at a differently named node.

    Its files end `_scanner_artifact_step2` where the previous generation ended
    `_scannerpulse_corrected`. Discovery has to read the current source tree, which now
    carries only the newer name.
    """
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_processed_brainvision_runs,
    )

    header = _write_processed_triplet(
        tmp_path,
        participant_directory="sub-0001",
        stem=f"{RAW_STEM}_scanner_artifact_step2",
    )

    sources = discover_processed_brainvision_runs(tmp_path, excluded_subjects=())

    assert [source.header_path for source in sources] == [header]


def test_processed_discovery_rejects_both_export_generations_of_one_run(
    tmp_path: Path,
) -> None:
    """Two generations of the same run is an ambiguity, not a preference order."""
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_processed_brainvision_runs,
    )

    _write_processed_triplet(tmp_path, participant_directory="sub-0001")
    _write_processed_triplet(
        tmp_path,
        participant_directory="sub-0001",
        stem=f"{RAW_STEM}_scanner_artifact_step2",
    )

    with pytest.raises(ValueError, match="run"):
        discover_processed_brainvision_runs(tmp_path, excluded_subjects=())


def test_raw_archive_discovery_rejects_participant_directory_mismatch(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_raw_brainvision_runs,
    )

    participant = tmp_path / "sub-0002" / "eeg" / "original_untrimmed_5khz"
    participant.mkdir(parents=True)
    with ZipFile(participant / "raw.zip", "w") as archive:
        archive.writestr(f"raw/{RAW_STEM}.vhdr", _header(RAW_STEM, 200))
        archive.writestr(f"raw/{RAW_STEM}.vmrk", "marker")
        archive.writestr(f"raw/{RAW_STEM}.eeg", b"\x00\x00")

    with pytest.raises(FileNotFoundError, match="No eligible Study 1 raw EEG runs"):
        discover_raw_brainvision_runs(tmp_path, excluded_subjects=())


def test_raw_discovery_applies_subject_exclusions(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_raw_brainvision_runs,
    )

    expected = _write_raw_triplet(tmp_path, "sub_0001_2026_03_02", RAW_STEM)
    second_stem = RAW_STEM.replace("sub0001", "sub0002")
    _write_raw_triplet(tmp_path, "sub_0002_2026_03_05_EXCL", second_stem)

    sources = discover_raw_brainvision_runs(tmp_path, excluded_subjects=("sub-0002",))

    assert [source.header_path for source in sources] == [expected]


def test_raw_discovery_applies_exact_correction_and_exclusion(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        BrainVisionSourceCorrection,
        BrainVisionSourceExclusion,
        discover_raw_brainvision_runs,
    )

    _write_raw_triplet(tmp_path, "sub_0003_03_23_2026", RAW_SUB3_ABORTED_STEM)
    _write_raw_triplet(tmp_path, "sub_0003_03_23_2026", RAW_SUB3_RUN1_STEM)
    run3_header = _write_raw_triplet(
        tmp_path,
        "sub_0003_03_23_2026",
        RAW_SUB3_RUN3_STEM,
    )
    correction = BrainVisionSourceCorrection(
        header_filename=run3_header.name,
        subject_id="sub-0003",
        run_id="3",
        data_filename=f"{RAW_SUB3_RUN3_STEM}.eeg",
        marker_filename=f"{RAW_SUB3_RUN3_STEM}.vmrk",
        expected_data_reference=f"{RAW_SUB3_RUN3_STEM}.eeg",
        expected_marker_reference=f"{RAW_SUB3_RUN3_STEM}.vmrk",
        reason="Known task-sequence naming issue.",
    )
    exclusion = BrainVisionSourceExclusion(
        header_filename=f"{RAW_SUB3_ABORTED_STEM}.vhdr",
        reason="Aborted 8.76-second run start.",
    )

    sources = discover_raw_brainvision_runs(
        tmp_path,
        excluded_subjects=(),
        source_corrections=(correction,),
        source_exclusions=(exclusion,),
    )

    assert [(source.run_id, source.source_correction) for source in sources] == [
        ("1", None),
        ("3", correction.reason),
    ]


def test_raw_archive_discovery_applies_exact_correction_and_exclusion(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        BrainVisionSourceCorrection,
        BrainVisionSourceExclusion,
        discover_raw_brainvision_runs,
    )

    participant = tmp_path / "sub-0003" / "eeg" / "original_untrimmed_5khz"
    participant.mkdir(parents=True)
    archive_path = participant / "raw.zip"
    stems = (RAW_SUB3_ABORTED_STEM, RAW_SUB3_RUN1_STEM, RAW_SUB3_RUN3_STEM)
    with ZipFile(archive_path, "w") as archive:
        for stem in stems:
            archive.writestr(f"raw/{stem}.vhdr", _header(stem, 200))
            archive.writestr(f"raw/{stem}.vmrk", "marker")
            archive.writestr(f"raw/{stem}.eeg", b"\x00\x00")
    correction = BrainVisionSourceCorrection(
        header_filename=f"{RAW_SUB3_RUN3_STEM}.vhdr",
        subject_id="sub-0003",
        run_id="3",
        data_filename=f"{RAW_SUB3_RUN3_STEM}.eeg",
        marker_filename=f"{RAW_SUB3_RUN3_STEM}.vmrk",
        expected_data_reference=f"{RAW_SUB3_RUN3_STEM}.eeg",
        expected_marker_reference=f"{RAW_SUB3_RUN3_STEM}.vmrk",
        reason="Known temperature-sequence naming issue.",
    )
    exclusion = BrainVisionSourceExclusion(
        header_filename=f"{RAW_SUB3_ABORTED_STEM}.vhdr",
        reason="Aborted 8.76-second run start.",
    )

    with pytest.raises(ValueError, match="source corrections were not found"):
        discover_raw_brainvision_runs(
            tmp_path,
            excluded_subjects=(),
            source_corrections=(correction,),
            source_exclusions=(exclusion,),
        )


def test_discover_processed_brainvision_runs_reads_1000_hz_triplet(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        BrainVisionFileRunSource,
        discover_processed_brainvision_runs,
    )

    header = _write_processed_triplet(tmp_path)

    sources = discover_processed_brainvision_runs(tmp_path, excluded_subjects=())

    assert len(sources) == 1
    source = sources[0]
    assert isinstance(source, BrainVisionFileRunSource)
    assert source.subject_id == "sub-0001"
    assert source.run_id == "1"
    assert source.representation == "brainvision_file"
    assert source.header_path == header
    assert source.source_path == str(header)


def test_processed_discovery_rejects_participant_directory_mismatch(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_processed_brainvision_runs,
    )

    _write_processed_triplet(
        tmp_path,
        participant_directory="sub_0002_2026_03_02",
    )

    with pytest.raises(ValueError, match="participant directory sub-0002"):
        discover_processed_brainvision_runs(tmp_path, excluded_subjects=())


def test_discover_processed_brainvision_runs_requires_1000_hz(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_processed_brainvision_runs,
    )

    _write_processed_triplet(tmp_path, sampling_interval_us=200)

    with pytest.raises(ValueError, match="expected 1000.0 Hz"):
        discover_processed_brainvision_runs(tmp_path, excluded_subjects=())


def test_discover_processed_brainvision_runs_rejects_duplicate_subject_run(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_processed_brainvision_runs,
    )

    _write_processed_triplet(tmp_path, participant_directory="sub_0001_2026_03_02")
    _write_processed_triplet(tmp_path, participant_directory="sub_0001_2026_03_03")

    with pytest.raises(ValueError, match="Duplicate EEG source for sub-0001 run 1"):
        discover_processed_brainvision_runs(tmp_path, excluded_subjects=())


def test_discover_processed_brainvision_runs_applies_exact_source_correction(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        BrainVisionSourceCorrection,
        discover_processed_brainvision_runs,
    )

    header = _write_processed_triplet(
        tmp_path,
        participant_directory="sub_0003_03_23_2026",
        stem=CORRECTED_STEM,
        internal_reference_stem=CORRECTED_STEM.replace("run3 (1)", "run1"),
    )
    correction = BrainVisionSourceCorrection(
        header_filename=header.name,
        subject_id="sub-0003",
        run_id="3",
        data_filename=f"{CORRECTED_STEM}.eeg",
        marker_filename=f"{CORRECTED_STEM}.vmrk",
        expected_data_reference=f"{CORRECTED_STEM.replace('run3 (1)', 'run1')}.eeg",
        expected_marker_reference=f"{CORRECTED_STEM.replace('run3 (1)', 'run1')}.vmrk",
        reason="Known task-sequence naming issue.",
    )

    sources = discover_processed_brainvision_runs(
        tmp_path,
        excluded_subjects=(),
        source_corrections=(correction,),
    )

    assert len(sources) == 1
    assert sources[0].subject_id == "sub-0003"
    assert sources[0].run_id == "3"
    assert sources[0].data_path == header.with_suffix(".eeg")
    assert sources[0].marker_path == header.with_suffix(".vmrk")
    assert sources[0].source_correction == correction.reason


def test_brainvision_discovery_applies_requested_subjects_and_exclusions(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_processed_brainvision_runs,
    )

    expected = _write_processed_triplet(tmp_path)
    second_stem = RAW_STEM.replace("sub0001", "sub0002")
    _write_processed_triplet(
        tmp_path,
        participant_directory="sub_0002_2026_03_03",
        stem=f"{second_stem}_scannerpulse_corrected",
    )

    selected = discover_processed_brainvision_runs(
        tmp_path,
        excluded_subjects=("sub-0002",),
        requested_subjects=("sub-0001",),
    )

    assert [source.header_path for source in selected] == [expected]


def test_discover_mne_runs_reuses_final_clean_contract(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        FifRunSource,
        discover_mne_runs,
    )

    path = tmp_path / "sub-0001" / "eeg" / "sub-0001_task-thermalactive_run-2_proc-clean_raw.fif"
    path.parent.mkdir(parents=True)
    path.touch()

    sources = discover_mne_runs(
        tmp_path,
        task="thermalactive",
        excluded_subjects=(),
    )

    assert sources == (
        FifRunSource(
            subject_id="sub-0001",
            run_id="2",
            source_path=str(path),
            path=path,
        ),
    )
    assert sources[0].representation == "fif"


def test_estimate_source_spectrum_loads_and_types_brainvision_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mne

    import studies.pain_study.study1.figures.preprocessing_psd_sources as module

    _write_processed_triplet(tmp_path)
    source = module.discover_processed_brainvision_runs(
        tmp_path,
        excluded_subjects=(),
    )[0]
    raw = _ClosableRaw()
    typed = []
    monkeypatch.setattr(mne.io, "read_raw_brainvision", lambda *args, **kwargs: raw)
    monkeypatch.setattr(module, "set_channel_types", lambda loaded: typed.append(loaded))
    monkeypatch.setattr(
        module,
        "estimate_raw_continuous_run_spectrum",
        lambda loaded, **kwargs: (loaded, kwargs),
    )

    loaded, arguments = module.estimate_source_spectrum(source, _specification(1000.0))

    assert loaded is raw
    assert typed == [raw]
    assert arguments["subject_id"] == "sub-0001"
    assert arguments["run_id"] == "1"
    assert arguments["source_file"] == source.source_path
    assert raw.closed


def test_estimate_source_spectrum_materializes_corrected_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mne

    import studies.pain_study.study1.figures.preprocessing_psd_sources as module

    header = _write_processed_triplet(
        tmp_path,
        participant_directory="sub_0003_03_23_2026",
        stem=CORRECTED_STEM,
        internal_reference_stem=CORRECTED_STEM.replace("run3 (1)", "run1"),
    )
    correction = module.BrainVisionSourceCorrection(
        header_filename=header.name,
        subject_id="sub-0003",
        run_id="3",
        data_filename=f"{CORRECTED_STEM}.eeg",
        marker_filename=f"{CORRECTED_STEM}.vmrk",
        expected_data_reference=f"{CORRECTED_STEM.replace('run3 (1)', 'run1')}.eeg",
        expected_marker_reference=f"{CORRECTED_STEM.replace('run3 (1)', 'run1')}.vmrk",
        reason="Known task-sequence naming issue.",
    )
    source = module.discover_processed_brainvision_runs(
        tmp_path,
        excluded_subjects=(),
        source_corrections=(correction,),
    )[0]
    raw = _ClosableRaw()
    observed_header = None

    def read_raw_brainvision(path, **kwargs):
        nonlocal observed_header
        observed_header = Path(path)
        text = observed_header.read_text(encoding="utf-8-sig")
        assert f"DataFile={CORRECTED_STEM}.eeg" in text
        assert f"MarkerFile={CORRECTED_STEM}.vmrk" in text
        assert (observed_header.parent / f"{CORRECTED_STEM}.eeg").is_file()
        assert (observed_header.parent / f"{CORRECTED_STEM}.vmrk").is_file()
        return raw

    monkeypatch.setattr(mne.io, "read_raw_brainvision", read_raw_brainvision)
    monkeypatch.setattr(module, "set_channel_types", lambda loaded: None)
    monkeypatch.setattr(
        module,
        "estimate_raw_continuous_run_spectrum",
        lambda loaded, **kwargs: kwargs["source_file"],
    )

    result = module.estimate_source_spectrum(source, _specification(1000.0))

    assert result == source.source_path
    assert observed_header is not None
    assert not observed_header.exists()
    assert raw.closed


def _specification(sampling_frequency_hz: float):
    from studies.pain_study.study1.figures.continuous_spectrum import (
        ContinuousSpectrumSpecification,
    )

    return ContinuousSpectrumSpecification(
        frequency_range_hz=(1.0, 90.0),
        n_fft=int(16.384 * sampling_frequency_hz),
        n_overlap=int(8.192 * sampling_frequency_hz),
        sampling_frequency_hz=sampling_frequency_hz,
    )


class _ClosableRaw:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _write_raw_archive(tmp_path: Path, *, include_data: bool = True) -> Path:
    participant = tmp_path / "sub-0001" / "eeg" / "original_untrimmed_5khz"
    participant.mkdir(parents=True)
    archive = participant / "raw.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr(f"raw/{RAW_STEM}.vhdr", _header(RAW_STEM, 200))
        handle.writestr(f"raw/{RAW_STEM}.vmrk", "Brain Vision Data Exchange Marker File")
        if include_data:
            handle.writestr(f"raw/{RAW_STEM}.eeg", b"\x00\x00")
    return archive


def _write_raw_triplet(tmp_path: Path, participant_directory: str, stem: str) -> Path:
    subject = participant_directory.replace("sub_", "sub-").split("_", maxsplit=1)[0]
    directory = tmp_path / subject / "eeg" / "original_untrimmed_5khz"
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / f"{stem}.vhdr"
    header.write_text(_header(stem, 200), encoding="utf-8")
    header.with_suffix(".vmrk").write_text(
        "Brain Vision Data Exchange Marker File\n" "[Common Infos]\n" f"DataFile={stem}.eeg\n",
        encoding="utf-8",
    )
    header.with_suffix(".eeg").write_bytes(b"\x00\x00")
    return header


def _write_processed_triplet(
    tmp_path: Path,
    *,
    participant_directory: str = "sub_0001_2026_03_02",
    stem: str = PROCESSED_STEM,
    internal_reference_stem: str | None = None,
    sampling_interval_us: int = 1000,
) -> Path:
    subject = participant_directory.replace("sub_", "sub-").split("_", maxsplit=1)[0]
    directory = tmp_path / subject / "eeg" / "brainvision_processed_1khz"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / f"{stem}.vhdr").exists():
        stem = stem.replace("_scannerpulse_corrected", "_duplicate_scannerpulse_corrected")
    header = directory / f"{stem}.vhdr"
    reference_stem = internal_reference_stem or stem
    header.write_text(_header(reference_stem, sampling_interval_us), encoding="utf-8")
    header.with_suffix(".vmrk").write_text(
        "Brain Vision Data Exchange Marker File\n"
        "[Common Infos]\n"
        f"DataFile={reference_stem}.eeg\n",
        encoding="utf-8",
    )
    header.with_suffix(".eeg").write_bytes(b"\x00\x00")
    return header


def _header(stem: str, sampling_interval_us: int) -> str:
    return (
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={stem}.eeg\n"
        f"MarkerFile={stem}.vmrk\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        "NumberOfChannels=64\n"
        f"SamplingInterval={sampling_interval_us}\n"
    )
