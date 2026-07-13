from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest


RAW_STEM = "ThermalPainEEGFMRI_run1_sub0001_2026-03-02_10h55.27.564"
PROCESSED_STEM = f"{RAW_STEM}_scannerpulse_corrected"


def test_discover_raw_brainvision_runs_reads_complete_5000_hz_archive(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        BrainVisionArchiveRunSource,
        discover_raw_brainvision_runs,
    )

    archive = _write_raw_archive(tmp_path)

    sources = discover_raw_brainvision_runs(tmp_path, excluded_subjects=())

    assert len(sources) == 1
    source = sources[0]
    assert isinstance(source, BrainVisionArchiveRunSource)
    assert source.subject_id == "sub-0001"
    assert source.run_id == "1"
    assert source.representation == "brainvision_zip"
    assert source.archive_path == archive
    assert source.header_member == f"raw/{RAW_STEM}.vhdr"
    assert source.source_path == f"{archive}::raw/{RAW_STEM}.vhdr"


def test_discover_raw_brainvision_runs_requires_complete_triplet(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.preprocessing_psd_sources import (
        discover_raw_brainvision_runs,
    )

    _write_raw_archive(tmp_path, include_data=False)

    with pytest.raises(ValueError, match="missing BrainVision member"):
        discover_raw_brainvision_runs(tmp_path, excluded_subjects=())


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

    path = (
        tmp_path
        / "sub-0001"
        / "eeg"
        / "sub-0001_task-thermalactive_run-2_proc-clean_raw.fif"
    )
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


def test_estimate_source_spectrum_extracts_one_archive_triplet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mne

    import studies.pain_study.study1.figures.preprocessing_psd_sources as module

    _write_raw_archive(tmp_path)
    source = module.discover_raw_brainvision_runs(tmp_path, excluded_subjects=())[0]
    raw = _ClosableRaw()
    observed_header = None

    def read_raw_brainvision(path, **kwargs):
        nonlocal observed_header
        observed_header = Path(path)
        assert observed_header.is_file()
        assert observed_header.with_suffix(".eeg").is_file()
        assert observed_header.with_suffix(".vmrk").is_file()
        return raw

    monkeypatch.setattr(mne.io, "read_raw_brainvision", read_raw_brainvision)
    monkeypatch.setattr(module, "set_channel_types", lambda loaded: None)
    monkeypatch.setattr(
        module,
        "estimate_raw_continuous_run_spectrum",
        lambda loaded, **kwargs: kwargs["source_file"],
    )

    result = module.estimate_source_spectrum(source, _specification(5000.0))

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
    participant = tmp_path / "sub_0001_2026_03_02"
    participant.mkdir()
    archive = participant / "raw.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr(f"raw/{RAW_STEM}.vhdr", _header(RAW_STEM, 200))
        handle.writestr(f"raw/{RAW_STEM}.vmrk", "Brain Vision Data Exchange Marker File")
        if include_data:
            handle.writestr(f"raw/{RAW_STEM}.eeg", b"\x00\x00")
    return archive


def _write_processed_triplet(
    tmp_path: Path,
    *,
    participant_directory: str = "sub_0001_2026_03_02",
    stem: str = PROCESSED_STEM,
    sampling_interval_us: int = 1000,
) -> Path:
    directory = tmp_path / participant_directory / "processed"
    directory.mkdir(parents=True)
    header = directory / f"{stem}.vhdr"
    header.write_text(_header(stem, sampling_interval_us), encoding="utf-8")
    header.with_suffix(".vmrk").write_text(
        "Brain Vision Data Exchange Marker File",
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
