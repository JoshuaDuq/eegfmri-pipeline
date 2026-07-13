"""Strict source discovery for Study 1 preprocessing-stage spectra."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from shutil import copy2, copyfileobj
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from eeg_pipeline.utils.data.preprocessing import set_channel_types
from studies.pain_study.study1.figures.continuous_spectrum import (
    ContinuousRunSpectrum,
    ContinuousSpectrumSpecification,
    discover_final_clean_runs,
    estimate_continuous_run_spectrum,
    estimate_raw_continuous_run_spectrum,
    parse_final_clean_filename,
)

RAW_RUN_PATTERN = re.compile(r"^ThermalPainEEGFMRI_run(?P<run>\d+)_sub(?P<subject>\d{4})_.+\.vhdr$")
PROCESSED_RUN_PATTERN = re.compile(
    r"^ThermalPainEEGFMRI_run(?P<run>\d+)_sub(?P<subject>\d{4})_"
    r".+_scannerpulse_corrected\.vhdr$"
)
HEADER_VALUE_PATTERN = re.compile(
    r"^(?P<key>[^\r\n=]+)=(?P<value>[^\r\n]+)$",
    re.MULTILINE,
)
PARTICIPANT_DIRECTORY_PATTERN = re.compile(
    r"^sub_(?P<subject>\d{4})_(?:\d{4}_\d{2}_\d{2}|\d{2}_\d{2}_\d{4})$"
)


@dataclass(frozen=True)
class BrainVisionArchiveRunSource:
    """One BrainVision triplet contained in a ZIP archive."""

    subject_id: str
    run_id: str
    source_path: str
    archive_path: Path
    header_member: str
    marker_member: str
    data_member: str
    data_reference: str
    marker_reference: str
    source_correction: str | None

    @property
    def representation(self) -> str:
        return "brainvision_zip"


@dataclass(frozen=True)
class BrainVisionSourceCorrection:
    """Exact, configured repair for one documented BrainVision source issue."""

    header_filename: str
    subject_id: str
    run_id: str
    data_filename: str
    marker_filename: str
    expected_data_reference: str
    expected_marker_reference: str
    reason: str


@dataclass(frozen=True)
class BrainVisionSourceExclusion:
    """Exact exclusion for one documented non-run BrainVision recording."""

    header_filename: str
    reason: str


@dataclass(frozen=True)
class BrainVisionFileRunSource:
    """One on-disk BrainVision triplet."""

    subject_id: str
    run_id: str
    source_path: str
    header_path: Path
    data_path: Path
    marker_path: Path
    data_reference: str
    marker_reference: str
    source_correction: str | None

    @property
    def representation(self) -> str:
        return "brainvision_file"


@dataclass(frozen=True)
class FifRunSource:
    """One final-clean FIF run."""

    subject_id: str
    run_id: str
    source_path: str
    path: Path

    @property
    def representation(self) -> str:
        return "fif"

    @property
    def source_correction(self) -> None:
        return None


EegRunSource = BrainVisionArchiveRunSource | BrainVisionFileRunSource | FifRunSource


def discover_raw_brainvision_runs(
    source_root: Path,
    *,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
    source_corrections: Sequence[BrainVisionSourceCorrection] = (),
    source_exclusions: Sequence[BrainVisionSourceExclusion] = (),
) -> tuple[BrainVisionArchiveRunSource | BrainVisionFileRunSource, ...]:
    """Discover original 5,000-Hz thermal runs in raw directories or ZIP archives."""
    corrections = _index_source_corrections(source_corrections)
    exclusions = _index_source_exclusions(source_exclusions)
    observed_corrections: set[str] = set()
    observed_exclusions: set[str] = set()
    selected: list[BrainVisionArchiveRunSource | BrainVisionFileRunSource] = []
    file_candidates: list[tuple[Path, str]] = []
    for participant_directory in _participant_directories(source_root):
        participant_subject_id = _participant_subject_id(participant_directory)
        archive_path = participant_directory / "raw.zip"
        raw_directory = participant_directory / "raw"
        if archive_path.is_file() and raw_directory.is_dir():
            raise ValueError(
                f"Participant has both raw.zip and raw directory: {participant_directory}"
            )
        if raw_directory.is_dir():
            file_candidates.extend(
                (path, participant_subject_id)
                for path in raw_directory.glob("ThermalPainEEGFMRI*.vhdr")
            )
            continue
        if not archive_path.is_file():
            continue
        selected.extend(
            _discover_archive_sources(
                archive_path,
                participant_subject_id=participant_subject_id,
                excluded_subjects=excluded_subjects,
                requested_subjects=requested_subjects,
                corrections=corrections,
                exclusions=exclusions,
                observed_corrections=observed_corrections,
                observed_exclusions=observed_exclusions,
            )
        )
    selected.extend(
        _discover_file_sources(
            file_candidates,
            filename_pattern=RAW_RUN_PATTERN,
            expected_sampling_frequency_hz=5000.0,
            excluded_subjects=excluded_subjects,
            requested_subjects=requested_subjects,
            corrections=corrections,
            exclusions=exclusions,
            observed_corrections=observed_corrections,
            observed_exclusions=observed_exclusions,
        )
    )
    _validate_manifest_entries(corrections, observed_corrections, "corrections")
    _validate_manifest_entries(exclusions, observed_exclusions, "exclusions")
    _validate_selected_sources(selected, stage="raw")
    return tuple(sorted(selected, key=_source_sort_key))


def discover_processed_brainvision_runs(
    source_root: Path,
    *,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
    source_corrections: Sequence[BrainVisionSourceCorrection] = (),
    source_exclusions: Sequence[BrainVisionSourceExclusion] = (),
) -> tuple[BrainVisionFileRunSource, ...]:
    """Discover BrainVision-processed 1,000-Hz thermal runs."""
    corrections = _index_source_corrections(source_corrections)
    exclusions = _index_source_exclusions(source_exclusions)
    observed_corrections: set[str] = set()
    observed_exclusions: set[str] = set()
    candidates = (
        (path, _participant_subject_id(participant_directory))
        for participant_directory in _participant_directories(source_root)
        for path in (participant_directory / "processed").rglob("ThermalPainEEGFMRI*.vhdr")
    )
    selected = _discover_file_sources(
        candidates,
        filename_pattern=PROCESSED_RUN_PATTERN,
        expected_sampling_frequency_hz=1000.0,
        excluded_subjects=excluded_subjects,
        requested_subjects=requested_subjects,
        corrections=corrections,
        exclusions=exclusions,
        observed_corrections=observed_corrections,
        observed_exclusions=observed_exclusions,
    )
    _validate_manifest_entries(corrections, observed_corrections, "corrections")
    _validate_manifest_entries(exclusions, observed_exclusions, "exclusions")
    _validate_selected_sources(selected, stage="processed")
    return tuple(sorted(selected, key=_source_sort_key))


def _discover_file_sources(
    candidates: Iterable[tuple[Path, str]],
    *,
    filename_pattern: re.Pattern[str],
    expected_sampling_frequency_hz: float,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str],
    corrections: dict[str, BrainVisionSourceCorrection],
    exclusions: dict[str, BrainVisionSourceExclusion],
    observed_corrections: set[str],
    observed_exclusions: set[str],
) -> list[BrainVisionFileRunSource]:
    selected: list[BrainVisionFileRunSource] = []
    candidates = sorted(
        (path, participant_subject_id)
        for path, participant_subject_id in candidates
        if not path.name.startswith("._")
    )
    for header_path, participant_subject_id in candidates:
        if header_path.name in exclusions:
            subject_id, _ = _parse_run_name(header_path.name, filename_pattern)
            _validate_participant_subject(
                subject_id,
                participant_subject_id,
                source=str(header_path),
            )
            observed_exclusions.add(header_path.name)
            continue
        correction = corrections.get(header_path.name)
        if correction is None:
            subject_id, run_id = _parse_run_name(header_path.name, filename_pattern)
        else:
            observed_corrections.add(header_path.name)
            subject_id, run_id = correction.subject_id, correction.run_id
        _validate_participant_subject(
            subject_id,
            participant_subject_id,
            source=str(header_path),
        )
        if not _subject_is_selected(
            subject_id,
            excluded_subjects=excluded_subjects,
            requested_subjects=requested_subjects,
        ):
            continue
        header_text = header_path.read_text(encoding="utf-8-sig")
        values = _header_values(header_text)
        data_path, marker_path = _resolve_file_triplet(
            header_path,
            values,
            correction,
        )
        _validate_sampling_frequency(
            header_text,
            expected_hz=expected_sampling_frequency_hz,
            source=str(header_path),
        )
        selected.append(
            BrainVisionFileRunSource(
                subject_id=subject_id,
                run_id=run_id,
                source_path=str(header_path),
                header_path=header_path,
                data_path=data_path,
                marker_path=marker_path,
                data_reference=values["DataFile"],
                marker_reference=values["MarkerFile"],
                source_correction=correction.reason if correction is not None else None,
            )
        )
    return selected


def _discover_archive_sources(
    archive_path: Path,
    *,
    participant_subject_id: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str],
    corrections: dict[str, BrainVisionSourceCorrection],
    exclusions: dict[str, BrainVisionSourceExclusion],
    observed_corrections: set[str],
    observed_exclusions: set[str],
) -> list[BrainVisionArchiveRunSource]:
    selected: list[BrainVisionArchiveRunSource] = []
    with ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        header_members = sorted(
            member
            for member in members
            if PurePosixPath(member).name.startswith("ThermalPainEEGFMRI")
            and PurePosixPath(member).suffix == ".vhdr"
        )
        for header_member in header_members:
            source = _archive_run_source(
                archive,
                archive_path=archive_path,
                header_member=header_member,
                archive_members=members,
                participant_subject_id=participant_subject_id,
                excluded_subjects=excluded_subjects,
                requested_subjects=requested_subjects,
                corrections=corrections,
                exclusions=exclusions,
                observed_corrections=observed_corrections,
                observed_exclusions=observed_exclusions,
            )
            if source is not None:
                selected.append(source)
    return selected


def _archive_run_source(
    archive: ZipFile,
    *,
    archive_path: Path,
    header_member: str,
    archive_members: set[str],
    participant_subject_id: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str],
    corrections: dict[str, BrainVisionSourceCorrection],
    exclusions: dict[str, BrainVisionSourceExclusion],
    observed_corrections: set[str],
    observed_exclusions: set[str],
) -> BrainVisionArchiveRunSource | None:
    header_filename = PurePosixPath(header_member).name
    if header_filename in exclusions:
        subject_id, _ = _parse_run_name(header_filename, RAW_RUN_PATTERN)
        _validate_participant_subject(
            subject_id,
            participant_subject_id,
            source=f"{archive_path}::{header_member}",
        )
        observed_exclusions.add(header_filename)
        return None
    correction = corrections.get(header_filename)
    if correction is None:
        subject_id, run_id = _parse_run_name(header_filename, RAW_RUN_PATTERN)
    else:
        observed_corrections.add(header_filename)
        subject_id, run_id = correction.subject_id, correction.run_id
    source_path = f"{archive_path}::{header_member}"
    _validate_participant_subject(
        subject_id,
        participant_subject_id,
        source=source_path,
    )
    if not _subject_is_selected(
        subject_id,
        excluded_subjects=excluded_subjects,
        requested_subjects=requested_subjects,
    ):
        return None
    header_text = archive.read(header_member).decode("utf-8-sig")
    values = _header_values(header_text)
    data_member, marker_member = _resolve_archive_triplet(
        header_member,
        values,
        archive_members,
        correction,
    )
    _validate_sampling_frequency(
        header_text,
        expected_hz=5000.0,
        source=source_path,
    )
    return BrainVisionArchiveRunSource(
        subject_id=subject_id,
        run_id=run_id,
        source_path=source_path,
        archive_path=archive_path,
        header_member=header_member,
        marker_member=marker_member,
        data_member=data_member,
        data_reference=values["DataFile"],
        marker_reference=values["MarkerFile"],
        source_correction=correction.reason if correction is not None else None,
    )


def discover_mne_runs(
    derivative_root: Path,
    *,
    task: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
) -> tuple[FifRunSource, ...]:
    """Discover final MNE-preprocessed FIF runs."""
    paths = discover_final_clean_runs(
        derivative_root,
        task=task,
        excluded_subjects=excluded_subjects,
        requested_subjects=requested_subjects,
    )
    return tuple(
        FifRunSource(
            subject_id=subject_id,
            run_id=run_id,
            source_path=str(path),
            path=path,
        )
        for path in paths
        for subject_id, run_id in (parse_final_clean_filename(path),)
    )


def estimate_source_spectrum(
    source: EegRunSource,
    specification: ContinuousSpectrumSpecification,
) -> ContinuousRunSpectrum:
    """Load one source, estimate its spectrum, and release its resources."""
    if isinstance(source, FifRunSource):
        return estimate_continuous_run_spectrum(source.path, specification)
    if isinstance(source, BrainVisionFileRunSource):
        if source.source_correction is None:
            return _estimate_brainvision_file(source, source.header_path, specification)
        with TemporaryDirectory(prefix="study1-psd-corrected-") as temporary_directory:
            header_path = _materialize_corrected_file_source(
                source,
                Path(temporary_directory),
            )
            return _estimate_brainvision_file(source, header_path, specification)
    with TemporaryDirectory(prefix="study1-psd-") as temporary_directory:
        header_path = _extract_archive_triplet(source, Path(temporary_directory))
        return _estimate_brainvision_file(source, header_path, specification)


def _estimate_brainvision_file(
    source: BrainVisionArchiveRunSource | BrainVisionFileRunSource,
    header_path: Path,
    specification: ContinuousSpectrumSpecification,
) -> ContinuousRunSpectrum:
    import mne

    raw = mne.io.read_raw_brainvision(header_path, preload=False, verbose="ERROR")
    try:
        set_channel_types(raw)
        return estimate_raw_continuous_run_spectrum(
            raw,
            subject_id=source.subject_id,
            run_id=source.run_id,
            source_file=source.source_path,
            specification=specification,
        )
    finally:
        raw.close()


def _extract_archive_triplet(
    source: BrainVisionArchiveRunSource,
    destination: Path,
) -> Path:
    members = (source.header_member, source.marker_member, source.data_member)
    with ZipFile(source.archive_path) as archive:
        for member in members:
            output_path = destination / PurePosixPath(member).name
            with archive.open(member) as input_handle, open(output_path, "wb") as output_handle:
                copyfileobj(input_handle, output_handle)
    header_path = destination / PurePosixPath(source.header_member).name
    if source.source_correction is None:
        return header_path
    data_path = destination / PurePosixPath(source.data_member).name
    marker_path = destination / PurePosixPath(source.marker_member).name
    _rewrite_brainvision_references(
        header_path=header_path,
        data_path=data_path,
        marker_path=marker_path,
        data_reference=source.data_reference,
        marker_reference=source.marker_reference,
    )
    return header_path


def _materialize_corrected_file_source(
    source: BrainVisionFileRunSource,
    destination: Path,
) -> Path:
    header_path = destination / source.header_path.name
    data_path = destination / source.data_path.name
    marker_path = destination / source.marker_path.name
    copy2(source.data_path, data_path)
    copy2(source.header_path, header_path)
    copy2(source.marker_path, marker_path)
    _rewrite_brainvision_references(
        header_path=header_path,
        data_path=data_path,
        marker_path=marker_path,
        data_reference=source.data_reference,
        marker_reference=source.marker_reference,
    )
    return header_path


def _rewrite_brainvision_references(
    *,
    header_path: Path,
    data_path: Path,
    marker_path: Path,
    data_reference: str,
    marker_reference: str,
) -> None:
    header_bytes = header_path.read_bytes()
    header_bytes = _replace_reference(
        header_bytes,
        key="DataFile",
        existing=data_reference,
        replacement=data_path.name,
    )
    header_bytes = _replace_reference(
        header_bytes,
        key="MarkerFile",
        existing=marker_reference,
        replacement=marker_path.name,
    )
    header_path.write_bytes(header_bytes)

    marker_bytes = marker_path.read_bytes()
    marker_bytes = _replace_reference(
        marker_bytes,
        key="DataFile",
        existing=data_reference,
        replacement=data_path.name,
    )
    marker_path.write_bytes(marker_bytes)


def _replace_reference(
    content: bytes,
    *,
    key: str,
    existing: str,
    replacement: str,
) -> bytes:
    current = f"{key}={existing}".encode()
    if content.count(current) != 1:
        raise ValueError(f"Expected exactly one {key}={existing!r} reference.")
    return content.replace(current, f"{key}={replacement}".encode())


def _parse_run_name(filename: str, pattern: re.Pattern[str]) -> tuple[str, str]:
    match = pattern.fullmatch(filename)
    if match is None:
        raise ValueError(f"Invalid Study 1 BrainVision run filename: {filename}")
    return f"sub-{match.group('subject')}", match.group("run")


def _header_values(header_text: str) -> dict[str, str]:
    return {
        match.group("key").strip(): match.group("value").strip()
        for match in HEADER_VALUE_PATTERN.finditer(header_text.replace("\r\n", "\n"))
    }


def _resolve_archive_triplet(
    header_member: str,
    header_values: dict[str, str],
    archive_members: set[str],
    correction: BrainVisionSourceCorrection | None,
) -> tuple[str, str]:
    if correction is None:
        filenames = (header_values["DataFile"], header_values["MarkerFile"])
    else:
        _validate_correction_references(header_member, header_values, correction)
        filenames = (correction.data_filename, correction.marker_filename)
    parent = PurePosixPath(header_member).parent
    data_member = str(parent / filenames[0])
    marker_member = str(parent / filenames[1])
    for member in (data_member, marker_member):
        if member not in archive_members:
            raise ValueError(f"{header_member} is missing BrainVision member {member}.")
    return data_member, marker_member


def _resolve_file_triplet(
    header_path: Path,
    header_values: dict[str, str],
    correction: BrainVisionSourceCorrection | None,
) -> tuple[Path, Path]:
    if correction is None:
        filenames = (header_values["DataFile"], header_values["MarkerFile"])
    else:
        _validate_correction_references(str(header_path), header_values, correction)
        filenames = (correction.data_filename, correction.marker_filename)
    paths = tuple(header_path.parent / filename for filename in filenames)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"{header_path} is missing BrainVision member {path}.")
    return paths[0], paths[1]


def _validate_correction_references(
    source: str,
    header_values: dict[str, str],
    correction: BrainVisionSourceCorrection,
) -> None:
    if header_values["DataFile"] != correction.expected_data_reference:
        raise ValueError(f"Configured DataFile correction does not match {source}.")
    if header_values["MarkerFile"] != correction.expected_marker_reference:
        raise ValueError(f"Configured MarkerFile correction does not match {source}.")


def _index_source_corrections(
    corrections: Sequence[BrainVisionSourceCorrection],
) -> dict[str, BrainVisionSourceCorrection]:
    indexed = {correction.header_filename: correction for correction in corrections}
    if len(indexed) != len(corrections):
        raise ValueError("BrainVision source correction filenames must be unique.")
    return indexed


def _index_source_exclusions(
    exclusions: Sequence[BrainVisionSourceExclusion],
) -> dict[str, BrainVisionSourceExclusion]:
    indexed = {exclusion.header_filename: exclusion for exclusion in exclusions}
    if len(indexed) != len(exclusions):
        raise ValueError("BrainVision source exclusion filenames must be unique.")
    return indexed


def _validate_manifest_entries(configured: dict, observed: set[str], name: str) -> None:
    missing = set(configured) - observed
    if missing:
        raise ValueError(f"Configured BrainVision source {name} were not found: {sorted(missing)}")


def _participant_directories(source_root: Path) -> tuple[Path, ...]:
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Kingston source root not found: {root}")
    return tuple(
        sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and PARTICIPANT_DIRECTORY_PATTERN.fullmatch(path.name)
        )
    )


def _participant_subject_id(participant_directory: Path) -> str:
    match = PARTICIPANT_DIRECTORY_PATTERN.fullmatch(participant_directory.name)
    if match is None:
        raise ValueError(f"Invalid canonical participant directory: {participant_directory}")
    return f"sub-{match.group('subject')}"


def _validate_participant_subject(
    source_subject_id: str,
    participant_subject_id: str,
    *,
    source: str,
) -> None:
    if source_subject_id != participant_subject_id:
        raise ValueError(
            f"EEG source {source} identifies {source_subject_id}, "
            f"but participant directory {participant_subject_id}."
        )


def _validate_sampling_frequency(
    header_text: str,
    *,
    expected_hz: float,
    source: str,
) -> None:
    sampling_interval_us = float(_header_values(header_text)["SamplingInterval"])
    if sampling_interval_us <= 0.0:
        raise ValueError(f"BrainVision SamplingInterval must be positive: {source}")
    sampling_frequency_hz = 1_000_000.0 / sampling_interval_us
    if sampling_frequency_hz != expected_hz:
        raise ValueError(
            f"Unexpected sampling frequency in {source}: {sampling_frequency_hz} Hz; "
            f"expected {expected_hz} Hz."
        )


def _subject_is_selected(
    subject_id: str,
    *,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str],
) -> bool:
    requested = set(requested_subjects)
    return subject_id not in set(excluded_subjects) and (not requested or subject_id in requested)


def _validate_selected_sources(sources: Sequence[EegRunSource], *, stage: str) -> None:
    if not sources:
        raise FileNotFoundError(f"No eligible Study 1 {stage} EEG runs found.")
    seen: set[tuple[str, str]] = set()
    for source in sources:
        key = (source.subject_id, source.run_id)
        if key in seen:
            raise ValueError(f"Duplicate EEG source for {source.subject_id} run {source.run_id}.")
        seen.add(key)


def _source_sort_key(source: EegRunSource) -> tuple[str, int]:
    return source.subject_id, int(source.run_id)


__all__ = [
    "BrainVisionArchiveRunSource",
    "BrainVisionFileRunSource",
    "BrainVisionSourceCorrection",
    "BrainVisionSourceExclusion",
    "EegRunSource",
    "FifRunSource",
    "discover_mne_runs",
    "discover_processed_brainvision_runs",
    "discover_raw_brainvision_runs",
    "estimate_source_spectrum",
]
