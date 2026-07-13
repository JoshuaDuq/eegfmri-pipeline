"""Strict source discovery for Study 1 preprocessing-stage spectra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from studies.pain_study.study1.figures.continuous_spectrum import (
    discover_final_clean_runs,
    parse_final_clean_filename,
)

RAW_RUN_PATTERN = re.compile(
    r"^ThermalPainEEGFMRI_run(?P<run>\d+)_sub(?P<subject>\d{4})_.+\.vhdr$"
)
PROCESSED_RUN_PATTERN = re.compile(
    r"^ThermalPainEEGFMRI_run(?P<run>\d+)_sub(?P<subject>\d{4})_"
    r".+_scannerpulse_corrected\.vhdr$"
)
HEADER_VALUE_PATTERN = re.compile(
    r"^(?P<key>[^\r\n=]+)=(?P<value>[^\r\n]+)$",
    re.MULTILINE,
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

    @property
    def representation(self) -> str:
        return "brainvision_zip"


@dataclass(frozen=True)
class BrainVisionFileRunSource:
    """One on-disk BrainVision triplet."""

    subject_id: str
    run_id: str
    source_path: str
    header_path: Path

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


EegRunSource = BrainVisionArchiveRunSource | BrainVisionFileRunSource | FifRunSource


def discover_raw_brainvision_runs(
    source_root: Path,
    *,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
) -> tuple[BrainVisionArchiveRunSource, ...]:
    """Discover original 5,000-Hz thermal runs inside participant archives."""
    selected: list[BrainVisionArchiveRunSource] = []
    for archive_path in sorted(Path(source_root).glob("sub_[0-9][0-9][0-9][0-9]_*/raw.zip")):
        with ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            header_members = sorted(
                member
                for member in members
                if PurePosixPath(member).name.startswith("ThermalPainEEGFMRI")
                and PurePosixPath(member).suffix == ".vhdr"
            )
            for header_member in header_members:
                subject_id, run_id = _parse_run_name(
                    PurePosixPath(header_member).name,
                    RAW_RUN_PATTERN,
                )
                if not _subject_is_selected(
                    subject_id,
                    excluded_subjects=excluded_subjects,
                    requested_subjects=requested_subjects,
                ):
                    continue
                header_text = archive.read(header_member).decode("utf-8-sig")
                data_member, marker_member = _archive_triplet_members(
                    header_member,
                    header_text,
                    members,
                )
                _validate_sampling_frequency(
                    header_text,
                    expected_hz=5000.0,
                    source=f"{archive_path}::{header_member}",
                )
                selected.append(
                    BrainVisionArchiveRunSource(
                        subject_id=subject_id,
                        run_id=run_id,
                        source_path=f"{archive_path}::{header_member}",
                        archive_path=archive_path,
                        header_member=header_member,
                        marker_member=marker_member,
                        data_member=data_member,
                    )
                )
    _validate_selected_sources(selected, stage="raw")
    return tuple(sorted(selected, key=_source_sort_key))


def discover_processed_brainvision_runs(
    source_root: Path,
    *,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
) -> tuple[BrainVisionFileRunSource, ...]:
    """Discover BrainVision-processed 1,000-Hz thermal runs."""
    candidates = Path(source_root).glob(
        "sub_[0-9][0-9][0-9][0-9]_*/processed/**/ThermalPainEEGFMRI*.vhdr"
    )
    selected: list[BrainVisionFileRunSource] = []
    for header_path in sorted(path for path in candidates if not path.name.startswith("._")):
        subject_id, run_id = _parse_run_name(header_path.name, PROCESSED_RUN_PATTERN)
        if not _subject_is_selected(
            subject_id,
            excluded_subjects=excluded_subjects,
            requested_subjects=requested_subjects,
        ):
            continue
        header_text = header_path.read_text(encoding="utf-8-sig")
        _validate_file_triplet(header_path, header_text)
        _validate_sampling_frequency(header_text, expected_hz=1000.0, source=str(header_path))
        selected.append(
            BrainVisionFileRunSource(
                subject_id=subject_id,
                run_id=run_id,
                source_path=str(header_path),
                header_path=header_path,
            )
        )
    _validate_selected_sources(selected, stage="processed")
    return tuple(sorted(selected, key=_source_sort_key))


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


def _archive_triplet_members(
    header_member: str,
    header_text: str,
    archive_members: set[str],
) -> tuple[str, str]:
    values = _header_values(header_text)
    parent = PurePosixPath(header_member).parent
    data_member = str(parent / values["DataFile"])
    marker_member = str(parent / values["MarkerFile"])
    for member in (data_member, marker_member):
        if member not in archive_members:
            raise ValueError(f"{header_member} is missing BrainVision member {member}.")
    return data_member, marker_member


def _validate_file_triplet(header_path: Path, header_text: str) -> None:
    values = _header_values(header_text)
    for filename in (values["DataFile"], values["MarkerFile"]):
        path = header_path.parent / filename
        if not path.is_file():
            raise ValueError(f"{header_path} is missing BrainVision member {path}.")


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
            raise ValueError(
                f"Duplicate EEG source for {source.subject_id} run {source.run_id}."
            )
        seen.add(key)


def _source_sort_key(source: EegRunSource) -> tuple[str, int]:
    return source.subject_id, int(source.run_id)


__all__ = [
    "BrainVisionArchiveRunSource",
    "BrainVisionFileRunSource",
    "EegRunSource",
    "FifRunSource",
    "discover_mne_runs",
    "discover_processed_brainvision_runs",
    "discover_raw_brainvision_runs",
]
