"""Organize original and BrainVision-processed EEG source recordings."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ORIGINAL_DIRECTORY = "original_untrimmed_5khz"
PROCESSED_DIRECTORY = "analyzer_brainvision_processed_1khz"
TRIPLET_SUFFIXES = (".vhdr", ".vmrk", ".eeg")
SAMPLING_INTERVAL_PATTERN = re.compile(r"^SamplingInterval=(?P<value>[0-9.]+)$", re.MULTILINE)


@dataclass(frozen=True)
class OrganizationResult:
    """Validated organization counts for one participant."""

    subject: str
    original_triplets: int
    processed_triplets: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _triplets(
    root: Path,
    expected_frequency_hz: float,
    *,
    recursive: bool = False,
    allow_empty: bool = False,
) -> tuple[tuple[Path, ...], ...]:
    candidates = root.rglob("*.vhdr") if recursive else root.glob("*.vhdr")
    headers = sorted(path for path in candidates if not path.name.startswith("._"))
    if not headers and not allow_empty:
        raise FileNotFoundError(f"No BrainVision headers found in {root}")
    triplets = []
    for header in headers:
        files = tuple(header.with_suffix(suffix) for suffix in TRIPLET_SUFFIXES)
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete BrainVision triplet for {header}: {missing}")
        text = header.read_text(encoding="utf-8-sig")
        match = SAMPLING_INTERVAL_PATTERN.search(text)
        if match is None:
            raise ValueError(f"BrainVision header has no SamplingInterval: {header}")
        frequency_hz = 1_000_000.0 / float(match.group("value"))
        if frequency_hz != expected_frequency_hz:
            raise ValueError(
                f"Expected {expected_frequency_hz} Hz in {header}, got {frequency_hz} Hz"
            )
        triplets.append(files)
    return tuple(triplets)


def _copy_original_triplets(
    source: Path, destination: Path, triplets: tuple[tuple[Path, ...], ...]
) -> int:
    destination.mkdir()
    for triplet in triplets:
        for source_path in triplet:
            destination_path = destination / source_path.relative_to(source)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            if _sha256(destination_path) != _sha256(source_path):
                raise RuntimeError(f"Checksum mismatch after copying {source_path}")
    return len(triplets)


def _move_processed_triplets(
    source: Path, destination: Path, triplets: tuple[tuple[Path, ...], ...]
) -> int:
    old_directories = sorted(
        {path.parent for triplet in triplets for path in triplet if path.parent != source},
        reverse=True,
    )
    destination.mkdir()
    for triplet in triplets:
        for source_path in triplet:
            destination_path = destination / source_path.relative_to(source)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            source_hash = _sha256(source_path)
            shutil.move(source_path, destination_path)
            if _sha256(destination_path) != source_hash:
                raise RuntimeError(f"Checksum mismatch after moving {source_path}")
    for metadata_path in source.rglob("._*"):
        metadata_path.unlink()
    for directory in old_directories:
        if directory.is_dir():
            directory.rmdir()
    return len(triplets)


def organize_subject_eeg(
    subject: str,
    participant_raw: Path,
    source_subject: Path,
) -> OrganizationResult:
    """Copy 5 kHz triplets and move 1 kHz triplets into labeled directories."""
    eeg_directory = source_subject / "eeg"
    if not participant_raw.is_dir():
        raise FileNotFoundError(f"Participant raw directory not found: {participant_raw}")
    if not eeg_directory.is_dir():
        raise FileNotFoundError(f"Source EEG directory not found: {eeg_directory}")
    original_destination = eeg_directory / ORIGINAL_DIRECTORY
    processed_destination = eeg_directory / PROCESSED_DIRECTORY
    if original_destination.exists():
        raise FileExistsError(f"EEG source is already organized: {eeg_directory}")
    if processed_destination.exists():
        raise FileExistsError(f"Processed EEG destination exists: {processed_destination}")

    original_sources = _triplets(participant_raw, 5_000.0)
    processed_sources = _triplets(
        eeg_directory,
        1_000.0,
        recursive=True,
        allow_empty=True,
    )
    original_triplets = _copy_original_triplets(
        participant_raw, original_destination, original_sources
    )
    processed_triplets = 0
    if processed_sources:
        processed_triplets = _move_processed_triplets(
            eeg_directory, processed_destination, processed_sources
        )
    _triplets(original_destination, 5_000.0)
    if processed_sources:
        _triplets(processed_destination, 1_000.0, recursive=True)
    return OrganizationResult(subject, original_triplets, processed_triplets)


def _participant_raw_directory(kingston_root: Path, subject: str) -> Path:
    candidates = sorted(
        path / "raw"
        for path in kingston_root.glob(f"sub_{subject}_*")
        if path.is_dir() and "EXCL" not in path.name and "eeg_only" not in path.name
    )
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one main-cohort raw directory for sub-{subject}, found {candidates}"
        )
    return candidates[0]


def discover_unorganized_subjects(
    source_data_root: Path,
    subjects: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Discover source-data subjects whose EEG recordings are not yet organized."""
    subject_directories = {
        path.name.removeprefix("sub-"): path
        for path in source_data_root.glob("sub-*")
        if path.is_dir() and (path / "eeg").is_dir()
    }
    if subjects is not None:
        requested_subjects = set(subjects)
        missing_subjects = sorted(requested_subjects - set(subject_directories))
        if missing_subjects:
            raise FileNotFoundError(
                f"Source EEG directories do not exist for subjects: {missing_subjects}"
            )
        subject_directories = {
            subject: subject_directories[subject] for subject in requested_subjects
        }

    unorganized_subjects = []
    for subject, source_subject in sorted(subject_directories.items()):
        eeg_directory = source_subject / "eeg"
        original_exists = (eeg_directory / ORIGINAL_DIRECTORY).exists()
        if not original_exists:
            unorganized_subjects.append(subject)

    if not unorganized_subjects:
        raise FileNotFoundError(f"No unorganized EEG subjects found in {source_data_root}")
    return tuple(unorganized_subjects)


def organize_cohort(
    kingston_root: Path,
    source_data_root: Path,
    subjects: Sequence[str] | None = None,
) -> tuple[OrganizationResult, ...]:
    """Organize every newly discovered simultaneous EEG-fMRI subject."""
    discovered_subjects = discover_unorganized_subjects(source_data_root, subjects)
    return tuple(
        organize_subject_eeg(
            subject,
            _participant_raw_directory(kingston_root, subject),
            source_data_root / f"sub-{subject}",
        )
        for subject in discovered_subjects
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kingston-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Subject label without 'sub-'; repeat to select subjects. Defaults to all new.",
    )
    arguments = parser.parse_args()
    for result in organize_cohort(
        arguments.kingston_root,
        arguments.source_data_root,
        arguments.subject,
    ):
        print(
            f"sub-{result.subject}: {result.original_triplets} original 5 kHz, "
            f"{result.processed_triplets} BrainVision-processed 1 kHz"
        )


if __name__ == "__main__":
    main()
