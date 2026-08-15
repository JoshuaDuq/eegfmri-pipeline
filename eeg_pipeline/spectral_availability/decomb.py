from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from eeg_pipeline.spectral_availability.model import (
    FrequencyInterval,
    RecordingExclusions,
    RecordingKey,
)


_REQUIRED_COLUMN_NAMES = (
    "recording",
    "unavailable_low_hz",
    "unavailable_high_hz",
    "outcome",
    "removal_round",
)
_REQUIRED_COLUMNS = frozenset(_REQUIRED_COLUMN_NAMES)
_CHUNK_SIZE = 10_000
_RECORDING_PATTERN = re.compile(
    r"sub-(?P<subject>[A-Za-z0-9+]+)"
    r"(?:_ses-(?P<session>[A-Za-z0-9+]+))?"
    r"_task-(?P<task>[A-Za-z0-9+]+)"
    r"_run-(?P<run>[0-9]+)_eeg"
)
_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TERMINAL_OUTCOME = "no_line_detected"
_FINITE_OUTCOMES = frozenset(
    {
        "line_detected",
        "scanner_harmonics_detected",
    }
)


@dataclass(frozen=True)
class DecombManifest:
    path: Path
    sha256: str
    exclusions: tuple[RecordingExclusions, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str):
            raise TypeError("sha256 must be a string")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a 64-character lowercase hexadecimal string")

        exclusions = tuple(self.exclusions)
        if any(not isinstance(exclusion, RecordingExclusions) for exclusion in exclusions):
            raise TypeError("exclusions must contain RecordingExclusions values")

        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "exclusions", exclusions)


class _ManifestRow(NamedTuple):
    recording: str
    unavailable_low_hz: str
    unavailable_high_hz: str
    outcome: str
    removal_round: str


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} must be a file: {path}")


def _validate_provenance(description_path: Path) -> None:
    description = json.loads(description_path.read_text(encoding="utf-8"))
    if not isinstance(description, Mapping):
        raise ValueError("dataset_description.json must contain a JSON mapping")

    generated_by = description.get("GeneratedBy")
    if not isinstance(generated_by, list):
        raise ValueError("dataset_description.json GeneratedBy must be a list")

    decomb_count = 0
    for entry in generated_by:
        if not isinstance(entry, Mapping):
            raise ValueError("dataset_description.json GeneratedBy entries must be mappings")
        name = entry.get("Name")
        if not isinstance(name, str) or not name:
            raise ValueError("dataset_description.json GeneratedBy entries require a string Name")
        decomb_count += name.casefold() == "decomb"

    if decomb_count != 1:
        raise ValueError(
            "dataset_description.json GeneratedBy must contain exactly one Decomb entry"
        )


def _recording_key(recording: str, row_number: int) -> RecordingKey:
    match = _RECORDING_PATTERN.fullmatch(recording)
    if match is None:
        raise ValueError(
            f"row {row_number} recording {recording!r} is not a supported BIDS EEG name"
        )
    try:
        return RecordingKey(**match.groupdict())
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number} recording {recording!r} has invalid BIDS identity"
        ) from error


def _frequency_interval(
    low_text: str,
    high_text: str,
    *,
    row_number: int,
    recording: str,
) -> FrequencyInterval:
    context = f"row {row_number} recording {recording!r}"
    if _NUMBER_PATTERN.fullmatch(low_text) is None:
        raise ValueError(f"{context} has invalid unavailable_low_hz {low_text!r}")
    if _NUMBER_PATTERN.fullmatch(high_text) is None:
        raise ValueError(f"{context} has invalid unavailable_high_hz {high_text!r}")
    try:
        return FrequencyInterval(float(low_text), float(high_text))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} has invalid unavailable interval geometry") from error


def _missing_columns_error(columns: set[str] | frozenset[str]) -> ValueError:
    return ValueError("Decomb manifest is missing required columns: " + ", ".join(sorted(columns)))


def _parse_header(header_bytes: bytes) -> tuple[str, ...]:
    if not header_bytes:
        raise _missing_columns_error(_REQUIRED_COLUMNS)

    try:
        header = header_bytes.removesuffix(b"\n").removesuffix(b"\r").decode("utf-8")
        columns = tuple(next(csv.reader([header], delimiter="\t", strict=True)))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Decomb manifest has an invalid UTF-8 TSV header") from error

    if any(column == "" for column in columns):
        raise ValueError("Decomb manifest has a blank header name")

    counts = Counter(columns)
    duplicates = sorted(column for column, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("Decomb manifest has duplicate header names: " + ", ".join(duplicates))

    missing_columns = _REQUIRED_COLUMNS - set(columns)
    if missing_columns:
        raise _missing_columns_error(missing_columns)
    return columns


def _validate_raw_structure(manifest_bytes: bytes) -> None:
    stream = io.BytesIO(manifest_bytes)
    _parse_header(stream.readline())

    has_data_row = False
    for row_number, raw_line in enumerate(stream, start=2):
        has_data_row = True
        row = raw_line.removesuffix(b"\n").removesuffix(b"\r")
        if not row or all(field == b"" for field in row.split(b"\t")):
            raise ValueError(f"row {row_number} is an empty Decomb manifest data row")

    if not has_data_row:
        raise ValueError("Decomb manifest must contain at least one data row")


def _read_rows(manifest_bytes: bytes):
    _validate_raw_structure(manifest_bytes)
    try:
        chunks = pd.read_csv(
            io.BytesIO(manifest_bytes),
            sep="\t",
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
            usecols=_REQUIRED_COLUMN_NAMES,
            chunksize=_CHUNK_SIZE,
        )
    except pd.errors.EmptyDataError as error:
        raise _missing_columns_error(_REQUIRED_COLUMNS) from error

    row_number = 2
    for chunk in chunks:
        positions = tuple(chunk.columns.get_loc(name) for name in _REQUIRED_COLUMN_NAMES)
        for values in chunk.itertuples(index=False, name=None):
            yield row_number, _ManifestRow(*(values[position] for position in positions))
            row_number += 1


def load_decomb_manifest(path: str | Path) -> DecombManifest:
    manifest_path = Path(path)
    _require_file(manifest_path, "Decomb manifest")

    description_path = manifest_path.with_name("dataset_description.json")
    _require_file(description_path, "dataset_description.json")
    _validate_provenance(description_path)

    manifest_bytes = manifest_path.read_bytes()
    recording_order: list[str] = []
    keys: dict[str, RecordingKey] = {}
    recording_by_key: dict[RecordingKey, str] = {}
    intervals: dict[str, list[FrequencyInterval]] = {}
    terminal_counts: dict[str, int] = {}

    for row_number, row in _read_rows(manifest_bytes):
        recording = row.recording
        key = _recording_key(recording, row_number)
        previous_recording = recording_by_key.get(key)
        if previous_recording is not None and previous_recording != recording:
            raise ValueError(
                f"row {row_number} recording {recording!r} conflicts with "
                f"recording {previous_recording!r} for the same BIDS identity"
            )
        if recording not in keys:
            recording_order.append(recording)
            keys[recording] = key
            recording_by_key[key] = recording
            intervals[recording] = []
            terminal_counts[recording] = 0

        low_text = row.unavailable_low_hz
        high_text = row.unavailable_high_hz
        outcome = row.outcome
        low_is_blank = low_text == ""
        high_is_blank = high_text == ""
        context = f"row {row_number} recording {recording!r}"

        if low_is_blank != high_is_blank:
            raise ValueError(f"{context} must provide both unavailable interval fields")
        if low_is_blank:
            if outcome != _TERMINAL_OUTCOME:
                raise ValueError(f"{context} has blank interval fields without terminal outcome")
            terminal_counts[recording] += 1
            continue
        if outcome not in _FINITE_OUTCOMES:
            supported = ", ".join(sorted(_FINITE_OUTCOMES))
            raise ValueError(
                f"{context} has unsupported finite outcome {outcome!r}; "
                f"expected one of: {supported}"
            )

        intervals[recording].append(
            _frequency_interval(
                low_text,
                high_text,
                row_number=row_number,
                recording=recording,
            )
        )

    exclusions = []
    for recording in recording_order:
        terminal_count = terminal_counts[recording]
        if terminal_count != 1:
            raise ValueError(
                f"recording {recording!r} requires exactly one terminal "
                f"{_TERMINAL_OUTCOME!r} row; found {terminal_count}"
            )
        exclusions.append(
            RecordingExclusions(
                key=keys[recording],
                intervals=tuple(intervals[recording]),
            )
        )

    return DecombManifest(
        path=manifest_path,
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        exclusions=tuple(exclusions),
    )


__all__ = ["DecombManifest", "load_decomb_manifest"]
