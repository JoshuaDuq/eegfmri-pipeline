#!/usr/bin/env python3
"""Trim BrainVision recordings to the first contiguous volume block.

This file is standalone and uses only the Python standard library. Copy it to any
computer with Python 3.11 or newer, edit the configuration block below, and run:

    python trim_brainvision_to_volume_bounds.py

The source files are never modified. For each selected ``.vhdr``, the script writes a
new ``.vhdr``/``.vmrk``/``.eeg`` triplet beneath OUTPUT_ROOT. The input directory tree
is preserved.

The first ``Volume,V  1`` of the first contiguous 0.9 s volume train becomes sample 1
(time 0.000 s). That is the first saved BOLD volume. The last sample is one repetition
time after that block's last volume marker, clipped to the end of the recording, so a
partial last TR is kept when EEG stopped mid-volume. Pre-first-marker samples and a
later scanner restart are dropped.

The script deliberately fails on ambiguous, incomplete, unsupported, or inconsistent
BrainVision data instead of guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path


###################################################################
# Configuration — edit these values
###################################################################

# Directory searched for BrainVision header files.
SOURCE_ROOT = Path("/path/to/source_data")

# Separate destination directory. It must not be inside SOURCE_ROOT.
OUTPUT_ROOT = Path("/path/to/trimmed_data")

# Recursive glob evaluated beneath SOURCE_ROOT. Examples:
#   "**/original_untrimmed_5khz/*.vhdr"
#   "**/*.vhdr"
HEADER_GLOB = "**/original_untrimmed_5khz/*.vhdr"

# Explicit SOURCE_ROOT-relative glob exclusions. Remove or edit these for another dataset.
# Documented aborted acquisitions with no volume marker.
EXCLUDED_HEADER_GLOBS = (
    "**/ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h10.39.899.vhdr",
    "**/ThermalPainEEGFMRI_run1_sub0017_2026-08-05_11h02.06.404.vhdr",
)

# A gap this many times the median volume interval starts a new acquisition block.
VOLUME_BLOCK_GAP_FACTOR = 1.5

# Added to every output triplet stem.
OUTPUT_SUFFIX = "_first_to_last_volume"

# BrainVision marker fields identifying the cutting boundary.
VOLUME_MARKER_TYPE = "Volume"
VOLUME_MARKER_DESCRIPTION = "V  1"

# False performs a fast one-pass copy with exact byte/frame validation. True rereads
# every output .eeg and verifies SHA-256, approximately doubling external-drive I/O.
VERIFY_COPIED_DATA_SHA256 = False

# False lets the operating system batch disk flushes, which is much faster for cohorts.
# True forces every individual output file to stable storage before continuing.
SYNC_EACH_OUTPUT_FILE = False


###################################################################
# Implementation — no project-specific dependencies below this line
###################################################################

BINARY_SAMPLE_BYTES = {
    "INT_16": 2,
    "INT_32": 4,
    "IEEE_FLOAT_32": 4,
}
MARKER_LINE_PATTERN = re.compile(r"^Mk(?P<number>\d+)=(?P<payload>.*)$")
TIMESTAMP_PATTERN = re.compile(r"^\d{20}$")
COPY_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class TextFile:
    text: str
    newline: str
    has_utf8_bom: bool


@dataclass(frozen=True)
class Marker:
    marker_type: str
    description: str
    position: int
    size: int
    channel: str
    extra_fields: tuple[str, ...]

    def shifted(self, first_input_sample: int) -> "Marker":
        return replace(self, position=self.position - first_input_sample + 1)

    def render(self, number: int) -> str:
        fields = (
            self.marker_type,
            self.description,
            str(self.position),
            str(self.size),
            self.channel,
            *self.extra_fields,
        )
        return f"Mk{number}={','.join(fields)}"


@dataclass(frozen=True)
class TrimPlan:
    input_header: Path
    input_marker: Path
    input_data: Path
    output_header: Path
    output_marker: Path
    output_data: Path
    header_file: TextFile
    marker_file: TextFile
    number_of_channels: int
    sampling_interval_us: Decimal
    bytes_per_sample: int
    input_samples: int
    first_volume_sample: int
    last_volume_sample: int
    last_inclusive_sample: int
    tr_samples: int
    output_samples: int
    output_markers: tuple[Marker, ...]

    @property
    def frame_bytes(self) -> int:
        return self.number_of_channels * self.bytes_per_sample

    @property
    def cut_bytes(self) -> int:
        return (self.first_volume_sample - 1) * self.frame_bytes


def fail(message: str) -> None:
    raise ValueError(message)


def volume_blocks(positions: tuple[int, ...] | list[int], gap_factor: float) -> list[list[int]]:
    """Split 1-indexed volume samples into contiguous acquisition blocks."""
    ordered = sorted(positions)
    if not ordered:
        return []
    if len(ordered) == 1:
        return [ordered]
    intervals = [current - previous for previous, current in zip(ordered, ordered[1:])]
    median_interval = sorted(intervals)[len(intervals) // 2]
    blocks: list[list[int]] = [[ordered[0]]]
    for current, interval in zip(ordered[1:], intervals):
        if interval > gap_factor * median_interval:
            blocks.append([current])
        else:
            blocks[-1].append(current)
    return blocks


def read_utf8_brainvision(path: Path) -> TextFile:
    raw = path.read_bytes()
    has_utf8_bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        fail(f"{path}: expected UTF-8 BrainVision text: {error}")

    if "\r\n" in text:
        newline = "\r\n"
    elif "\n" in text:
        newline = "\n"
    elif "\r" in text:
        newline = "\r"
    else:
        fail(f"{path}: text file has no detectable line ending")
    return TextFile(text=text, newline=newline, has_utf8_bom=has_utf8_bom)


def encode_text_file(content: str, source: TextFile) -> bytes:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    encoded = normalized.replace("\n", source.newline).encode("utf-8")
    if source.has_utf8_bom:
        return b"\xef\xbb\xbf" + encoded
    return encoded


def parse_sections(path: Path, text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current_section: dict[str, str] | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1]
            if section_name in sections:
                fail(f"{path}:{line_number}: duplicate [{section_name}] section")
            sections[section_name] = {}
            current_section = None if section_name == "Comment" else sections[section_name]
            continue
        if current_section is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in current_section:
            fail(f"{path}:{line_number}: duplicate {key!r} setting")
        current_section[key] = value.strip()

    return sections


def required_setting(
    path: Path,
    sections: dict[str, dict[str, str]],
    section: str,
    key: str,
) -> str:
    try:
        return sections[section][key]
    except KeyError:
        fail(f"{path}: missing required [{section}] {key} setting")


def referenced_triplet_file(header: Path, reference: str, label: str) -> Path:
    if not reference or Path(reference).name != reference or "/" in reference or "\\" in reference:
        fail(f"{header}: {label} must reference a file in the header directory, got {reference!r}")
    path = header.parent / reference
    if not path.is_file():
        fail(f"{header}: referenced {label} does not exist: {path}")
    return path


def parse_marker(path: Path, line_number: int, line: str) -> Marker:
    match = MARKER_LINE_PATTERN.fullmatch(line)
    if match is None:
        fail(f"{path}:{line_number}: malformed marker line: {line!r}")
    fields = match.group("payload").split(",")
    if len(fields) < 5:
        fail(f"{path}:{line_number}: marker has fewer than five fields")

    try:
        position = int(fields[2])
        size = int(fields[3])
    except ValueError:
        fail(f"{path}:{line_number}: marker position and size must be integers")
    if position < 1 or size < 1:
        fail(f"{path}:{line_number}: marker position and size must be positive")

    return Marker(
        marker_type=fields[0],
        description=fields[1],
        position=position,
        size=size,
        channel=fields[4],
        extra_fields=tuple(fields[5:]),
    )


def parse_markers(path: Path, text: str) -> tuple[Marker, ...]:
    markers = []
    in_marker_section = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line == "[Marker Infos]":
            in_marker_section = True
            continue
        if in_marker_section and line.startswith("["):
            fail(f"{path}:{line_number}: sections after [Marker Infos] are unsupported")
        if in_marker_section and line.startswith("Mk"):
            markers.append(parse_marker(path, line_number, line))

    if not in_marker_section:
        fail(f"{path}: missing [Marker Infos] section")
    if not markers:
        fail(f"{path}: contains no markers")
    if any(current.position < previous.position for previous, current in zip(markers, markers[1:])):
        fail(f"{path}: marker positions are not monotonically ordered")
    return tuple(markers)


def adjusted_new_segment(initial_segment: Marker, offset_us: int) -> Marker:
    if len(initial_segment.extra_fields) != 1:
        fail("Initial New Segment marker must contain exactly one timestamp field")
    timestamp = initial_segment.extra_fields[0]
    if TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
        fail(f"Initial New Segment timestamp must contain 20 digits, got {timestamp!r}")

    start = datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S")
    start += timedelta(microseconds=int(timestamp[14:]))
    trimmed_start = start + timedelta(microseconds=offset_us)
    adjusted_timestamp = trimmed_start.strftime("%Y%m%d%H%M%S%f")
    return replace(initial_segment, position=1, size=1, extra_fields=(adjusted_timestamp,))


def build_output_markers(
    marker_path: Path,
    markers: tuple[Marker, ...],
    first_volume_sample: int,
    last_inclusive_sample: int,
    last_volume_sample: int,
    sampling_interval_us: Decimal,
) -> tuple[Marker, ...]:
    initial_segments = [
        marker
        for marker in markers
        if marker.marker_type == "New Segment" and marker.position == 1
    ]
    if len(initial_segments) != 1:
        fail(f"{marker_path}: expected exactly one New Segment marker at sample 1")

    offset_us_decimal = Decimal(first_volume_sample - 1) * sampling_interval_us
    if offset_us_decimal != offset_us_decimal.to_integral_value():
        fail(f"{marker_path}: cut time cannot be represented as an integer microsecond")
    new_segment = adjusted_new_segment(initial_segments[0], int(offset_us_decimal))

    retained = []
    output_samples = last_inclusive_sample - first_volume_sample + 1
    for marker in markers:
        marker_end = marker.position + marker.size - 1
        if marker.position < first_volume_sample <= marker_end:
            fail(f"{marker_path}: a marker spans the trimming boundary: {marker}")
        if marker.position <= last_inclusive_sample < marker_end:
            fail(f"{marker_path}: a marker spans the final trimming boundary: {marker}")
        if (
            marker is initial_segments[0]
            or marker.position < first_volume_sample
            or marker.position > last_inclusive_sample
        ):
            continue
        shifted = marker.shifted(first_volume_sample)
        if shifted.position + shifted.size - 1 > output_samples:
            fail(f"{marker_path}: retained marker extends beyond output data: {marker}")
        retained.append(shifted)

    output = (new_segment, *retained)
    matching_boundaries = [
        marker
        for marker in output
        if marker.marker_type == VOLUME_MARKER_TYPE
        and marker.description == VOLUME_MARKER_DESCRIPTION
        and marker.position == 1
    ]
    if len(matching_boundaries) != 1:
        fail(f"{marker_path}: expected exactly one first volume marker at output sample 1")

    last_volume_output = last_volume_sample - first_volume_sample + 1
    matching_last_volumes = [
        marker
        for marker in output
        if marker.marker_type == VOLUME_MARKER_TYPE
        and marker.description == VOLUME_MARKER_DESCRIPTION
        and marker.position == last_volume_output
    ]
    if len(matching_last_volumes) != 1:
        fail(
            f"{marker_path}: expected the last volume marker at output sample "
            f"{last_volume_output}"
        )
    return output


def replace_setting(text: str, key: str, value: str, path: Path) -> str:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(key)}\s*=).*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        fail(f"{path}: expected exactly one {key} setting, found {len(matches)}")
    return pattern.sub(lambda match: f"{match.group('prefix')}{value}", text)


def marker_preamble(marker_path: Path, text: str) -> str:
    lines = text.splitlines()
    try:
        section_index = next(i for i, line in enumerate(lines) if line.strip() == "[Marker Infos]")
    except StopIteration:
        fail(f"{marker_path}: missing [Marker Infos] section")

    preamble_lines = lines[: section_index + 1]
    for line in lines[section_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("["):
            fail(f"{marker_path}: sections after [Marker Infos] are unsupported")
        if not stripped.startswith("Mk"):
            preamble_lines.append(line)
    while preamble_lines and not preamble_lines[-1].strip():
        preamble_lines.pop()
    return "\n".join(preamble_lines)


def render_marker_file(plan: TrimPlan) -> str:
    text = replace_setting(
        plan.marker_file.text,
        "DataFile",
        plan.output_data.name,
        plan.input_marker,
    )
    preamble = marker_preamble(plan.input_marker, text)
    marker_lines = [marker.render(number) for number, marker in enumerate(plan.output_markers, 1)]
    return f"{preamble}\n" + "\n".join(marker_lines) + "\n"


def render_header_file(plan: TrimPlan) -> str:
    text = replace_setting(
        plan.header_file.text,
        "DataFile",
        plan.output_data.name,
        plan.input_header,
    )
    return replace_setting(text, "MarkerFile", plan.output_marker.name, plan.input_header)


def create_trim_plan(header: Path, source_root: Path, output_root: Path) -> TrimPlan:
    header_file = read_utf8_brainvision(header)
    sections = parse_sections(header, header_file.text)
    common = "Common Infos"

    codepage = required_setting(header, sections, common, "Codepage")
    if codepage.upper() != "UTF-8":
        fail(f"{header}: only Codepage=UTF-8 is supported, got {codepage!r}")
    data_format = required_setting(header, sections, common, "DataFormat")
    if data_format.upper() != "BINARY":
        fail(f"{header}: only DataFormat=BINARY is supported, got {data_format!r}")
    orientation = required_setting(header, sections, common, "DataOrientation")
    if orientation.upper() != "MULTIPLEXED":
        fail(f"{header}: only DataOrientation=MULTIPLEXED is supported, got {orientation!r}")

    try:
        number_of_channels = int(required_setting(header, sections, common, "NumberOfChannels"))
    except ValueError:
        fail(f"{header}: NumberOfChannels must be an integer")
    if number_of_channels < 1:
        fail(f"{header}: NumberOfChannels must be positive")

    try:
        sampling_interval_us = Decimal(
            required_setting(header, sections, common, "SamplingInterval")
        )
    except InvalidOperation as error:
        fail(f"{header}: invalid SamplingInterval: {error}")
    if not sampling_interval_us.is_finite() or sampling_interval_us <= 0:
        fail(f"{header}: SamplingInterval must be finite and positive")

    binary_format = required_setting(header, sections, "Binary Infos", "BinaryFormat").upper()
    if binary_format not in BINARY_SAMPLE_BYTES:
        supported = ", ".join(BINARY_SAMPLE_BYTES)
        fail(f"{header}: unsupported BinaryFormat={binary_format!r}; supported: {supported}")
    bytes_per_sample = BINARY_SAMPLE_BYTES[binary_format]

    input_data = referenced_triplet_file(
        header,
        required_setting(header, sections, common, "DataFile"),
        "DataFile",
    )
    input_marker = referenced_triplet_file(
        header,
        required_setting(header, sections, common, "MarkerFile"),
        "MarkerFile",
    )
    marker_file = read_utf8_brainvision(input_marker)
    marker_sections = parse_sections(input_marker, marker_file.text)
    marker_data_reference = required_setting(input_marker, marker_sections, common, "DataFile")
    if marker_data_reference != input_data.name:
        fail(
            f"{input_marker}: DataFile references {marker_data_reference!r}, "
            f"but the header references {input_data.name!r}"
        )

    frame_bytes = number_of_channels * bytes_per_sample
    data_bytes = input_data.stat().st_size
    if data_bytes == 0 or data_bytes % frame_bytes:
        fail(f"{input_data}: byte size {data_bytes} is not aligned to {frame_bytes}-byte frames")
    input_samples = data_bytes // frame_bytes

    markers = parse_markers(input_marker, marker_file.text)
    volume_markers = [
        marker
        for marker in markers
        if marker.marker_type == VOLUME_MARKER_TYPE
        and marker.description == VOLUME_MARKER_DESCRIPTION
    ]
    if not volume_markers:
        fail(
            f"{input_marker}: no {VOLUME_MARKER_TYPE!r},"
            f"{VOLUME_MARKER_DESCRIPTION!r} marker"
        )
    blocks = volume_blocks(
        [marker.position for marker in volume_markers],
        VOLUME_BLOCK_GAP_FACTOR,
    )
    first_block = blocks[0]
    if len(first_block) < 2:
        fail(f"{input_marker}: first volume block has fewer than two markers")
    first_volume_sample = first_block[0]
    last_volume_sample = first_block[-1]
    if sum(marker.position == first_volume_sample for marker in volume_markers) != 1:
        fail(f"{input_marker}: first volume boundary is duplicated at sample {first_volume_sample}")
    if sum(marker.position == last_volume_sample for marker in volume_markers) != 1:
        fail(f"{input_marker}: last volume boundary is duplicated at sample {last_volume_sample}")
    if first_volume_sample > input_samples:
        fail(f"{input_marker}: first volume marker is beyond the end of the data")
    if last_volume_sample > input_samples:
        fail(f"{input_marker}: last volume marker is beyond the end of the data")

    block_intervals = [
        current - previous for previous, current in zip(first_block, first_block[1:])
    ]
    tr_samples = sorted(block_intervals)[len(block_intervals) // 2]
    if tr_samples < 1:
        fail(f"{input_marker}: first volume block has a non-positive repetition time")
    jitter = max(abs(interval - tr_samples) for interval in block_intervals)
    if jitter > 1:
        fail(
            f"{input_marker}: first volume block is not a regular train "
            f"(median {tr_samples} samples, max deviation {jitter})"
        )
    last_inclusive_sample = min(last_volume_sample + tr_samples - 1, input_samples)
    if last_inclusive_sample < last_volume_sample:
        fail(f"{input_marker}: last volume marker is beyond the end of the data")

    output_directory = output_root / header.parent.relative_to(source_root)
    output_stem = f"{header.stem}{OUTPUT_SUFFIX}"
    output_header = output_directory / f"{output_stem}.vhdr"
    output_marker = output_directory / f"{output_stem}.vmrk"
    output_data = output_directory / f"{output_stem}.eeg"
    output_samples = last_inclusive_sample - first_volume_sample + 1
    output_markers = build_output_markers(
        input_marker,
        markers,
        first_volume_sample,
        last_inclusive_sample,
        last_volume_sample,
        sampling_interval_us,
    )

    return TrimPlan(
        input_header=header,
        input_marker=input_marker,
        input_data=input_data,
        output_header=output_header,
        output_marker=output_marker,
        output_data=output_data,
        header_file=header_file,
        marker_file=marker_file,
        number_of_channels=number_of_channels,
        sampling_interval_us=sampling_interval_us,
        bytes_per_sample=bytes_per_sample,
        input_samples=input_samples,
        first_volume_sample=first_volume_sample,
        last_volume_sample=last_volume_sample,
        last_inclusive_sample=last_inclusive_sample,
        tr_samples=tr_samples,
        output_samples=output_samples,
        output_markers=output_markers,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim BrainVision recordings to the first contiguous volume block."
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--header-glob", default=None)
    return parser.parse_args(argv)


def validate_configuration(
    source_root: Path | None = None,
    output_root: Path | None = None,
    header_glob: str | None = None,
) -> tuple[Path, Path, str]:
    resolved_source = (source_root or SOURCE_ROOT).expanduser().resolve()
    resolved_output = (output_root or OUTPUT_ROOT).expanduser().resolve()
    resolved_glob = HEADER_GLOB if header_glob is None else header_glob
    if not resolved_source.is_dir():
        fail(f"SOURCE_ROOT is not a directory: {resolved_source}")
    if resolved_source == resolved_output or resolved_source in resolved_output.parents:
        fail("OUTPUT_ROOT must be separate from and not inside SOURCE_ROOT")
    if not resolved_glob.strip():
        fail("HEADER_GLOB cannot be empty")
    if not OUTPUT_SUFFIX or any(separator in OUTPUT_SUFFIX for separator in ("/", "\\")):
        fail("OUTPUT_SUFFIX must be a non-empty filename suffix")
    return resolved_source, resolved_output, resolved_glob


def build_plans(
    source_root: Path, output_root: Path, header_glob: str = HEADER_GLOB
) -> tuple[TrimPlan, ...]:
    headers = sorted(
        path
        for path in source_root.glob(header_glob)
        if path.is_file() and not path.name.startswith("._")
    )
    if not headers:
        fail(f"No .vhdr files matched {header_glob!r} beneath {source_root}")

    excluded = [
        path
        for path in headers
        if any(path.relative_to(source_root).match(pattern) for pattern in EXCLUDED_HEADER_GLOBS)
    ]
    headers = [path for path in headers if path not in excluded]
    if not headers:
        fail("Every discovered header was explicitly excluded")
    for path in excluded:
        print(f"Explicitly excluded: {path.relative_to(source_root)}")

    plans = tuple(create_trim_plan(header, source_root, output_root) for header in headers)
    output_paths = [
        path
        for plan in plans
        for path in (plan.output_header, plan.output_marker, plan.output_data)
    ]
    if len(output_paths) != len(set(output_paths)):
        fail("Multiple inputs resolve to the same output path")
    existing = [path for path in output_paths if path.exists()]
    if existing:
        formatted = "\n  ".join(str(path) for path in existing)
        fail(f"Refusing to overwrite existing output files:\n  {formatted}")
    return plans


def copy_trimmed_data(
    plan: TrimPlan,
    destination: Path,
    verify_sha256: bool,
) -> str | None:
    digest = hashlib.sha256() if verify_sha256 else None
    with plan.input_data.open("rb") as source, destination.open("xb") as output:
        source.seek(plan.cut_bytes)
        remaining_bytes = plan.output_samples * plan.frame_bytes
        while remaining_bytes:
            chunk = source.read(min(COPY_CHUNK_BYTES, remaining_bytes))
            if not chunk:
                fail(f"{plan.input_data}: data ended before the final volume boundary")
            output.write(chunk)
            remaining_bytes -= len(chunk)
            if digest is not None:
                digest.update(chunk)
        if SYNC_EACH_OUTPUT_FILE:
            output.flush()
            os.fsync(output.fileno())

    expected_bytes = plan.output_samples * plan.frame_bytes
    if destination.stat().st_size != expected_bytes:
        fail(
            f"{destination}: wrote {destination.stat().st_size} bytes; "
            f"expected {expected_bytes}"
        )
    return digest.hexdigest() if digest is not None else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_exclusively(path: Path, content: bytes) -> None:
    with path.open("xb") as file:
        file.write(content)
        if SYNC_EACH_OUTPUT_FILE:
            file.flush()
            os.fsync(file.fileno())


def trim_recording(plan: TrimPlan) -> None:
    plan.output_header.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{plan.output_header.stem}-", dir=plan.output_header.parent)
    )
    temporary_data = temporary_directory / plan.output_data.name
    temporary_marker = temporary_directory / plan.output_marker.name
    temporary_header = temporary_directory / plan.output_header.name

    try:
        copied_digest = copy_trimmed_data(
            plan,
            temporary_data,
            VERIFY_COPIED_DATA_SHA256,
        )
        if copied_digest is not None and sha256_file(temporary_data) != copied_digest:
            fail(f"{plan.input_data}: copied EEG data failed SHA-256 verification")

        marker_content = encode_text_file(render_marker_file(plan), plan.marker_file)
        header_content = encode_text_file(render_header_file(plan), plan.header_file)
        write_bytes_exclusively(temporary_marker, marker_content)
        write_bytes_exclusively(temporary_header, header_content)

        verified_markers = parse_markers(
            temporary_marker,
            read_utf8_brainvision(temporary_marker).text,
        )
        first_volume = next(
            marker
            for marker in verified_markers
            if marker.marker_type == VOLUME_MARKER_TYPE
            and marker.description == VOLUME_MARKER_DESCRIPTION
        )
        if first_volume.position != 1:
            fail(f"{temporary_marker}: first volume marker is not at sample 1")

        for output_path in (plan.output_data, plan.output_marker, plan.output_header):
            if output_path.exists():
                fail(f"Refusing to overwrite output created during this run: {output_path}")
        os.replace(temporary_data, plan.output_data)
        os.replace(temporary_marker, plan.output_marker)
        os.replace(temporary_header, plan.output_header)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def format_seconds(samples: int, sampling_interval_us: Decimal) -> str:
    seconds = Decimal(samples) * sampling_interval_us / Decimal(1_000_000)
    return f"{seconds:.6f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root, output_root, header_glob = validate_configuration(
        source_root=args.source_root,
        output_root=args.output_root,
        header_glob=args.header_glob,
    )
    plans = build_plans(source_root, output_root, header_glob)
    print(
        f"Validated {len(plans)} recording(s). "
        "Trimming to first volume marker through last marker + 1 TR...\n"
    )

    for index, plan in enumerate(plans, start=1):
        removed_samples = plan.first_volume_sample - 1
        removed_seconds = format_seconds(removed_samples, plan.sampling_interval_us)
        trailing_samples = plan.input_samples - plan.last_inclusive_sample
        trailing_seconds = format_seconds(trailing_samples, plan.sampling_interval_us)
        kept_after_last = plan.last_inclusive_sample - plan.last_volume_sample + 1
        kept_after_last_seconds = format_seconds(kept_after_last, plan.sampling_interval_us)
        trim_recording(plan)
        print(
            f"[{index:>{len(str(len(plans)))}}/{len(plans)}] "
            f"{plan.input_header.name}\n"
            f"    removed before: {removed_samples:,} samples ({removed_seconds} s)\n"
            f"    removed after:  {trailing_samples:,} samples ({trailing_seconds} s)\n"
            f"    last TR kept:   {kept_after_last:,} of {plan.tr_samples:,} samples "
            f"({kept_after_last_seconds} s)\n"
            f"    first V  1 is sample 1; last V  1 is not the final sample unless "
            f"EEG stopped on that marker\n"
            f"    wrote   {plan.output_header}"
        )

    print(f"\nCompleted {len(plans)} recording(s). Sources were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
