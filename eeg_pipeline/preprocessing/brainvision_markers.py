"""Strict BrainVision marker sanitation for scanner-trigger collisions."""

from __future__ import annotations

import re
from dataclasses import dataclass

VOLUME_MARKER_TYPE = "Volume"
VAS_MARKER_TYPE = "Vas_on"
SCANNER_DESCRIPTION = "V  1"
SANITIZED_VAS_DESCRIPTION = "VAS_ON"

_MARKER_ID_PATTERN = re.compile(r"Mk\d+")


@dataclass(frozen=True)
class MarkerSanitizationResult:
    """Sanitized marker text and verified target-marker counts."""

    text: str
    volume_count: int
    vas_count: int


def _parse_integer(value: str, field_name: str, marker_id: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{marker_id} has an invalid {field_name}: {value!r}") from error


def _validate_coordinates(fields: list[str], marker_id: str, n_samples: int) -> None:
    position = _parse_integer(fields[2], "position", marker_id)
    size = _parse_integer(fields[3], "size", marker_id)
    channel = _parse_integer(fields[4], "channel", marker_id)

    if position < 1 or position > n_samples:
        raise ValueError(f"{marker_id} position {position} is outside 1..{n_samples}")
    if size < 1 or position + size - 1 > n_samples:
        raise ValueError(f"{marker_id} size {size} extends outside the recording")
    if channel < 0:
        raise ValueError(f"{marker_id} channel must be non-negative, got {channel}")


def _sanitize_marker_fields(fields: list[str], marker_id: str) -> tuple[int, int]:
    marker_type, description = fields[:2]

    if marker_type == VOLUME_MARKER_TYPE:
        if description != SCANNER_DESCRIPTION:
            raise ValueError(
                f"{marker_id} Volume marker has unexpected description {description!r}"
            )
        return 1, 0

    if marker_type == VAS_MARKER_TYPE:
        if description != SCANNER_DESCRIPTION:
            raise ValueError(
                f"{marker_id} Vas_on marker has unexpected description {description!r}"
            )
        fields[1] = SANITIZED_VAS_DESCRIPTION
        return 0, 1

    if description == SCANNER_DESCRIPTION:
        raise ValueError(
            f"{marker_id} marker type {marker_type!r} unexpectedly uses {SCANNER_DESCRIPTION!r}"
        )

    return 0, 0


def sanitize_vas_marker_text(marker_text: str, *, n_samples: int) -> MarkerSanitizationResult:
    """Replace only ``Vas_on,V  1`` descriptions in a BrainVision marker file."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    output_lines: list[str] = []
    marker_ids: set[str] = set()
    volume_count = 0
    vas_count = 0

    for line in marker_text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content) :]
        if not content.startswith("Mk"):
            output_lines.append(line)
            continue

        marker_id, separator, payload = content.partition("=")
        if separator == "" or _MARKER_ID_PATTERN.fullmatch(marker_id) is None:
            raise ValueError(f"Invalid BrainVision marker record: {content!r}")
        if marker_id in marker_ids:
            raise ValueError(f"Duplicate BrainVision marker identifier: {marker_id}")
        marker_ids.add(marker_id)

        fields = payload.split(",")
        if len(fields) < 5:
            raise ValueError(f"{marker_id} must contain at least five marker fields")

        _validate_coordinates(fields, marker_id, n_samples)
        volume_increment, vas_increment = _sanitize_marker_fields(fields, marker_id)
        volume_count += volume_increment
        vas_count += vas_increment
        output_lines.append(f"{marker_id}={','.join(fields)}{newline}")

    if vas_count == 0:
        raise ValueError("Marker file contains no Vas_on,V  1 markers")

    return MarkerSanitizationResult(
        text="".join(output_lines),
        volume_count=volume_count,
        vas_count=vas_count,
    )
