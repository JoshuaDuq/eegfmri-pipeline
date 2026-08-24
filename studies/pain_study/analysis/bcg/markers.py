"""Read and write BrainVision marker files, so Analyzer can correct the beats we found.

Analyzer's own correction beats ours at the beats it marked -- 0.16% R-locked residual
against our 2.03%, and 0.54 alpha retained against 0.34 -- so the better use of recovered
beats is to hand them back rather than to correct with them ourselves. These exports sit at
the `Pulse Artifact Correction (Mark R peaks)` node, which is exactly where an augmented
marker set belongs.

Two details make this less trivial than appending lines. Markers are numbered, and the
`[Marker User Infos]` section assigns properties *by marker number*, so inserting a marker
renumbers the rest and invalidates every reference unless they are remapped. And Analyzer
flags each of its own R peaks with `BrainVision.CustomMarker`, so a recovered marker
carries the same property or it is not the same kind of object.

Everything not deliberately changed is preserved byte for byte, CRLF endings included,
because Analyzer has to accept the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

MARKER_PATTERN = re.compile(r"^Mk(\d+)=(.*)$")
PROPERTY_PATTERN = re.compile(r"^Prop(\d+)=Mk(\d+),(.*)$")
USER_INFO_HEADER = "[Marker User Infos]"
PULSE_TYPE = "Pulse Artifact"
PULSE_DESCRIPTION = "R"
CUSTOM_MARKER_PROPERTY = "bool,BrainVision.CustomMarker,true"
LINE_ENDING = "\r\n"


@dataclass(frozen=True)
class Marker:
    type: str
    description: str
    position: int  # 1-based sample index, the BrainVision convention
    size: int
    channel: int
    extra: tuple[str, ...] = ()  # trailing fields, e.g. New Segment's timestamp
    properties: tuple[str, ...] = ()  # payloads from [Marker User Infos]

    def render(self, number: int) -> str:
        fields = [
            self.type,
            self.description,
            str(self.position),
            str(self.size),
            str(self.channel),
            *self.extra,
        ]
        return f"Mk{number}=" + ",".join(fields)


@dataclass(frozen=True)
class MarkerFile:
    preamble: list[str]
    markers: list[Marker]
    user_info_preamble: list[str]

    @property
    def pulse_channel(self) -> int:
        """The channel Analyzer relates its own R markers to, so new ones can match it."""
        for marker in self.markers:
            if marker.type == PULSE_TYPE and marker.description == PULSE_DESCRIPTION:
                return marker.channel
        return 0


def read_marker_file(path: Path | str) -> MarkerFile:
    """Parse a .vmrk into its preamble, its markers, and their user properties."""
    lines = Path(path).read_bytes().decode("utf-8").split(LINE_ENDING)
    # The file ends with a line ending, so the split leaves a final empty element. Drop it
    # here rather than carrying it into a section; the writer re-adds the ending.
    if lines and lines[-1] == "":
        lines = lines[:-1]

    preamble: list[str] = []
    user_info_preamble: list[str] = []
    markers: list[Marker] = []
    properties: dict[int, list[str]] = {}
    in_user_info = False
    # Blank lines sit between the last marker and the user-info header; they belong to that
    # section's preamble, but are only recognisable as such once the header arrives.
    pending: list[str] = []

    for line in lines:
        if line == USER_INFO_HEADER:
            in_user_info = True
            user_info_preamble.extend(pending)
            user_info_preamble.append(line)
            pending = []
            continue

        property_match = PROPERTY_PATTERN.match(line)
        if property_match is not None:
            properties.setdefault(int(property_match.group(2)), []).append(property_match.group(3))
            continue

        marker_match = MARKER_PATTERN.match(line)
        if marker_match is not None:
            fields = marker_match.group(2).split(",")
            markers.append(
                Marker(
                    type=fields[0],
                    description=fields[1],
                    position=int(fields[2]),
                    size=int(fields[3]),
                    channel=int(fields[4]),
                    extra=tuple(fields[5:]),
                )
            )
            continue

        if in_user_info:
            user_info_preamble.append(line)
        elif not markers:
            preamble.append(line)
        else:
            pending.append(line)

    numbered = [
        replace(marker, properties=tuple(properties.get(number, ())))
        for number, marker in enumerate(markers, start=1)
    ]
    return MarkerFile(preamble=preamble, markers=numbered, user_info_preamble=user_info_preamble)


def write_marker_file(path: Path | str, marker_file: MarkerFile) -> Path:
    """Write a .vmrk with markers renumbered from 1 and properties remapped to match.

    Properties are re-emitted against each marker's new number, which is what keeps the
    `[Marker User Infos]` references valid after an insertion shifts everything below it.
    """
    lines = list(marker_file.preamble)
    lines += [marker.render(number) for number, marker in enumerate(marker_file.markers, 1)]

    if marker_file.user_info_preamble:
        lines += marker_file.user_info_preamble
        counter = 1
        for number, marker in enumerate(marker_file.markers, start=1):
            for payload in marker.properties:
                lines.append(f"Prop{counter}=Mk{number},{payload}")
                counter += 1

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((LINE_ENDING.join(lines) + LINE_ENDING).encode("utf-8"))
    return destination


def add_pulse_markers(
    marker_file: MarkerFile,
    beat_seconds: np.ndarray,
    sfreq: float,
    *,
    channel: int | None = None,
    custom_marker: bool = True,
) -> MarkerFile:
    """Return the marker set with one `Pulse Artifact/R` added per supplied beat.

    Positions are 1-based sample indices: MNE reads onset `(position - 1) / sfreq`, which
    this inverts exactly. Sorting is stable, so markers already sharing a position keep
    their original order. Recovered markers carry the same `BrainVision.CustomMarker`
    property Analyzer puts on its own R peaks.
    """
    if channel is None:
        channel = marker_file.pulse_channel

    positions = np.round(np.asarray(beat_seconds, dtype=float) * sfreq).astype(int) + 1
    if positions.size and positions.min() < 1:
        raise ValueError(
            f"beat at sample {positions.min()} falls before the first sample of the recording"
        )

    payload = (CUSTOM_MARKER_PROPERTY,) if custom_marker else ()
    added = [
        Marker(PULSE_TYPE, PULSE_DESCRIPTION, int(position), 1, channel, properties=payload)
        for position in positions
    ]
    combined = sorted(marker_file.markers + added, key=lambda marker: marker.position)
    return MarkerFile(
        preamble=marker_file.preamble,
        markers=combined,
        user_info_preamble=marker_file.user_info_preamble,
    )


def remove_pulse_markers(
    marker_file: MarkerFile,
    beat_seconds: np.ndarray,
    sfreq: float,
) -> MarkerFile:
    """Return the marker set with the `Pulse Artifact/R` mark at each supplied beat gone.

    `drop_double_marks` clears Analyzer's second mark inside a cardiac cycle before the gap
    search runs, but the written marker file was only ever added to, so it kept every one of
    them and carried a different beat train than the recovery reported. Step 3 re-runs the
    correction from this file, and a mark where no beat is makes it subtract a pulse
    template against nothing -- injecting artifact rather than removing it.

    A beat with no marker at its position is an error rather than a no-op: it means the
    train being reconciled against did not come from this file.
    """
    positions = np.round(np.asarray(beat_seconds, dtype=float) * sfreq).astype(int) + 1
    wanted = {int(position) for position in positions}

    kept: list[Marker] = []
    removed: set[int] = set()
    for marker in marker_file.markers:
        is_pulse = marker.type == PULSE_TYPE and marker.description == PULSE_DESCRIPTION
        if is_pulse and marker.position in wanted and marker.position not in removed:
            removed.add(marker.position)
            continue
        kept.append(marker)

    missing = sorted(wanted - removed)
    if missing:
        raise ValueError(f"no R marker at sample {missing[0]} to remove")

    return MarkerFile(
        preamble=marker_file.preamble,
        markers=kept,
        user_info_preamble=marker_file.user_info_preamble,
    )
