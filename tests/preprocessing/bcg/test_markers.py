import numpy as np
import pytest

from eeg_pipeline.preprocessing.bcg.markers import (
    CUSTOM_MARKER_PROPERTY,
    Marker,
    add_pulse_markers,
    read_marker_file,
    write_marker_file,
)

SFREQ = 1000.0

# Analyzer writes CRLF, UTF-8, markers ordered by position, a New Segment line whose sixth
# field is the acquisition timestamp, and a [Marker User Infos] section that assigns
# properties to markers *by marker number*.
SAMPLE_VMRK = (
    "BrainVision Data Exchange Marker File Version 2.0\r\n"
    "; Data created from history path: Run/Raw Data/Scanner Artifact Correction/"
    "Pulse Artifact Correction (Mark R peaks)\r\n"
    "\r\n"
    "[Common Infos]\r\n"
    "Codepage=UTF-8\r\n"
    "DataFile=run1_sub0009.eeg\r\n"
    "\r\n"
    "[Marker Infos]\r\n"
    "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\r\n"
    "Mk1=New Segment,,1,1,0,20260622104823594618\r\n"
    "Mk2=Pulse Artifact,R,468,1,32\r\n"
    "Mk3=Stimulus,S  1,900,1,0\r\n"
    "Mk4=Pulse Artifact,R,1358,1,32\r\n"
    "\r\n"
    "[Marker User Infos]\r\n"
    "; Properties are assigned to markers using their marker number.\r\n"
    "Prop1=Mk2,bool,BrainVision.CustomMarker,true\r\n"
    "Prop2=Mk4,bool,BrainVision.CustomMarker,true\r\n"
)


def _write(tmp_path, text=SAMPLE_VMRK):
    path = tmp_path / "run1_sub0009.vmrk"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_reading_then_writing_reproduces_the_file_byte_for_byte(tmp_path):
    """Analyzer has to accept the result, so anything not deliberately changed must not."""
    source = _write(tmp_path)

    destination = tmp_path / "out.vmrk"
    write_marker_file(destination, read_marker_file(source))

    assert destination.read_bytes() == source.read_bytes()


def test_markers_are_parsed_with_their_fields_and_properties(tmp_path):
    parsed = read_marker_file(_write(tmp_path))

    assert len(parsed.markers) == 4
    assert parsed.markers[0] == Marker("New Segment", "", 1, 1, 0, ("20260622104823594618",))
    assert parsed.markers[1].position == 468
    assert parsed.markers[1].properties == (CUSTOM_MARKER_PROPERTY,)
    assert parsed.markers[2].description == "S  1"  # internal spacing is significant
    assert parsed.markers[2].properties == ()
    assert parsed.pulse_channel == 32


def test_added_beats_become_r_markers_in_position_order(tmp_path):
    parsed = read_marker_file(_write(tmp_path))

    combined = add_pulse_markers(parsed, np.array([0.7, 2.0]), SFREQ)

    positions = [m.position for m in combined.markers]
    assert positions == sorted(positions)
    assert 701 in positions and 2001 in positions  # 1-based sample index
    added = [m for m in combined.markers if m.position in (701, 2001)]
    assert all(m.type == "Pulse Artifact" and m.description == "R" for m in added)
    assert all(m.channel == 32 for m in added)


def test_recovered_markers_carry_analyzer_s_custom_marker_property(tmp_path):
    """Analyzer flags its own R peaks this way; ours must be the same kind of object."""
    parsed = read_marker_file(_write(tmp_path))

    combined = add_pulse_markers(parsed, np.array([0.7]), SFREQ)

    assert next(m for m in combined.markers if m.position == 701).properties == (
        CUSTOM_MARKER_PROPERTY,
    )


def test_existing_markers_are_all_preserved(tmp_path):
    parsed = read_marker_file(_write(tmp_path))

    combined = add_pulse_markers(parsed, np.array([0.7, 2.0]), SFREQ)

    for original in parsed.markers:
        assert original in combined.markers
    assert len(combined.markers) == len(parsed.markers) + 2


def test_properties_are_remapped_to_the_renumbered_markers(tmp_path):
    """Inserting a marker shifts every number below it; stale references would mislabel.

    The inserted beat at 701 lands between Mk3 and Mk4, so Analyzer's second R marker
    moves from Mk4 to Mk5 and its property must follow it.
    """
    source = _write(tmp_path)
    combined = add_pulse_markers(read_marker_file(source), np.array([0.7]), SFREQ)

    destination = tmp_path / "out.vmrk"
    write_marker_file(destination, combined)
    text = destination.read_bytes().decode("utf-8")

    assert "Mk5=Pulse Artifact,R,1358,1,32\r\n" in text
    assert "Prop3=Mk5,bool,BrainVision.CustomMarker,true\r\n" in text
    # Every property still points at a Pulse Artifact marker, none at the Stimulus.
    numbers = {int(line[2 : line.index("=")]) for line in text.splitlines() if line[:2] == "Mk"}
    referenced = {
        int(line.split("=Mk")[1].split(",")[0])
        for line in text.splitlines()
        if line.startswith("Prop")
    }
    assert referenced <= numbers


def test_written_markers_are_renumbered_contiguously(tmp_path):
    combined = add_pulse_markers(read_marker_file(_write(tmp_path)), np.array([0.7, 2.0]), SFREQ)

    destination = tmp_path / "out.vmrk"
    write_marker_file(destination, combined)

    lines = destination.read_bytes().decode("utf-8").splitlines()
    numbers = [int(line[2 : line.index("=")]) for line in lines if line[:2] == "Mk"]
    assert numbers == list(range(1, len(combined.markers) + 1))


def test_new_segment_keeps_its_timestamp_field(tmp_path):
    combined = add_pulse_markers(read_marker_file(_write(tmp_path)), np.array([0.7]), SFREQ)

    destination = tmp_path / "out.vmrk"
    write_marker_file(destination, combined)

    text = destination.read_bytes().decode("utf-8")
    assert "Mk1=New Segment,,1,1,0,20260622104823594618\r\n" in text


def test_a_beat_outside_the_recording_is_refused(tmp_path):
    parsed = read_marker_file(_write(tmp_path))

    with pytest.raises(ValueError, match="before the first sample"):
        add_pulse_markers(parsed, np.array([-0.5]), SFREQ)
