"""Where an audit should learn which frequencies the removal acted on.

Two scripts have now drifted from the removal by keeping their own copy of the line list,
and detection makes a third copy impossible to keep correct by hand: the lines are resolved
per session, so no static list can name them. The manifest the removal writes is the only
thing that knows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from studies.pain_study.analysis.line_comb import removal as lr


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "removal_manifest.tsv"
    pd.DataFrame(
        [
            {"recording": "sub-0008_run-1", "isolated_hz": "27.981;47.185;94.370"},
            {"recording": "sub-0008_run-2", "isolated_hz": "27.985;47.180;94.365"},
            {"recording": "sub-0001_run-1", "isolated_hz": "57.315;58.167"},
        ]
    ).to_csv(path, sep="\t", index=False)
    return path


def test_the_lines_come_from_what_the_removal_recorded(manifest):
    found = lr.removed_isolated_lines(manifest)
    assert any(abs(f - 94.37) < 0.02 for f in found), found
    assert any(abs(f - 57.315) < 0.02 for f in found), found


def test_positions_of_one_line_across_recordings_collapse(manifest):
    """94.370 and 94.365 are one line seen twice, not two lines."""
    found = [f for f in lr.removed_isolated_lines(manifest) if 94.0 < f < 95.0]
    assert len(found) == 1, found


def test_the_result_is_ordered(manifest):
    found = lr.removed_isolated_lines(manifest)
    assert list(found) == sorted(found)


def test_lines_can_be_resolved_for_one_participant(manifest):
    found = lr.removed_isolated_lines(manifest, subject="sub-0008")

    assert any(abs(frequency - 94.37) < 0.02 for frequency in found)
    assert not any(abs(frequency - 57.315) < 0.02 for frequency in found)


def test_a_subject_missing_from_the_manifest_is_an_error(manifest):
    with pytest.raises(ValueError, match="no rows for sub-9999"):
        lr.removed_isolated_lines(manifest, subject="sub-9999")


def test_a_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        lr.removed_isolated_lines(tmp_path / "absent.tsv")


def test_a_manifest_without_the_column_is_an_error(tmp_path):
    path = tmp_path / "removal_manifest.tsv"
    pd.DataFrame([{"recording": "missing-isolated-column"}]).to_csv(
        path,
        sep="\t",
        index=False,
    )
    with pytest.raises(ValueError, match="no isolated_hz column"):
        lr.removed_isolated_lines(path)


def test_a_malformed_frequency_is_an_error(tmp_path):
    path = tmp_path / "removal_manifest.tsv"
    pd.DataFrame([{"recording": "a", "isolated_hz": "47.0;not-a-frequency"}]).to_csv(
        path,
        sep="\t",
        index=False,
    )
    with pytest.raises(ValueError, match="could not convert string to float"):
        lr.removed_isolated_lines(path)


def test_blank_and_nan_entries_are_skipped(tmp_path):
    """A run where nothing resolved writes an empty cell, which is a measurement not a fault."""
    path = tmp_path / "removal_manifest.tsv"
    pd.DataFrame(
        [
            {"recording": "a", "isolated_hz": "47.0;94.3"},
            {"recording": "b", "isolated_hz": ""},
            {"recording": "c", "isolated_hz": None},
        ]
    ).to_csv(path, sep="\t", index=False)
    found = lr.removed_isolated_lines(path)
    assert len(found) == 2, found
