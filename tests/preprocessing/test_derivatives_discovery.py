"""Derivative discovery must work on every BIDS layout, not just this project's.

Each of these layouts previously matched nothing rather than raising, so a stage would
report "no runs found" — or silently do nothing — on a perfectly valid dataset.
"""

from __future__ import annotations

import pytest

from eeg_pipeline.preprocessing.derivatives import (
    FILTERED_RAW_SUFFIX,
    clean_raw_for_filtered,
    entity_prefix,
    find_filtered_raw_runs,
    find_ica_solutions,
    resolve_subjects,
    runs_for_prefix,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_finds_runs_in_a_session_organized_dataset(tmp_path) -> None:
    eeg = tmp_path / "sub-01" / "ses-02" / "eeg"
    run = _touch(eeg / f"sub-01_ses-02_task-pain_run-1{FILTERED_RAW_SUFFIX}")

    assert find_filtered_raw_runs(tmp_path, subject="01", task="pain") == [run]


def test_finds_a_run_with_no_run_entity(tmp_path) -> None:
    eeg = tmp_path / "sub-01" / "eeg"
    run = _touch(eeg / f"sub-01_task-pain{FILTERED_RAW_SUFFIX}")

    assert find_filtered_raw_runs(tmp_path, subject="01", task="pain") == [run]


def test_ignores_other_tasks_and_appledouble_files(tmp_path) -> None:
    eeg = tmp_path / "sub-01" / "eeg"
    wanted = _touch(eeg / f"sub-01_task-pain_run-1{FILTERED_RAW_SUFFIX}")
    _touch(eeg / f"sub-01_task-rest_run-1{FILTERED_RAW_SUFFIX}")
    _touch(eeg / f"._sub-01_task-pain_run-2{FILTERED_RAW_SUFFIX}")

    assert find_filtered_raw_runs(tmp_path, subject="01", task="pain") == [wanted]


def test_does_not_match_a_different_subject_sharing_a_prefix(tmp_path) -> None:
    """sub-01 must not pick up sub-011's runs."""
    _touch(tmp_path / "sub-01" / "eeg" / f"sub-011_task-pain_run-1{FILTERED_RAW_SUFFIX}")

    assert find_filtered_raw_runs(tmp_path, subject="01", task="pain") == []


def test_split_recordings_are_rejected_rather_than_half_processed(tmp_path) -> None:
    eeg = tmp_path / "sub-01" / "eeg"
    _touch(eeg / f"sub-01_task-pain_split-01_run-1{FILTERED_RAW_SUFFIX}")

    with pytest.raises(RuntimeError, match="Split"):
        find_filtered_raw_runs(tmp_path, subject="01", task="pain")


def test_a_missing_subject_directory_is_empty_not_an_error(tmp_path) -> None:
    assert find_filtered_raw_runs(tmp_path, subject="99", task="pain") == []
    assert find_ica_solutions(tmp_path, subject="99") == []


def test_runs_are_grouped_under_the_session_decomposition_they_belong_to(tmp_path) -> None:
    ica_one = _touch(tmp_path / "sub-01" / "ses-01" / "eeg" / "sub-01_ses-01_proc-ica_ica.fif")
    ica_two = _touch(tmp_path / "sub-01" / "ses-02" / "eeg" / "sub-01_ses-02_proc-ica_ica.fif")
    run_one = _touch(
        tmp_path / "sub-01" / "ses-01" / "eeg" / f"sub-01_ses-01_task-pain{FILTERED_RAW_SUFFIX}"
    )
    run_two = _touch(
        tmp_path / "sub-01" / "ses-02" / "eeg" / f"sub-01_ses-02_task-pain{FILTERED_RAW_SUFFIX}"
    )

    assert find_ica_solutions(tmp_path, subject="01") == [ica_one, ica_two]
    runs = find_filtered_raw_runs(tmp_path, subject="01", task="pain")
    assert runs_for_prefix(runs, "sub-01_ses-01") == [run_one]
    assert runs_for_prefix(runs, "sub-01_ses-02") == [run_two]


def test_clean_counterpart_must_exist(tmp_path) -> None:
    filtered = _touch(tmp_path / f"sub-01_task-pain_run-1{FILTERED_RAW_SUFFIX}")

    with pytest.raises(FileNotFoundError, match="ICA-cleaned"):
        clean_raw_for_filtered(filtered)

    clean = _touch(tmp_path / "sub-01_task-pain_run-1_proc-clean_raw.fif")
    assert clean_raw_for_filtered(filtered) == clean


def test_resolve_subjects_discovers_all_or_normalizes_explicit(tmp_path) -> None:
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-02").mkdir()

    assert resolve_subjects(tmp_path, ["all"]) == ["01", "02"]
    assert resolve_subjects(tmp_path, ["sub-07", "08"]) == ["07", "08"]
    with pytest.raises(FileNotFoundError):
        resolve_subjects(tmp_path / "empty", ["all"])


def test_entity_prefix_rejects_a_mismatched_suffix(tmp_path) -> None:
    with pytest.raises(ValueError, match="ending in"):
        entity_prefix(tmp_path / "sub-01_task-pain_epo.fif", FILTERED_RAW_SUFFIX)
