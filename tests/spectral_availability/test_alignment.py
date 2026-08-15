from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.spectral_availability.alignment import align_decomb_to_epochs
from eeg_pipeline.spectral_availability.decomb import DecombManifest
from eeg_pipeline.spectral_availability.model import (
    FrequencyInterval,
    RecordingExclusions,
    RecordingKey,
)


def _exclusion(
    *,
    subject="0001",
    task="thermalactive",
    run="1",
    session=None,
    low_hz=59.0,
):
    return RecordingExclusions(
        key=RecordingKey(
            subject=subject,
            task=task,
            run=run,
            session=session,
        ),
        intervals=(FrequencyInterval(low_hz, low_hz + 1.0),),
    )


def _manifest(*exclusions):
    return DecombManifest(
        path=Path("/data/line_notch_manifest.tsv"),
        sha256="0" * 64,
        exclusions=tuple(exclusions),
    )


def test_aligns_integer_valued_numeric_runs_in_original_event_order() -> None:
    manifest = _manifest(*[_exclusion(run=str(run), low_hz=float(run * 10)) for run in range(1, 7)])
    events = pd.DataFrame({"run_id": [6.0, 1.0, 3.0, 2.0, 5.0, 4.0]})

    aligned = align_decomb_to_epochs(
        manifest,
        subject="0001",
        task="thermalactive",
        events=events,
    )

    assert [key.run for key in aligned.recording_keys] == [
        "6",
        "1",
        "3",
        "2",
        "5",
        "4",
    ]
    assert [intervals[0].low_hz for intervals in aligned.exclusions_by_epoch] == [
        60.0,
        10.0,
        30.0,
        20.0,
        50.0,
        40.0,
    ]


@pytest.mark.parametrize(
    "run_id",
    [9007199254740993, np.int64(9007199254740993)],
    ids=["python-int", "numpy-int64"],
)
def test_preserves_large_integral_run_ids_exactly(run_id) -> None:
    expected_run = "9007199254740993"
    manifest = _manifest(_exclusion(run=expected_run))
    events = pd.DataFrame(
        {"run_id": pd.Series([run_id], dtype=object)},
    )

    aligned = align_decomb_to_epochs(
        manifest,
        subject="0001",
        task="thermalactive",
        events=events,
    )

    assert aligned.recording_keys[0].run == expected_run


def test_accepts_one_matching_prefix_on_explicit_entity_strings() -> None:
    manifest = _manifest(
        _exclusion(run="01", session="baseline2"),
    )

    aligned = align_decomb_to_epochs(
        manifest,
        subject="sub-0001",
        task="task-thermalactive",
        events=pd.DataFrame(
            {
                "run_id": ["run-01"],
                "session_id": ["ses-baseline2"],
            }
        ),
    )

    assert aligned.recording_keys == (
        RecordingKey(
            subject="0001",
            task="thermalactive",
            run="01",
            session="baseline2",
        ),
    )


@pytest.mark.parametrize(
    "run_id",
    [None, pd.NA, True, 1.5, np.inf, -np.inf, np.nan, "1.0", "", "run-run-1"],
)
def test_rejects_missing_noncanonical_or_nonintegral_runs(run_id) -> None:
    manifest = _manifest(_exclusion())

    with pytest.raises((TypeError, ValueError), match=r"row 0.*run_id"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events=pd.DataFrame({"run_id": [run_id]}),
        )


def test_requires_a_dataframe_and_run_id_column() -> None:
    manifest = _manifest(_exclusion())

    with pytest.raises(TypeError, match="DataFrame"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events={"run_id": [1]},
        )

    with pytest.raises(ValueError, match="run_id"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events=pd.DataFrame({"condition": ["pain"]}),
        )


@pytest.mark.parametrize("session_values", [None, [None], [""], ["baseline3"]])
def test_session_recordings_require_exact_usable_session_ids(
    session_values,
) -> None:
    manifest = _manifest(_exclusion(session="baseline2"))
    data = {"run_id": [1]}
    if session_values is not None:
        data["session_id"] = session_values

    with pytest.raises(ValueError, match=r"session_id|unmatched"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events=pd.DataFrame(data),
        )


def test_no_session_manifest_rejects_explicit_event_sessions() -> None:
    manifest = _manifest(_exclusion())

    with pytest.raises(ValueError, match=r"session_id|unmatched"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events=pd.DataFrame(
                {
                    "run_id": [1],
                    "session_id": ["baseline2"],
                }
            ),
        )


def test_rejects_mixed_session_identity_for_selected_recordings() -> None:
    manifest = _manifest(
        _exclusion(run="1"),
        _exclusion(run="2", session="baseline2"),
    )

    with pytest.raises(ValueError, match="mixed|ambiguous"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events=pd.DataFrame(
                {
                    "run_id": [1],
                    "session_id": [None],
                }
            ),
        )


def test_rejects_duplicate_manifest_keys() -> None:
    duplicate = _exclusion()
    manifest = _manifest(duplicate, duplicate)

    with pytest.raises(ValueError, match="duplicate|ambiguous"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events=pd.DataFrame({"run_id": [1]}),
        )


def test_rejects_every_unmatched_epoch() -> None:
    manifest = _manifest(_exclusion(run="1"))

    with pytest.raises(ValueError, match=r"row 1.*unmatched"):
        align_decomb_to_epochs(
            manifest,
            subject="0001",
            task="thermalactive",
            events=pd.DataFrame({"run_id": [1, 2]}),
        )


def test_allows_irrelevant_manifest_recordings() -> None:
    selected = _exclusion(run="2")
    manifest = _manifest(
        _exclusion(subject="9999", run="2"),
        _exclusion(task="rest", run="2"),
        selected,
    )

    aligned = align_decomb_to_epochs(
        manifest,
        subject="0001",
        task="thermalactive",
        events=pd.DataFrame({"run_id": [2]}),
    )

    assert aligned.recording_keys == (selected.key,)
    assert aligned.exclusions_by_epoch == (selected.intervals,)


def test_package_public_api_exports_decomb_ingestion_and_alignment() -> None:
    import eeg_pipeline.spectral_availability as spectral_availability

    assert spectral_availability.DecombManifest is DecombManifest
    assert spectral_availability.align_decomb_to_epochs is align_decomb_to_epochs
