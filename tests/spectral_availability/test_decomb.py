from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eeg_pipeline.spectral_availability.decomb import load_decomb_manifest
from eeg_pipeline.spectral_availability.model import RecordingKey


REQUIRED_COLUMNS = (
    "recording",
    "unavailable_low_hz",
    "unavailable_high_hz",
    "outcome",
    "removal_round",
)


def test_package_import_keeps_decomb_adapter_lazy() -> None:
    command = (
        "import sys; "
        "import eeg_pipeline.spectral_availability; "
        "assert 'eeg_pipeline.spectral_availability.decomb' not in sys.modules"
    )

    subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


def _write_dataset_description(tmp_path, generated_by=None) -> None:
    if generated_by is None:
        generated_by = [
            {"Name": "MNE-BIDS", "Version": "0.19"},
            {"Name": "DeCoMb", "Version": "1.0"},
        ]
    description = {"Name": "Decomb derivative", "GeneratedBy": generated_by}
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(description),
        encoding="utf-8",
    )


def _write_manifest(tmp_path, rows, *, columns=REQUIRED_COLUMNS, extra_columns=()):
    all_columns = (*columns, *extra_columns)
    lines = ["\t".join(all_columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in all_columns))
    manifest_bytes = ("\n".join(lines) + "\n").encode()
    path = tmp_path / "line_notch_manifest.tsv"
    path.write_bytes(manifest_bytes)
    return path, manifest_bytes


def _finite_row(
    recording="sub-0000_task-thermalactive_run-1_eeg",
    low="59",
    high="61",
    *,
    outcome="line_detected",
    removal_round="1",
):
    return {
        "recording": recording,
        "unavailable_low_hz": low,
        "unavailable_high_hz": high,
        "outcome": outcome,
        "removal_round": removal_round,
    }


def _terminal_row(recording="sub-0000_task-thermalactive_run-1_eeg"):
    return {
        "recording": recording,
        "unavailable_low_hz": "",
        "unavailable_high_hz": "",
        "outcome": "no_line_detected",
        "removal_round": "",
    }


def test_loads_realistic_manifest_and_merges_all_interval_evidence(tmp_path) -> None:
    _write_dataset_description(tmp_path)
    first = "sub-0000_task-thermalactive_run-1_eeg"
    second = "sub-0000_ses-baseline2_task-thermalactive_run-2_eeg"
    extra_columns = tuple(f"evidence_{index:02d}" for index in range(40))
    rows = [
        _finite_row(first, "59", "61", removal_round="1"),
        _finite_row(first, "59", "61", removal_round="1"),
        _finite_row(first, "61", "62", removal_round="2"),
        _finite_row(first, "70", "71", removal_round="3"),
        _terminal_row(first),
        _finite_row(second, "39.5", "40.5", removal_round="1"),
        _terminal_row(second),
    ]
    path, manifest_bytes = _write_manifest(
        tmp_path,
        rows,
        extra_columns=extra_columns,
    )

    manifest = load_decomb_manifest(path)

    assert manifest.path == path
    assert manifest.sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert [exclusion.key for exclusion in manifest.exclusions] == [
        RecordingKey(subject="0000", task="thermalactive", run="1"),
        RecordingKey(
            subject="0000",
            session="baseline2",
            task="thermalactive",
            run="2",
        ),
    ]
    assert [
        (interval.low_hz, interval.high_hz) for interval in manifest.exclusions[0].intervals
    ] == [(59.0, 62.0), (70.0, 71.0)]
    assert [
        (interval.low_hz, interval.high_hz) for interval in manifest.exclusions[1].intervals
    ] == [(39.5, 40.5)]


def test_manifest_and_dataset_description_must_be_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest.*file"):
        load_decomb_manifest(tmp_path / "missing.tsv")

    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(), _terminal_row()],
    )
    with pytest.raises(FileNotFoundError, match="dataset_description.json.*file"):
        load_decomb_manifest(manifest_path)


@pytest.mark.parametrize(
    "generated_by",
    [
        [],
        [{"Name": "MNE-BIDS"}],
        [{"Name": "decomb"}, {"Name": "DECOMB"}],
        "decomb",
        ["decomb"],
        [{"Version": "1.0"}, {"Name": "decomb"}],
        [{"Name": 12}, {"Name": "decomb"}],
    ],
)
def test_rejects_missing_duplicate_or_malformed_decomb_provenance(
    tmp_path,
    generated_by,
) -> None:
    _write_dataset_description(tmp_path, generated_by)
    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(), _terminal_row()],
    )

    with pytest.raises(ValueError, match="GeneratedBy"):
        load_decomb_manifest(manifest_path)


def test_rejects_missing_required_columns_with_names(tmp_path) -> None:
    _write_dataset_description(tmp_path)
    columns = tuple(column for column in REQUIRED_COLUMNS if column != "removal_round")
    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(), _terminal_row()],
        columns=columns,
    )

    with pytest.raises(ValueError, match="removal_round"):
        load_decomb_manifest(manifest_path)


def test_empty_manifest_names_all_missing_required_columns(tmp_path) -> None:
    _write_dataset_description(tmp_path)
    manifest_path = tmp_path / "line_notch_manifest.tsv"
    manifest_path.write_bytes(b"")

    with pytest.raises(ValueError) as exc_info:
        load_decomb_manifest(manifest_path)

    assert all(column in str(exc_info.value) for column in REQUIRED_COLUMNS)


@pytest.mark.parametrize(
    "recording",
    [
        "sub-0000_task-thermalactive_acq-test_run-1_eeg",
        "sub-0000_task-thermalactive_run-1",
        "sub-0000_run-1_task-thermalactive_eeg",
        "sub-0000_ses-base-line_task-thermalactive_run-1_eeg",
    ],
)
def test_rejects_unrepresentable_or_malformed_recording_identity(
    tmp_path,
    recording,
) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(recording), _terminal_row(recording)],
    )

    with pytest.raises(ValueError, match=r"row 2.*recording"):
        load_decomb_manifest(manifest_path)


@pytest.mark.parametrize(
    "rows",
    [
        [_finite_row()],
        [_finite_row(), _terminal_row(), _terminal_row()],
        [
            _finite_row(),
            {
                **_terminal_row(),
                "outcome": "line_detected",
            },
        ],
        [_finite_row(low="", high="61"), _terminal_row()],
        [
            _finite_row(outcome="no_line_detected"),
            _terminal_row(),
        ],
    ],
)
def test_rejects_invalid_terminal_or_interval_blank_rows(tmp_path, rows) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(tmp_path, rows)

    with pytest.raises(ValueError, match=r"recording|row"):
        load_decomb_manifest(manifest_path)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ("not-a-number", "61"),
        ("nan", "61"),
        ("59", "inf"),
        ("-1", "2"),
        ("61", "61"),
        ("62", "61"),
    ],
)
def test_rejects_malformed_or_invalid_interval_geometry_with_row_context(
    tmp_path,
    low,
    high,
) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(low=low, high=high), _terminal_row()],
    )

    with pytest.raises(ValueError, match=r"row 2.*recording"):
        load_decomb_manifest(manifest_path)
