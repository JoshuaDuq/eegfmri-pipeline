from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import eeg_pipeline.spectral_availability.decomb as decomb_adapter
from eeg_pipeline.spectral_availability.decomb import (
    DecombManifest,
    load_decomb_manifest,
)
from eeg_pipeline.spectral_availability.model import (
    FrequencyInterval,
    RecordingExclusions,
    RecordingKey,
)


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


def test_checksum_and_exclusions_use_one_immutable_byte_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, original_bytes = _write_manifest(
        tmp_path,
        [_finite_row(low="59", high="61"), _terminal_row()],
    )
    _, replacement_bytes = _write_manifest(
        tmp_path,
        [_finite_row(low="70", high="71"), _terminal_row()],
    )
    manifest_path.write_bytes(original_bytes)
    original_read_bytes = Path.read_bytes

    def read_then_replace(path):
        snapshot = original_read_bytes(path)
        if path == manifest_path:
            manifest_path.write_bytes(replacement_bytes)
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)

    manifest = load_decomb_manifest(manifest_path)

    assert manifest.sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert manifest.exclusions[0].intervals == (FrequencyInterval(59.0, 61.0),)


def test_manifest_coerces_path_and_exclusions_to_immutable_values() -> None:
    exclusion = RecordingExclusions(
        key=RecordingKey(subject="0000", task="thermalactive", run="1"),
        intervals=(),
    )

    manifest = DecombManifest(
        path="manifest.tsv",
        sha256="0" * 64,
        exclusions=[exclusion],
    )

    assert manifest.path == Path("manifest.tsv")
    assert manifest.exclusions == (exclusion,)


@pytest.mark.parametrize(
    "sha256",
    ["0" * 63, "0" * 65, "A" * 64, "g" * 64],
)
def test_manifest_rejects_invalid_sha256_text(sha256) -> None:
    with pytest.raises(ValueError, match="sha256"):
        DecombManifest(path="manifest.tsv", sha256=sha256, exclusions=())


def test_manifest_rejects_non_string_sha256() -> None:
    with pytest.raises(TypeError, match="sha256"):
        DecombManifest(path="manifest.tsv", sha256=None, exclusions=())


def test_manifest_rejects_non_recording_exclusions() -> None:
    with pytest.raises(TypeError, match="RecordingExclusions"):
        DecombManifest(
            path="manifest.tsv",
            sha256="0" * 64,
            exclusions=(object(),),
        )


def test_accepts_plus_in_recording_label_entities(tmp_path) -> None:
    _write_dataset_description(tmp_path)
    recording = "sub-family+control_ses-base+2_task-thermal+active_run-01_eeg"
    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(recording), _terminal_row(recording)],
    )

    manifest = load_decomb_manifest(manifest_path)

    assert manifest.exclusions[0].key == RecordingKey(
        subject="family+control",
        session="base+2",
        task="thermal+active",
        run="1",
    )


def test_rejects_recordings_that_conflict_after_run_canonicalization(tmp_path) -> None:
    _write_dataset_description(tmp_path)
    run_one = "sub-0000_task-thermalactive_run-1_eeg"
    padded_run_one = "sub-0000_task-thermalactive_run-01_eeg"
    manifest_path, _ = _write_manifest(
        tmp_path,
        [
            _finite_row(run_one),
            _terminal_row(run_one),
            _finite_row(padded_run_one),
            _terminal_row(padded_run_one),
        ],
    )

    with pytest.raises(ValueError, match="same BIDS identity"):
        load_decomb_manifest(manifest_path)


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


def test_header_only_manifest_requires_a_data_row(tmp_path) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(tmp_path, [])

    with pytest.raises(ValueError, match="at least one data row"):
        load_decomb_manifest(manifest_path)


@pytest.mark.parametrize(
    ("header", "error_match"),
    [
        ((*REQUIRED_COLUMNS, "recording"), r"duplicate.*recording"),
        ((*REQUIRED_COLUMNS, "evidence", "evidence"), r"duplicate.*evidence"),
        ((*REQUIRED_COLUMNS, ""), "blank header"),
    ],
)
def test_rejects_duplicate_or_blank_raw_header_names(
    tmp_path,
    header,
    error_match,
) -> None:
    _write_dataset_description(tmp_path)
    manifest_path = tmp_path / "line_notch_manifest.tsv"
    manifest_path.write_text("\t".join(header) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_decomb_manifest(manifest_path)


@pytest.mark.parametrize("empty_row", ["", "\t\t\t\t"])
def test_rejects_interior_empty_data_rows(tmp_path, empty_row) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, manifest_bytes = _write_manifest(
        tmp_path,
        [_finite_row(), _terminal_row()],
    )
    lines = manifest_bytes.decode().splitlines()
    manifest_path.write_text(
        "\n".join([lines[0], lines[1], empty_row, lines[2]]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"row 3.*empty"):
        load_decomb_manifest(manifest_path)


def test_pandas_reads_only_required_columns_in_bounded_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(), _terminal_row()],
        extra_columns=("channel", "evidence"),
    )
    read_csv = decomb_adapter.pd.read_csv
    observed_kwargs = {}

    def spy_read_csv(*args, **kwargs):
        observed_kwargs.update(kwargs)
        return read_csv(*args, **kwargs)

    monkeypatch.setattr(decomb_adapter.pd, "read_csv", spy_read_csv)

    load_decomb_manifest(manifest_path)

    assert set(observed_kwargs["usecols"]) == set(REQUIRED_COLUMNS)
    assert observed_kwargs["skip_blank_lines"] is False
    assert observed_kwargs["chunksize"] == 10_000


def test_global_row_numbers_are_preserved_across_chunks(tmp_path, monkeypatch) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(
        tmp_path,
        [
            _finite_row(),
            _finite_row(recording="not-a-recording"),
            _terminal_row(),
        ],
    )
    monkeypatch.setattr(decomb_adapter, "_CHUNK_SIZE", 1)

    with pytest.raises(ValueError, match=r"row 3 recording"):
        load_decomb_manifest(manifest_path)


@pytest.mark.parametrize(
    "recording",
    [
        "sub-0000_task-thermalactive_acq-test_run-1_eeg",
        "sub-0000_task-thermalactive_run-1",
        "sub-0000_run-1_task-thermalactive_eeg",
        "sub-0000_ses-base-line_task-thermalactive_run-1_eeg",
        "sub-0000_task-thermalactive_run-alpha_eeg",
        "sub-0000_task-thermalactive_run-١_eeg",
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


def test_accepts_scanner_harmonics_detected_finite_outcome(tmp_path) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(
        tmp_path,
        [
            _finite_row(outcome="scanner_harmonics_detected"),
            _terminal_row(),
        ],
    )

    manifest = load_decomb_manifest(manifest_path)

    assert [
        (interval.low_hz, interval.high_hz) for interval in manifest.exclusions[0].intervals
    ] == [(59.0, 61.0)]


@pytest.mark.parametrize("outcome", ["", "line_detectd", "artifact_detected"])
def test_rejects_unsupported_finite_outcomes_with_row_context(
    tmp_path,
    outcome,
) -> None:
    _write_dataset_description(tmp_path)
    manifest_path, _ = _write_manifest(
        tmp_path,
        [_finite_row(outcome=outcome), _terminal_row()],
    )

    with pytest.raises(ValueError, match=r"row 2.*recording.*outcome"):
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
