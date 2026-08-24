from __future__ import annotations

from eeg_pipeline.preprocessing.pipeline.io import (
    read_channels_tsv,
    write_channels_tsv,
)


def test_channels_tsv_round_trip_preserves_bids_missing_values(tmp_path) -> None:
    path = tmp_path / "sub-0001_task-test_channels.tsv"
    original = (
        "name\ttype\tunits\tstatus\tstatus_description\n"
        "Cz\tEEG\tuV\tgood\tn/a\n"
        "ECG\tECG\tn/a\tgood\tn/a\n"
    )
    path.write_text(original, encoding="utf-8")

    channels = read_channels_tsv(path)
    write_channels_tsv(channels, path)

    assert path.read_text(encoding="utf-8") == original
