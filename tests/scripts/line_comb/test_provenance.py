"""A cleaned dataset must say what cleaned it.

mirror_sidecars copies every sidecar byte-for-byte, dataset_description.json included, so
the cleaned root declared DatasetType "raw" and credited only MNE-BIDS -- no line-comb
settings, no code revision, no link back to the root it came from. BIDS asks derivatives to
carry GeneratedBy for exactly this reason: without it the delivered data cannot be traced
to the transformation that produced it.
"""

from __future__ import annotations

import json

import pytest

from studies.pain_study.scripts.line_comb import remove as rlc


def test_the_cleaned_dataset_declares_what_made_it(tmp_path):
    source = tmp_path / "raw"
    (source / "sub-0001" / "eeg").mkdir(parents=True)
    (source / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "study",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
                "GeneratedBy": [{"Name": "mne-bids"}],
            }
        )
    )
    output = tmp_path / "cleaned"
    output.mkdir()
    rlc.mirror_sidecars(source, output)

    settings = rlc.RemovalSettings()
    rlc.write_derivative_description(output, source, settings, source_version="source-digest")

    described = json.loads((output / "dataset_description.json").read_text())
    assert described["DatasetType"] == "derivative"
    names = [entry.get("Name") for entry in described["GeneratedBy"]]
    assert any("line-comb" in str(name) for name in names), names

    generated = next(e for e in described["GeneratedBy"] if "line-comb" in str(e.get("Name")))
    assert generated.get("Version"), "no code revision recorded"
    assert generated["Parameters"]["settings_fingerprint"] == rlc.settings_fingerprint(settings)
    assert described["SourceDatasets"], "no link back to the root this was made from"
    assert described["SourceDatasets"][0]["Version"] == "source-digest"


def test_the_raw_description_is_not_left_in_place(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "dataset_description.json").write_text(json.dumps({"DatasetType": "raw"}))
    output = tmp_path / "cleaned"
    output.mkdir()
    rlc.mirror_sidecars(source, output)
    before = (output / "dataset_description.json").read_text()

    rlc.write_derivative_description(
        output,
        source,
        rlc.RemovalSettings(),
        source_version="source-digest",
    )
    assert (output / "dataset_description.json").read_text() != before


def test_malformed_source_description_surfaces_instead_of_being_replaced(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "dataset_description.json").write_text("not-json", encoding="utf-8")
    output = tmp_path / "cleaned"
    output.mkdir()
    rlc.mirror_sidecars(source, output)

    with pytest.raises(json.JSONDecodeError):
        rlc.write_derivative_description(
            output,
            source,
            rlc.RemovalSettings(),
            source_version="source-digest",
        )


def test_each_manifest_row_identifies_its_input_plan_and_settings():
    row = rlc.record_manifest_provenance(
        {"recording": "sub-0001_run-1"},
        input_digest="input-digest",
        plan_digest="plan-digest",
        fingerprint="settings-fingerprint",
    )

    assert row["input_digest"] == "input-digest"
    assert row["plan_digest"] == "plan-digest"
    assert row["settings_fingerprint"] == "settings-fingerprint"
