from __future__ import annotations

import json
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.analysis.features.preparation import precompute_data
from eeg_pipeline.pipelines import features as features_pipeline
from eeg_pipeline.spectral_availability import (
    align_decomb_to_epochs,
    load_decomb_manifest,
    welch_half_support,
)
from eeg_pipeline.spectral_availability.audit import SpectralAvailabilityAudit

SFREQ = 200.0
N_TIMES = 512
SUBJECT = "0001"
TASK = "thermalactive"

MANIFEST_COLUMNS = (
    "recording",
    "unavailable_low_hz",
    "unavailable_high_hz",
    "outcome",
    "removal_round",
)


class _Config:
    def __init__(self, values: dict) -> None:
        self._values = values

    def get(self, key: str, default=None):
        return self._values.get(key, default)


def _write_decomb_derivative(directory: Path) -> Path:
    (directory / "dataset_description.json").write_text(
        json.dumps({"Name": "Decomb", "GeneratedBy": [{"Name": "decomb", "Version": "1.0"}]}),
        encoding="utf-8",
    )

    run_one = f"sub-{SUBJECT}_task-{TASK}_run-1_eeg"
    run_two = f"sub-{SUBJECT}_task-{TASK}_run-2_eeg"
    rows = [
        # Duplicate evidence and multiple removal rounds must collapse to one geometry.
        (run_one, "59", "61", "line_detected", "1"),
        (run_one, "59", "61", "line_detected", "1"),
        (run_one, "60.5", "62", "line_detected", "2"),
        (run_one, "", "", "no_line_detected", ""),
        (run_two, "9", "11", "line_detected", "1"),
        (run_two, "", "", "no_line_detected", ""),
    ]

    lines = ["\t".join(MANIFEST_COLUMNS)]
    lines.extend("\t".join(row) for row in rows)
    path = directory / "line_notch_manifest.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _two_run_epochs(n_per_run: int = 3) -> mne.EpochsArray:
    generator = np.random.default_rng(5)
    info = mne.create_info(["C3", "C4"], SFREQ, ch_types="eeg")
    data = generator.normal(size=(2 * n_per_run, 2, N_TIMES))
    return mne.EpochsArray(data, info, verbose=False)


def _two_run_events(n_per_run: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"run_id": [1.0] * n_per_run + [2.0] * n_per_run})


@pytest.fixture
def aligned(tmp_path: Path):
    manifest_path = _write_decomb_derivative(tmp_path)
    manifest = load_decomb_manifest(manifest_path)
    availability = align_decomb_to_epochs(
        manifest,
        subject=SUBJECT,
        task=TASK,
        events=_two_run_events(),
    )
    return manifest, availability


def test_two_run_alignment_resolves_each_recording_separately(aligned) -> None:
    _manifest, availability = aligned

    assert [key.run for key in availability.recording_keys] == ["1", "1", "1", "2", "2", "2"]
    assert availability.exclusions_by_epoch[0] == availability.exclusions_by_epoch[1]
    # Rounds 1 and 2 overlap, so 59-61 and 60.5-62 merge into one interval.
    first_run = availability.exclusions_by_epoch[0]
    assert len(first_run) == 1
    assert (first_run[0].low_hz, first_run[0].high_hz) == (59.0, 62.0)
    assert availability.contiguous_band_eligible(8.0, 13.0).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_shared_intermediates_apply_run_specific_geometry(aligned) -> None:
    _manifest, availability = aligned
    config = _Config(
        {
            "feature_engineering.task_is_rest": True,
            "preprocessing.task_is_rest": True,
        }
    )

    precomputed = precompute_data(
        _two_run_epochs(),
        ["alpha"],
        config,
        logging.getLogger("spectral-availability-integration"),
        frequency_bands_override={"alpha": [8.0, 13.0]},
        spectral_availability=availability,
    )

    band = precomputed.band_data["alpha"]
    assert band.eligible_epochs.tolist() == [True, True, True, False, False, False]
    assert np.all(np.isfinite(band.envelope[:3]))
    assert np.all(np.isnan(band.envelope[3:]))

    psd = precomputed.psd_data
    freqs = psd.freqs
    n_fft = min(N_TIMES, int(2.0 * SFREQ))
    expected = availability.valid_frequency_mask(freqs, welch_half_support(SFREQ, n_fft, "hann"))
    assert np.array_equal(psd.valid_frequency_mask, expected)

    # Run 1 loses the mains cluster; run 2 loses the alpha band instead.
    assert not psd.valid_frequency_mask[0][np.argmin(np.abs(freqs - 60.0))]
    assert psd.valid_frequency_mask[3][np.argmin(np.abs(freqs - 60.0))]
    assert psd.valid_frequency_mask[0][np.argmin(np.abs(freqs - 10.0))]
    assert not psd.valid_frequency_mask[3][np.argmin(np.abs(freqs - 10.0))]


def test_audit_names_and_describes_every_recording_and_target(aligned, tmp_path: Path) -> None:
    manifest, availability = aligned
    config = _Config(
        {
            "feature_engineering.task_is_rest": True,
            "preprocessing.task_is_rest": True,
        }
    )
    precomputed = precompute_data(
        _two_run_epochs(),
        ["alpha"],
        config,
        logging.getLogger("spectral-availability-integration"),
        frequency_bands_override={"alpha": [8.0, 13.0]},
        spectral_availability=availability,
    )

    audit = SpectralAvailabilityAudit(
        availability,
        manifest.sha256,
        subject=SUBJECT,
        task=TASK,
    )
    features_pipeline._register_availability_targets(audit, precomputed)
    out_dir = tmp_path / "features"
    path = audit.write(out_dir)

    assert path.name == "sub-0001_task-thermalactive_desc-spectralavailability.tsv"

    table = pd.read_csv(path, sep="\t")
    assert set(table["analysis_target"]) == {"psd_welch", "band_alpha"}
    assert len(table) == 4
    assert set(table["manifest_sha256"]) == {manifest.sha256}

    alpha = table[table["analysis_target"] == "band_alpha"].set_index("run")
    assert bool(alpha.loc[1, "contiguous_band_eligible"]) is True
    assert bool(alpha.loc[2, "contiguous_band_eligible"]) is False
    assert alpha.loc[2, "n_epochs_eligible"] == 0
    assert alpha.loc[2, "n_epochs_ineligible"] == 3
    assert alpha.loc[2, "retained_bandwidth_hz"] == pytest.approx(3.0)


def test_pipeline_does_not_load_decomb_without_a_configured_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args, **_kwargs):
        raise AssertionError("the Decomb adapter must not be reached without a manifest")

    monkeypatch.setattr(
        "eeg_pipeline.spectral_availability.decomb.load_decomb_manifest",
        _fail,
    )

    pipeline = features_pipeline.FeaturePipeline.__new__(features_pipeline.FeaturePipeline)
    pipeline.config = _Config({})
    pipeline.logger = logging.getLogger("spectral-availability-inactive")
    pipeline._decomb_manifest = None

    assert pipeline._load_spectral_availability(SUBJECT, TASK, _two_run_events()) == (None, None)


def test_pipeline_rejects_a_manifest_configured_with_a_downstream_notch(tmp_path: Path) -> None:
    manifest_path = _write_decomb_derivative(tmp_path)

    pipeline = features_pipeline.FeaturePipeline.__new__(features_pipeline.FeaturePipeline)
    pipeline.config = _Config(
        {
            "paths.decomb_manifest": str(manifest_path),
            "preprocessing.notch_freq": 60,
        }
    )
    pipeline.logger = logging.getLogger("spectral-availability-notch")
    pipeline._decomb_manifest = None

    with pytest.raises(ValueError, match="notch_freq"):
        pipeline._load_spectral_availability(SUBJECT, TASK, _two_run_events())


def test_pipeline_builds_availability_and_audit_from_a_configured_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _write_decomb_derivative(tmp_path)

    pipeline = features_pipeline.FeaturePipeline.__new__(features_pipeline.FeaturePipeline)
    pipeline.config = _Config({"paths.decomb_manifest": str(manifest_path)})
    pipeline.logger = logging.getLogger("spectral-availability-active")
    pipeline._decomb_manifest = None

    availability, audit = pipeline._load_spectral_availability(
        SUBJECT,
        TASK,
        _two_run_events(),
    )

    assert [key.run for key in availability.recording_keys] == ["1", "1", "1", "2", "2", "2"]
    assert audit.filename() == "sub-0001_task-thermalactive_desc-spectralavailability.tsv"

    cached = pipeline._decomb_manifest
    pipeline._load_spectral_availability(SUBJECT, TASK, _two_run_events())
    assert pipeline._decomb_manifest is cached
