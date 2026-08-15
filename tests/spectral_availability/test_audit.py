from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.spectral_availability import (
    EpochSpectralAvailability,
    FrequencyInterval,
    RecordingKey,
)
from eeg_pipeline.spectral_availability.audit import SpectralAvailabilityAudit

CHECKSUM = "0" * 64
FREQS = np.array([8.0, 9.0, 10.0, 11.0, 12.0])


def _availability() -> EpochSpectralAvailability:
    first = RecordingKey(subject="0001", task="thermalactive", run="1")
    second = RecordingKey(subject="0001", task="thermalactive", run="2")
    return EpochSpectralAvailability(
        recording_keys=(first, first, second, second),
        exclusions_by_epoch=(
            (),
            (),
            (FrequencyInterval(9.5, 10.5),),
            (FrequencyInterval(9.5, 10.5),),
        ),
    )


def _audit() -> SpectralAvailabilityAudit:
    return SpectralAvailabilityAudit(
        _availability(),
        CHECKSUM,
        subject="0001",
        task="thermalactive",
    )


def _grid_mask(availability: EpochSpectralAvailability) -> np.ndarray:
    return availability.valid_frequency_mask(FREQS, 0.0)


def test_grid_registration_writes_one_row_per_recording() -> None:
    audit = _audit()
    availability = _availability()

    audit.register_grid(
        "psd_full",
        FREQS,
        _grid_mask(availability),
        estimator="welch",
        support_rule="half-power main lobe",
    )

    rows = audit.rows()
    assert [row.run for row in rows] == ["1", "2"]

    intact, notched = rows
    assert intact.unavailable_intervals_hz == "none"
    assert intact.retained_share == pytest.approx(1.0)
    assert intact.n_epochs_aligned == 2
    assert intact.n_epochs_eligible == 2
    assert intact.n_epochs_ineligible == 0
    assert intact.manifest_sha256 == CHECKSUM

    assert notched.unavailable_intervals_hz == "9.5-10.5"
    # The 10 Hz bin carries a weight of 1.0 out of the grid's nominal 4.0 Hz.
    assert notched.nominal_bandwidth_hz == pytest.approx(4.0)
    assert notched.retained_bandwidth_hz == pytest.approx(3.0)
    assert notched.retained_share == pytest.approx(0.75)
    assert notched.psd_eligible is True
    assert notched.contiguous_band_eligible is False


def test_band_registration_reports_retained_width_and_eligibility() -> None:
    audit = _audit()

    audit.register_band(
        "alpha_hilbert",
        8.0,
        12.0,
        estimator="hilbert",
        support_rule="contiguous passband",
    )

    intact, notched = audit.rows()
    assert intact.contiguous_band_eligible is True
    assert intact.n_epochs_eligible == 2
    assert notched.contiguous_band_eligible is False
    assert notched.n_epochs_eligible == 0
    assert notched.n_epochs_ineligible == 2
    assert notched.retained_bandwidth_hz == pytest.approx(3.0)
    assert notched.retained_share == pytest.approx(0.75)


def test_identical_registration_is_idempotent() -> None:
    audit = _audit()
    availability = _availability()

    for _ in range(2):
        audit.register_grid(
            "psd_full",
            FREQS,
            _grid_mask(availability),
            estimator="welch",
            support_rule="half-power main lobe",
        )

    assert len(audit.rows()) == 2


def test_conflicting_registration_for_the_same_target_raises() -> None:
    audit = _audit()
    availability = _availability()

    audit.register_grid(
        "psd_full",
        FREQS,
        _grid_mask(availability),
        estimator="welch",
        support_rule="half-power main lobe",
    )

    with pytest.raises(ValueError, match="Conflicting"):
        audit.register_grid(
            "psd_full",
            FREQS,
            _grid_mask(availability),
            estimator="multitaper",
            support_rule="bandwidth / 2",
        )


def test_rows_are_sorted_deterministically() -> None:
    audit = _audit()

    audit.register_band("theta", 4.0, 7.0, estimator="hilbert", support_rule="contiguous passband")
    audit.register_band("alpha", 8.0, 12.0, estimator="hilbert", support_rule="contiguous passband")

    assert [(row.run, row.analysis_target) for row in audit.rows()] == [
        ("1", "alpha"),
        ("1", "theta"),
        ("2", "alpha"),
        ("2", "theta"),
    ]


def test_grid_registration_rejects_a_mask_with_the_wrong_shape() -> None:
    audit = _audit()

    with pytest.raises(ValueError, match="valid_frequency_mask"):
        audit.register_grid(
            "psd_full",
            FREQS,
            np.ones((3, FREQS.size), dtype=bool),
            estimator="welch",
            support_rule="half-power main lobe",
        )


def test_write_replaces_atomically_without_leaving_a_temporary_file(tmp_path: Path) -> None:
    audit = _audit()
    audit.register_band("alpha", 8.0, 12.0, estimator="hilbert", support_rule="contiguous passband")

    path = audit.write(tmp_path)

    assert path.name == "sub-0001_task-thermalactive_desc-spectralavailability.tsv"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [path.name]

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t")[:5] == [
        "subject",
        "session",
        "task",
        "run",
        "analysis_target",
    ]
    assert len(lines) == 3

    audit.write(tmp_path)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [path.name]
