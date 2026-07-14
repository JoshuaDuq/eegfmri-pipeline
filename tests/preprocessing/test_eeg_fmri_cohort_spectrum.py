from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.preprocessing.eeg_fmri.cohort_spectrum import (
    aggregate_cohort_scanner_spectra,
    extract_run_scanner_spectra,
    write_cohort_scanner_spectra_tsv,
)
from eeg_pipeline.preprocessing.eeg_fmri.qc import HarmonicSpectrum, HarmonicStageQc


def _harmonic_stages(
    *,
    offset_db: float,
    frequencies_hz: np.ndarray | None = None,
) -> dict[str, HarmonicStageQc]:
    frequencies = (
        np.arange(10.0, 101.0, 1.0) if frequencies_hz is None else frequencies_hz
    )
    return {
        stage: HarmonicStageQc(
            summary={"n_channels": 12},
            spectrum=HarmonicSpectrum(
                frequencies_hz=frequencies,
                median_power_db=np.full(frequencies.shape, offset_db + stage_offset),
            ),
        )
        for stage, stage_offset in (
            ("raw", 0.0),
            ("gradient_corrected", -20.0),
            ("final", -25.0),
        )
    }


def _run(subject: str, run: int, offset_db: float):
    return extract_run_scanner_spectra(
        subject=subject,
        run=run,
        harmonic_stages=_harmonic_stages(offset_db=offset_db),
    )


def test_extract_run_scanner_spectra_keeps_only_common_qc_range() -> None:
    run = _run("sub-0001", 1, 0.0)

    assert run.subject == "sub-0001"
    assert run.run == 1
    assert run.channel_count == 12
    assert run.raw.frequencies_hz[[0, -1]] == pytest.approx([15.0, 90.0])
    assert np.array_equal(run.raw.frequencies_hz, run.final.frequencies_hz)
    assert not run.raw.frequencies_hz.flags.writeable
    assert not run.final.median_power_db.flags.writeable


def test_cohort_scanner_spectra_weight_participants_equally() -> None:
    runs = (
        _run("sub-0001", 1, 0.0),
        _run("sub-0001", 2, 0.0),
        _run("sub-0001", 3, 0.0),
        _run("sub-0002", 1, 100.0),
    )

    cohort = aggregate_cohort_scanner_spectra(
        runs,
        bootstrap_iterations=200,
        confidence_level=0.95,
        bootstrap_seed=42,
    )

    assert cohort.participant_count == 2
    assert cohort.run_count == 4
    assert cohort.channel_count == 12
    assert cohort.raw.median_power_db == pytest.approx(50.0)
    assert cohort.gradient_corrected.median_power_db == pytest.approx(30.0)
    assert cohort.final.median_power_db == pytest.approx(25.0)
    assert cohort.raw.confidence_low_power_db == pytest.approx(0.0)
    assert cohort.raw.confidence_high_power_db == pytest.approx(100.0)


def test_cohort_scanner_spectra_reject_duplicate_runs() -> None:
    duplicate = _run("sub-0001", 1, 0.0)

    with pytest.raises(ValueError, match="Duplicate scanner spectrum run"):
        aggregate_cohort_scanner_spectra(
            (duplicate, duplicate),
            bootstrap_iterations=20,
            confidence_level=0.95,
            bootstrap_seed=42,
        )


def test_cohort_scanner_spectra_reject_inconsistent_frequency_bins() -> None:
    inconsistent_stages = _harmonic_stages(
        offset_db=0.0,
        frequencies_hz=np.arange(10.0, 101.0, 0.5),
    )
    inconsistent = extract_run_scanner_spectra(
        subject="sub-0002",
        run=1,
        harmonic_stages=inconsistent_stages,
    )

    with pytest.raises(ValueError, match="frequency bins"):
        aggregate_cohort_scanner_spectra(
            (_run("sub-0001", 1, 0.0), inconsistent),
            bootstrap_iterations=20,
            confidence_level=0.95,
            bootstrap_seed=42,
        )


def test_write_cohort_scanner_spectra_tsv_preserves_plotted_values(
    tmp_path: Path,
) -> None:
    cohort = aggregate_cohort_scanner_spectra(
        (_run("sub-0001", 1, 0.0), _run("sub-0002", 1, 10.0)),
        bootstrap_iterations=20,
        confidence_level=0.95,
        bootstrap_seed=42,
    )
    output_path = tmp_path / "cohort_scanner_spectrum_qc.tsv"

    write_cohort_scanner_spectra_tsv(cohort, output_path)

    rows = list(csv.DictReader(output_path.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 3 * 76
    assert list(rows[0]) == [
        "stage",
        "frequency_hz",
        "median_psd_db_v2_hz",
        "ci_low_psd_db_v2_hz",
        "ci_high_psd_db_v2_hz",
        "n_participants",
        "n_runs",
    ]
    assert rows[0] == {
        "stage": "raw",
        "frequency_hz": "15.0",
        "median_psd_db_v2_hz": "5.0",
        "ci_low_psd_db_v2_hz": "0.0",
        "ci_high_psd_db_v2_hz": "10.0",
        "n_participants": "2",
        "n_runs": "2",
    }
    assert rows[-1]["stage"] == "final"
    assert rows[-1]["frequency_hz"] == "90.0"
