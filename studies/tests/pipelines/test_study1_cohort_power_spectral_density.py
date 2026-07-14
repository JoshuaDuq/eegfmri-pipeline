from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_cohort_psd_specification_loads_fixed_settings() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        cohort_psd_specification,
    )

    specification = cohort_psd_specification(load_study1_config())

    assert specification.spectrum.frequency_range_hz == (1.0, 90.0)
    assert specification.spectrum.n_fft == 8192
    assert specification.spectrum.n_overlap == 4096
    assert specification.spectrum.sampling_frequency_hz == 500.0
    assert specification.excluded_subjects == ("sub-0006",)


def test_build_cohort_psd_summary_aggregates_runs_in_linear_units() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        build_cohort_psd_summary,
    )

    summary = build_cohort_psd_summary(
        (
            _run("sub-01", 1, [1.0, 9.0]),
            _run("sub-01", 2, [9.0, 1.0]),
            _run("sub-02", 1, [4.0, 4.0]),
        ),
        _specification(),
        bootstrap=_bootstrap(),
    )

    participant = summary.participant_spectra.set_index("subject_id")
    sub_01 = participant.loc["sub-01", "psd_db_uv2_hz"].to_numpy(dtype=float)
    assert sub_01 == pytest.approx(10.0 * np.log10(np.asarray([5.0, 5.0]) * 1e12))
    assert participant.groupby(level=0)["n_runs"].first().to_dict() == {
        "sub-01": 2,
        "sub-02": 1,
    }
    assert summary.n_subjects == 2
    assert summary.n_runs == 3


def test_build_cohort_psd_summary_bootstraps_complete_participants() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        build_cohort_psd_summary,
    )

    summary = build_cohort_psd_summary(
        (
            _run("sub-01", 1, [1.0, 2.0]),
            _run("sub-02", 1, [3.0, 4.0]),
            _run("sub-03", 1, [8.0, 16.0]),
        ),
        _specification(),
        bootstrap=_bootstrap(),
    )

    matrix = summary.participant_spectra.pivot(
        index="subject_id",
        columns="frequency_hz",
        values="psd_db_uv2_hz",
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, matrix.shape[0], size=(200, matrix.shape[0]))
    estimates = np.median(matrix[indices], axis=1)
    cohort = summary.cohort_spectrum
    assert cohort["median_psd_db_uv2_hz"].to_numpy() == pytest.approx(np.median(matrix, axis=0))
    assert cohort["ci_low_psd_db_uv2_hz"].to_numpy() == pytest.approx(
        np.quantile(estimates, 0.025, axis=0)
    )
    assert cohort["ci_high_psd_db_uv2_hz"].to_numpy() == pytest.approx(
        np.quantile(estimates, 0.975, axis=0)
    )


def test_build_cohort_psd_summary_writes_exact_audit_columns() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        build_cohort_psd_summary,
    )

    summary = build_cohort_psd_summary(
        (_run("sub-01", 1, [1.0, 2.0]),),
        _specification(),
        bootstrap=_bootstrap(),
    )

    assert summary.run_audit.columns.tolist() == [
        "subject_id",
        "run",
        "source_file",
        "n_channels",
        "sampling_frequency_hz",
        "n_samples",
        "recording_duration_s",
        "bad_annotation_duration_s",
        "analyzed_duration_s",
        "frequency_min_hz",
        "frequency_max_hz",
        "n_fft",
        "n_overlap",
        "frequency_resolution_hz",
    ]
    assert summary.participant_spectra.columns.tolist() == [
        "subject_id",
        "frequency_hz",
        "psd_db_uv2_hz",
        "n_runs",
    ]
    assert summary.cohort_spectrum.columns.tolist() == [
        "frequency_hz",
        "median_psd_db_uv2_hz",
        "ci_low_psd_db_uv2_hz",
        "ci_high_psd_db_uv2_hz",
        "n_subjects",
    ]


def test_build_cohort_psd_summary_rejects_empty_runs() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        build_cohort_psd_summary,
    )

    with pytest.raises(ValueError, match="requires at least one run spectrum"):
        build_cohort_psd_summary((), _specification(), bootstrap=_bootstrap())


def test_build_cohort_psd_summary_rejects_inconsistent_frequency_bins() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        build_cohort_psd_summary,
    )

    first = _run("sub-01", 1, [1.0, 2.0])
    second = _run("sub-02", 1, [1.0, 2.0])
    second.frequencies_hz[1] = 11.0

    with pytest.raises(ValueError, match="inconsistent frequency bins"):
        build_cohort_psd_summary(
            (first, second),
            _specification(),
            bootstrap=_bootstrap(),
        )


def test_build_cohort_psd_summary_rejects_invalid_power() -> None:
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        build_cohort_psd_summary,
    )

    invalid = _run("sub-01", 1, [1.0, 0.0])

    with pytest.raises(ValueError, match="strictly positive finite power"):
        build_cohort_psd_summary(
            (invalid,),
            _specification(),
            bootstrap=_bootstrap(),
        )


def _specification():
    from studies.pain_study.study1.figures.cohort_power_spectral_density import (
        CohortPsdSpecification,
    )
    from studies.pain_study.study1.figures.continuous_spectrum import (
        ContinuousSpectrumSpecification,
    )

    return CohortPsdSpecification(
        spectrum=ContinuousSpectrumSpecification(
            frequency_range_hz=(1.0, 90.0),
            n_fft=8192,
            n_overlap=4096,
            sampling_frequency_hz=500.0,
        ),
        excluded_subjects=("sub-0006",),
    )


def _bootstrap():
    from studies.pain_study.study1.figures.spectral_statistics import (
        ParticipantBootstrapSpecification,
    )

    return ParticipantBootstrapSpecification(
        iterations=200,
        confidence_level=0.95,
        seed=42,
    )


def _run(subject_id: str, run: int, power: list[float]):
    from studies.pain_study.study1.figures.continuous_spectrum import (
        ContinuousRunSpectrum,
    )

    frequencies = np.asarray([1.0, 10.0])
    return ContinuousRunSpectrum(
        subject_id=subject_id,
        run_id=str(run),
        source_file=Path(f"{subject_id}_run-{run}.fif"),
        frequencies_hz=frequencies,
        median_psd_v2_hz=np.asarray(power, dtype=float),
        n_channels=55,
        sampling_frequency_hz=500.0,
        n_samples=50_000,
        recording_duration_s=100.0,
        bad_annotation_duration_s=3.0,
        analyzed_duration_s=97.0,
    )
