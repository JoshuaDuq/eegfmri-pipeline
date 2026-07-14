from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.analysis.qc.residual_gradient import (
    InjectionSettings,
    SelectionThresholds,
    build_validation_injection,
    compute_outside_harmonic_psd_change,
    compute_volume_locked_rms,
    evaluate_injection_recovery,
    select_component_count,
)


def test_validation_injection_is_deterministic_and_not_volume_locked() -> None:
    settings = InjectionSettings(
        frequencies_hz=(15.5, 26.0, 34.0, 49.5, 72.0),
        sinusoid_amplitude_uv=1.0,
        transient_amplitude_uv=2.0,
        transient_spacing_s=13.7,
    )

    first = build_validation_injection(60_000, 1_000.0, settings)
    second = build_validation_injection(60_000, 1_000.0, settings)

    np.testing.assert_array_equal(first.signal_v, second.signal_v)
    assert all(sample % 900 != 0 for sample in first.transient_samples)


def test_recovery_reports_identity() -> None:
    injection = build_validation_injection(60_000, 1_000.0, InjectionSettings())

    metrics = evaluate_injection_recovery(
        expected=injection,
        recovered_v=injection.signal_v,
        sfreq=1_000.0,
    )

    assert metrics.minimum_sinusoid_amplitude_ratio == pytest.approx(1.0)
    assert metrics.maximum_phase_error_deg == pytest.approx(0.0, abs=1e-12)
    assert metrics.transient_peak_ratio == pytest.approx(1.0)


def test_volume_locked_rms_matches_known_phase_locked_mean() -> None:
    template = np.array([1.0, -1.0, 1.0, -1.0])
    data = np.vstack([np.tile(template, 3), np.tile(2.0 * template, 3)])

    rms = compute_volume_locked_rms(data, np.array([0, 4, 8]), 4)

    assert rms == pytest.approx(np.sqrt(np.mean(np.vstack([template, 2.0 * template]) ** 2)))


def test_outside_harmonic_psd_change_is_zero_for_identity() -> None:
    time = np.arange(20_000) / 1_000.0
    data = np.vstack(
        [
            np.sin(2 * np.pi * 15.5 * time),
            np.sin(2 * np.pi * 26.0 * time),
        ]
    )

    metrics = compute_outside_harmonic_psd_change(
        data,
        data,
        sfreq=1_000.0,
        nperseg=4_096,
        harmonic_windows_hz=((18.0, 23.0), (38.0, 43.0)),
    )

    assert metrics.median_change_db == 0.0
    assert metrics.maximum_channel_absolute_change_db == 0.0


def test_selection_returns_smallest_fully_eligible_order() -> None:
    run_rows = []
    for run in range(1, 7):
        for count, reduction in ((0, 0.0), (1, 2.0), (2, 4.0), (3, 5.0)):
            row = {"run": run, "n_components": count, "volume_locked_rms": 10.0 - reduction}
            for label in ("18_23", "38_43", "56_67", "77_85"):
                row[f"harmonic_{label}_peak_power_db"] = 20.0 - reduction
                row[f"harmonic_{label}_prominence_db"] = 10.0 - reduction
            run_rows.append(row)
    preservation_rows = [
        {
            "n_components": count,
            "minimum_sinusoid_amplitude_ratio": 0.98 if count <= 2 else 0.90,
            "maximum_phase_error_deg": 2.0,
            "transient_peak_ratio": 0.98,
            "outside_harmonic_psd_change_db": 0.2,
        }
        for count in (1, 2, 3)
    ]

    decision = select_component_count(
        run_rows,
        preservation_rows,
        (0, 1, 2, 3),
        SelectionThresholds(),
    )

    assert decision.selected_components == 1
    assert decision.status == "accepted"
