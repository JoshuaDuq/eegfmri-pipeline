from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.analysis.qc.native_residual_obs import (
    NativeObsSelectionThresholds,
    select_native_obs_component_count,
    summarize_fixed_frequency_lines,
)


def test_fixed_frequency_summary_reports_power_and_local_prominence() -> None:
    sampling_frequency = 1_000.0
    time = np.arange(32_768) / sampling_frequency
    data = np.vstack(
        [
            5e-6 * np.sin(2 * np.pi * 41.137 * time),
            4e-6 * np.sin(2 * np.pi * 41.137 * time + 0.3),
        ]
    )

    rows = summarize_fixed_frequency_lines(
        data,
        sampling_frequency=sampling_frequency,
        reference_frequencies_hz=(41.137,),
        welch_duration_seconds=16.384,
        overlap_fraction=0.5,
        background_inner_hz=0.35,
        background_outer_hz=2.0,
    )

    assert len(rows) == 1
    assert rows[0]["evaluated_frequency_hz"] == pytest.approx(41.137, abs=0.031)
    assert rows[0]["local_prominence_db"] > 20.0


def test_selection_accepts_smallest_order_passing_all_gates() -> None:
    line_rows: list[dict[str, float | int | str]] = []
    run_rows: list[dict[str, float | int | str]] = []
    preservation_rows: list[dict[str, float | int | str]] = []
    for recording in ("sub-0001_run-1", "sub-0002_run-1", "sub-0003_run-1"):
        for components, reduction in ((0, 0.0), (1, 2.0), (2, 3.0)):
            run_rows.append(
                {
                    "recording_id": recording,
                    "n_components": components,
                    "volume_locked_rms_v": 10.0 - reduction,
                }
            )
            for frequency in (41.14, 61.10):
                line_rows.append(
                    {
                        "recording_id": recording,
                        "n_components": components,
                        "reference_frequency_hz": frequency,
                        "power_db": 20.0 - reduction,
                        "local_prominence_db": 12.0 - reduction,
                    }
                )
            if components:
                preservation_rows.append(
                    {
                        "recording_id": recording,
                        "n_components": components,
                        "minimum_sinusoid_amplitude_ratio": 0.98,
                        "maximum_phase_error_deg": 1.0,
                        "transient_peak_ratio": 0.98,
                        "outside_line_psd_change_db": 0.1,
                        "maximum_channel_outside_line_psd_change_db": 0.2,
                    }
                )

    decision = select_native_obs_component_count(
        line_rows=line_rows,
        run_rows=run_rows,
        preservation_rows=preservation_rows,
        component_counts=(0, 1, 2),
        thresholds=NativeObsSelectionThresholds(),
    )

    assert decision.status == "accepted"
    assert decision.selected_components == 1
    assert decision.reasons == ()


def test_selection_reports_specific_failed_gates() -> None:
    line_rows = [
        {
            "recording_id": "sub-0001_run-1",
            "n_components": components,
            "reference_frequency_hz": 61.10,
            "power_db": 20.0 - reduction,
            "local_prominence_db": 12.0 - reduction,
        }
        for components, reduction in ((0, 0.0), (1, 0.2))
    ]
    run_rows = [
        {
            "recording_id": "sub-0001_run-1",
            "n_components": components,
            "volume_locked_rms_v": rms,
        }
        for components, rms in ((0, 10.0), (1, 9.9))
    ]
    preservation_rows = [
        {
            "recording_id": "sub-0001_run-1",
            "n_components": 1,
            "minimum_sinusoid_amplitude_ratio": 0.90,
            "maximum_phase_error_deg": 1.0,
            "transient_peak_ratio": 0.98,
            "outside_line_psd_change_db": 0.1,
            "maximum_channel_outside_line_psd_change_db": 0.2,
        }
    ]

    decision = select_native_obs_component_count(
        line_rows=line_rows,
        run_rows=run_rows,
        preservation_rows=preservation_rows,
        component_counts=(0, 1),
        thresholds=NativeObsSelectionThresholds(),
    )

    assert decision.status == "rejected"
    assert any("line power reduction" in reason for reason in decision.reasons)
    assert any("sinusoid amplitude" in reason for reason in decision.reasons)
