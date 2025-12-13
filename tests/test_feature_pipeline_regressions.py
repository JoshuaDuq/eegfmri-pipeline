import numpy as np

from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec
from eeg_pipeline.analysis.features.aperiodic import _fit_single_epoch_channel
from eeg_pipeline.analysis.features.microstates import _compute_metrics


def test_time_windows_from_spec_matches_spec_masks():
    times = np.linspace(-1.0, 2.0, 301)
    config = {
        "time_frequency_analysis": {
            "baseline_window": [-0.5, -0.1],
            "plateau_window": [0.5, 1.5],
        },
        "feature_engineering": {
            "features": {"ramp_end": 0.5},
        },
    }

    spec = TimeWindowSpec(times=times, config=config, sampling_rate=100.0)
    windows = time_windows_from_spec(spec, n_plateau_windows=2, strict=True)

    assert np.array_equal(windows.baseline_mask, spec.get_mask("baseline"))
    assert np.array_equal(windows.active_mask, spec.get_mask("plateau"))
    assert np.array_equal(windows.get_mask("ramp"), spec.get_mask("ramp"))


def test_aperiodic_fit_returns_exact_kept_indices():
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    log_freqs = np.log10(freqs)

    # Mostly linear with one big outlier (to be rejected)
    psd_vals = np.array([0.0, 0.1, 100.0, 0.3, 0.4])

    (
        ep_idx,
        ch_idx,
        intercept,
        slope,
        valid_bins,
        kept_bins,
        peak_rejected,
        kept_indices,
        status,
    ) = _fit_single_epoch_channel(
        0,
        0,
        log_freqs,
        psd_vals,
        peak_rejection_z=0.5,
        min_fit_points=3,
    )

    assert status == 0
    assert valid_bins == 5
    assert kept_bins == kept_indices.size

    # The outlier at index 2 should be rejected
    assert 2 not in set(kept_indices.tolist())


def test_microstate_metrics_invalid_samples_break_runs():
    record = {}
    lbls = np.array([0, 0, 1, 1, 1, 0, 0], dtype=int)
    valid_mask = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)  # invalid at one sample

    _compute_metrics(
        lbls,
        sfreq=1.0,
        n_states=2,
        record=record,
        min_run_ms=0.0,
        valid_mask=valid_mask,
    )

    # Valid duration is 6s at 1 Hz (one invalid sample); transitions should be divided by 6.
    # Valid-sample run sequence should be: 0(2) -> 1(1) -> 0(2)
    # thus only one transition 0->1.
    assert np.isclose(record["trans_0_to_1"], 1.0 / 6.0)
    assert np.isclose(record["trans_1_to_0"], 0.0)
