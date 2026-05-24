from __future__ import annotations

import logging
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.analysis.features.api import (
    _compute_tfr_for_features,
    _extract_feature_with_error_handling,
    extract_precomputed_features,
)
from eeg_pipeline.analysis.features.bursts import (
    _parse_burst_config,
    extract_burst_features,
)
from eeg_pipeline.analysis.features.aperiodic import (
    _parse_line_noise_config,
    extract_aperiodic_features,
    extract_aperiodic_from_precomputed,
)
from eeg_pipeline.analysis.features.connectivity import (
    ConnectivityConfig,
    _apply_across_epochs_phase_estimates_inplace,
    _safe_n_cycles_for_segment,
    extract_connectivity_features,
    extract_connectivity_from_precomputed,
    extract_directed_connectivity_from_precomputed,
)
from eeg_pipeline.analysis.features.complexity import extract_complexity_from_precomputed
from eeg_pipeline.analysis.features.erp import _build_component_masks, _parse_lowpass_filter
from eeg_pipeline.analysis.features.phase import (
    _get_valid_pac_pairs,
    _nonnegative_float_or_default,
    _positive_float_or_default,
    extract_itpc_from_precomputed,
    extract_pac_from_precomputed,
    _rng_from_seed,
)
from eeg_pipeline.analysis.features.preparation import (
    _compute_psd_with_qc,
    _get_spatial_transform_type,
    precompute_data,
)
from eeg_pipeline.analysis.features.precomputed.erds import extract_erds_from_precomputed
from eeg_pipeline.analysis.features.precomputed.extras import (
    _get_psd_config,
    extract_asymmetry_from_precomputed,
    extract_band_ratios_from_precomputed,
)
from eeg_pipeline.analysis.features.microstates import _load_microstate_config
from eeg_pipeline.analysis.features.quality import _extract_quality_config
from eeg_pipeline.analysis.features.spectral import (
    _resolve_line_noise_freqs,
    extract_spectral_features,
)
from eeg_pipeline.types import BandData, PrecomputedData, TimeWindows
from eeg_pipeline.utils.analysis.spatial import build_roi_map_if_needed, crop_epochs_to_time_range
from eeg_pipeline.utils.analysis.spectral import subtract_evoked
from eeg_pipeline.utils.analysis.tfr import (
    apply_baseline_safe,
    apply_baseline_and_crop,
    compute_adaptive_n_cycles,
    get_tfr_config,
    get_tfr_decim,
    restrict_epochs_to_roi,
)
from tests.pipelines_test_utils import DotConfig, NoopProgress


def test_precomputed_psd_requires_baseline_for_event_related_compute(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("compute_psd should not run without a baseline")

    monkeypatch.setattr(
        "eeg_pipeline.analysis.features.preparation.compute_psd",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="baseline.*required"):
        _compute_psd_with_qc(
            data=np.ones((2, 2, 128), dtype=float),
            sfreq=100.0,
            baseline_mask=np.zeros(128, dtype=bool),
            config=DotConfig({"feature_engineering": {"task_is_rest": False}}),
            logger=logging.getLogger("strict-psd-baseline"),
        )


def test_subtract_evoked_rejects_missing_condition_labels() -> None:
    data = np.ones((4, 2, 20), dtype=float)

    with pytest.raises(ValueError, match="condition_labels"):
        subtract_evoked(data, condition_labels=None)


def test_subtract_evoked_rejects_underpowered_condition_groups() -> None:
    data = np.ones((4, 2, 20), dtype=float)
    labels = np.array(["a", "a", "b", "c"], dtype=object)

    with pytest.raises(ValueError, match="fewer than min_trials_per_condition"):
        subtract_evoked(data, condition_labels=labels, min_trials_per_condition=2)


def test_subtract_evoked_rejects_invalid_min_trials_threshold() -> None:
    data = np.ones((4, 2, 20), dtype=float)
    labels = np.array(["a", "a", "b", "b"], dtype=object)

    with pytest.raises(ValueError, match="min_trials_per_condition"):
        subtract_evoked(data, condition_labels=labels, min_trials_per_condition=0)


def test_requested_feature_empty_output_raises() -> None:
    ctx = SimpleNamespace(
        config=DotConfig({}),
        logger=logging.getLogger("strict-empty-feature"),
    )

    with pytest.raises(ValueError, match="power.*produced no features"):
        _extract_feature_with_error_handling(
            ctx,
            "power",
            lambda: (pd.DataFrame(), []),
            expected_trials=2,
            progress=NoopProgress(),
        )


def test_precomputed_requested_feature_empty_output_raises(monkeypatch) -> None:
    precomputed = PrecomputedData(
        data=np.ones((2, 1, 8), dtype=float),
        times=np.linspace(0.0, 0.7, 8),
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        config=DotConfig({}),
    )

    monkeypatch.setattr(
        "eeg_pipeline.analysis.features.api.extract_power_from_precomputed",
        lambda *_args, **_kwargs: (pd.DataFrame(), [], {}),
    )

    with pytest.raises(ValueError, match="spectral.*produced no features"):
        extract_precomputed_features(
            epochs=SimpleNamespace(),
            bands=["alpha"],
            config=DotConfig({}),
            logger=logging.getLogger("strict-precomputed-empty-feature"),
            feature_groups=["spectral"],
            precomputed=precomputed,
        )


def test_invalid_spatial_transform_values_raise() -> None:
    config = DotConfig(
        {
            "feature_engineering": {
                "spatial_transform": "not-a-transform",
                "spatial_transform_per_family": {},
            }
        }
    )

    with pytest.raises(ValueError, match="spatial_transform"):
        _get_spatial_transform_type(config, feature_family="connectivity")


def test_precompute_raises_when_no_eeg_channels_remain() -> None:
    data = np.ones((2, 1, 128), dtype=float)
    info = mne.create_info(["TRIG"], sfreq=100.0, ch_types=["stim"])
    epochs = mne.EpochsArray(data, info, verbose=False)

    with pytest.raises(ValueError, match="No EEG channels"):
        precompute_data(
            epochs,
            bands=["alpha"],
            config=DotConfig({}),
            logger=logging.getLogger("strict-no-eeg"),
            compute_bands=False,
            compute_psd_data=False,
        )


class _BaselineTFR:
    def __init__(self) -> None:
        self.times = np.array([0.0, 0.1, 0.2], dtype=float)
        self.comment = ""
        self.apply_baseline_called = False

    def apply_baseline(self, *_args, **_kwargs) -> None:
        self.apply_baseline_called = True


def test_baseline_window_outside_data_raises_without_clipping() -> None:
    tfr = _BaselineTFR()

    with pytest.raises(ValueError, match="outside available data range"):
        apply_baseline_safe(
            tfr,
            baseline=(-0.2, 0.0),
            mode="logratio",
            logger=logging.getLogger("strict-baseline-window"),
            min_samples=1,
            config=DotConfig({}),
        )

    assert not tfr.apply_baseline_called


class _CropTFR:
    def __init__(self) -> None:
        self.times = np.array([0.0, 0.1, 0.2], dtype=float)
        self.comment = "BASELINED:mode=logratio;win=(0.000,0.100)"
        self.cropped = None

    def crop(self, *, tmin, tmax) -> None:
        self.cropped = (tmin, tmax)


def test_crop_window_outside_data_raises_without_full_range_substitution() -> None:
    tfr = _CropTFR()

    with pytest.raises(ValueError, match="outside available data range"):
        apply_baseline_and_crop(
            tfr,
            baseline=(0.0, 0.1),
            crop_window=(1.0, 2.0),
            mode="logratio",
            logger=logging.getLogger("strict-crop-window"),
            min_samples=1,
            config=DotConfig({}),
        )

    assert tfr.cropped is None


def test_burst_condition_thresholds_require_condition_labels() -> None:
    n_epochs = 3
    n_times = 8
    baseline_mask = np.array([True, True, True, True, False, False, False, False])
    active_mask = ~baseline_mask
    windows = TimeWindows(
        baseline_mask=baseline_mask,
        active_mask=active_mask,
        masks={"baseline": baseline_mask, "active": active_mask},
        ranges={"baseline": (-0.4, 0.0), "active": (0.0, 0.4)},
        times=np.linspace(-0.4, 0.3, n_times),
    )
    envelope = np.ones((n_epochs, 1, n_times), dtype=float)
    precomputed = PrecomputedData(
        data=envelope.copy(),
        times=windows.times,
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        windows=windows,
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=envelope.copy(),
                analytic=envelope.astype(complex),
                envelope=envelope,
                phase=np.zeros_like(envelope),
                power=envelope**2,
            )
        },
        config=DotConfig({}),
        spatial_modes=["global"],
    )
    ctx = SimpleNamespace(
        precomputed=precomputed,
        config=DotConfig(
            {
                "feature_engineering": {
                    "bursts": {
                        "threshold_reference": "condition",
                        "min_trials_per_condition": 1,
                    }
                }
            }
        ),
        logger=logging.getLogger("strict-burst-condition-labels"),
        spatial_modes=["global"],
    )

    with pytest.raises(ValueError, match="condition labels"):
        extract_burst_features(ctx, ["alpha"])


def test_invalid_burst_threshold_config_raises() -> None:
    with pytest.raises(ValueError, match="threshold_method"):
        _parse_burst_config(
            DotConfig({"feature_engineering": {"bursts": {"threshold_method": "bad"}}}),
            ["alpha"],
        )

    with pytest.raises(ValueError, match="threshold_reference"):
        _parse_burst_config(
            DotConfig({"feature_engineering": {"bursts": {"threshold_reference": "bad"}}}),
            ["alpha"],
        )


def test_pac_rng_requires_explicit_valid_seed() -> None:
    with pytest.raises(ValueError, match="random_seed"):
        _rng_from_seed(None)

    with pytest.raises(ValueError, match="random_seed"):
        _rng_from_seed("not-an-int")

    first = _rng_from_seed(42).random(5)
    second = _rng_from_seed("42").random(5)
    np.testing.assert_allclose(first, second)


def test_microstate_config_rejects_invalid_values() -> None:
    invalid_configs = [
        {"n_states": 1},
        {"n_states": 13},
        {"min_peak_distance_ms": -1.0},
        {"max_gfp_peaks_per_epoch": 9},
        {"min_duration_ms": -1.0},
        {"gfp_peak_prominence": -0.1},
    ]

    for microstates_cfg in invalid_configs:
        with pytest.raises(ValueError):
            _load_microstate_config(
                DotConfig({"feature_engineering": {"microstates": microstates_cfg}})
            )


def test_erds_log_ratio_applies_min_active_power_floor() -> None:
    n_epochs = 1
    n_times = 6
    baseline_mask = np.array([True, True, True, False, False, False])
    active_mask = ~baseline_mask
    times = np.linspace(-0.3, 0.2, n_times)
    power = np.ones((n_epochs, 1, n_times), dtype=float)
    power[:, :, active_mask] = 0.0
    windows = TimeWindows(
        baseline_mask=baseline_mask,
        active_mask=active_mask,
        masks={"baseline": baseline_mask, "active": active_mask},
        ranges={"baseline": (-0.3, 0.0), "active": (0.0, 0.3)},
        times=times,
    )
    precomputed = PrecomputedData(
        data=np.ones_like(power),
        times=times,
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        windows=windows,
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=np.ones_like(power),
                analytic=np.ones_like(power, dtype=complex),
                envelope=np.sqrt(power),
                phase=np.zeros_like(power),
                power=power,
            )
        },
        config=DotConfig(
            {
                "feature_engineering": {
                    "constants": {
                        "min_epochs_for_features": 1,
                        "min_valid_fraction": 0.5,
                        "epsilon_std": 1e-12,
                    },
                    "erds": {
                        "use_log_ratio": True,
                        "min_baseline_power": 1e-12,
                        "min_active_power": 0.1,
                    },
                }
            }
        ),
        logger=logging.getLogger("strict-erds-active-floor"),
        spatial_modes=["channels"],
    )

    df, _, _ = extract_erds_from_precomputed(precomputed, ["alpha"])

    assert df.loc[0, "erds_active_alpha_ch_Cz_db"] == pytest.approx(-10.0)


def test_connectivity_subject_granularity_requires_explicit_across_epoch_estimator(
    monkeypatch,
) -> None:
    precomputed = PrecomputedData(
        data=np.ones((3, 2, 32), dtype=float),
        times=np.arange(32, dtype=float) / 100.0,
        sfreq=100.0,
        ch_names=["C3", "C4"],
        picks=np.array([0, 1], dtype=int),
        config=DotConfig({}),
        spatial_transform="none",
    )
    ctx = SimpleNamespace(
        config=DotConfig(
            {
                "feature_engineering": {
                    "spatial_transform": "none",
                    "connectivity": {
                        "granularity": "subject",
                        "phase_estimator": "within_epoch",
                        "measures": ["aec"],
                    },
                }
            }
        ),
        logger=logging.getLogger("strict-connectivity-estimator"),
        precomputed=precomputed,
        train_mask=None,
        name="active",
        windows=TimeWindows(
            masks={"active": np.ones(32, dtype=bool)},
            ranges={"active": (0.0, 0.31)},
            times=precomputed.times,
        ),
    )

    monkeypatch.setattr(
        "eeg_pipeline.analysis.features.connectivity.extract_connectivity_from_precomputed",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"conn_active_alpha_global_aec": [0.1, 0.1, 0.1]}),
            ["conn_active_alpha_global_aec"],
        ),
    )

    with pytest.raises(ValueError, match="phase_estimator='across_epochs'"):
        extract_connectivity_features(ctx, ["alpha"])


def test_phase_duration_guard_parsers_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        _nonnegative_float_or_default("not-a-number", 1.0)

    with pytest.raises(ValueError):
        _positive_float_or_default(0.0, 3.0)


def _precomputed_for_segment_strictness() -> PrecomputedData:
    n_epochs = 2
    n_times = 240
    times = np.arange(n_times, dtype=float) / 100.0
    short_mask = np.zeros(n_times, dtype=bool)
    short_mask[:20] = True
    long_mask = np.zeros(n_times, dtype=bool)
    long_mask[20:220] = True
    data = np.sin(2.0 * np.pi * 10.0 * times)[None, None, :]
    data = np.repeat(data, n_epochs, axis=0)
    data = np.repeat(data, 2, axis=1)
    return PrecomputedData(
        data=data,
        times=times,
        sfreq=100.0,
        ch_names=["C3", "C4"],
        picks=np.array([0, 1], dtype=int),
        windows=TimeWindows(
            masks={"short": short_mask, "long": long_mask},
            ranges={"short": (0.0, 0.19), "long": (0.2, 2.19)},
            times=times,
        ),
        config=DotConfig(
            {
                "feature_engineering": {
                    "spectral": {
                        "ratio_pairs": [["alpha", "theta"]],
                        "include_log_ratios": True,
                    },
                    "ratios": {
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                    },
                    "asymmetry": {
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                    },
                },
                "time_frequency_analysis": {"bands": {"theta": [4.0, 8.0], "alpha": [8.0, 12.0]}},
            }
        ),
        logger=logging.getLogger("strict-segment-skip"),
        spatial_modes=["global"],
    )


def test_band_ratios_fail_when_requested_segment_is_too_short() -> None:
    precomputed = _precomputed_for_segment_strictness()

    with pytest.raises(ValueError, match="Band ratios.*too short"):
        extract_band_ratios_from_precomputed(precomputed, precomputed.config)


def test_asymmetry_fails_when_requested_segment_is_too_short() -> None:
    precomputed = _precomputed_for_segment_strictness()

    with pytest.raises(ValueError, match="Asymmetry.*too short"):
        extract_asymmetry_from_precomputed(precomputed)


def test_spectral_descriptor_rejects_invalid_psd_method() -> None:
    info = mne.create_info(["Cz", "Pz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(np.ones((2, 2, 240), dtype=float), info, verbose=False)
    mask = np.ones(240, dtype=bool)
    ctx = SimpleNamespace(
        epochs=epochs,
        windows=TimeWindows(
            masks={"active": mask},
            ranges={"active": (float(epochs.times[0]), float(epochs.times[-1]))},
            times=epochs.times,
        ),
        config=DotConfig({"feature_engineering": {"spectral": {"psd_method": "invalid"}}}),
        logger=logging.getLogger("strict-spectral-psd-method"),
        spatial_modes=["global"],
    )

    with pytest.raises(ValueError, match="psd_method"):
        extract_spectral_features(ctx, ["alpha"])


def test_erp_lowpass_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="lowpass_hz"):
        _parse_lowpass_filter({"lowpass_hz": "not-a-number"})

    with pytest.raises(ValueError, match="lowpass_hz"):
        _parse_lowpass_filter({"lowpass_hz": -1.0})


def test_aperiodic_rejects_invalid_qc_thresholds() -> None:
    info = mne.create_info(["Cz", "Pz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(np.ones((2, 2, 300), dtype=float), info, verbose=False)
    mask = np.ones(300, dtype=bool)
    ctx = SimpleNamespace(
        epochs=epochs,
        windows=TimeWindows(
            masks={"active": mask},
            ranges={"active": (float(epochs.times[0]), float(epochs.times[-1]))},
            times=epochs.times,
        ),
        config=DotConfig({"feature_engineering": {"aperiodic": {"min_segment_sec": float("nan")}}}),
        logger=logging.getLogger("strict-aperiodic-min-segment"),
    )

    with pytest.raises(ValueError, match="min_segment_sec"):
        extract_aperiodic_features(ctx, ["alpha"])

    precomputed = PrecomputedData(
        data=np.ones((2, 2, 300), dtype=float),
        times=epochs.times,
        sfreq=100.0,
        ch_names=["Cz", "Pz"],
        picks=np.array([0, 1], dtype=int),
        windows=ctx.windows,
        config=DotConfig({"feature_engineering": {"aperiodic": {"min_r2": float("nan")}}}),
        logger=logging.getLogger("strict-aperiodic-min-r2"),
    )

    with pytest.raises(ValueError, match="min_r2"):
        extract_aperiodic_from_precomputed(precomputed, ["alpha"])


def test_quality_line_noise_config_rejects_invalid_preprocessing_fallback() -> None:
    with pytest.raises(ValueError, match="preprocessing.line_freq"):
        _extract_quality_config(
            DotConfig(
                {
                    "preprocessing": {"line_freq": "not-a-frequency"},
                    "feature_engineering": {"quality": {}},
                }
            )
        )


def test_missing_tfr_roi_selection_raises() -> None:
    info = mne.create_info(["Cz", "Pz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(np.ones((2, 2, 32), dtype=float), info, verbose=False)

    with pytest.raises(ValueError, match="ROI 'missing'"):
        restrict_epochs_to_roi(
            epochs,
            "missing",
            DotConfig({"time_frequency_analysis": {"rois": {"central": ["Cz"]}}}),
            logging.getLogger("strict-tfr-roi"),
        )


def test_connectivity_dynamic_config_rejects_invalid_values() -> None:
    invalid_configs = [
        {"dynamic_enabled": True, "sliding_window_len": 0.0},
        {"dynamic_enabled": True, "sliding_window_step": -1.0},
        {"dynamic_enabled": True, "dynamic_measures": ["bad"]},
        {"dynamic_enabled": True, "dynamic_autocorr_lag": 0},
        {"dynamic_enabled": True, "dynamic_min_windows": 1},
        {"dynamic_enabled": True, "dynamic_state_n_states": 1},
        {"dynamic_enabled": True, "dynamic_state_min_windows": 2},
    ]

    for connectivity_cfg in invalid_configs:
        with pytest.raises(ValueError):
            ConnectivityConfig.from_dict(
                {"feature_engineering": {"connectivity": connectivity_cfg}}
            )


def test_connectivity_dynamic_state_count_requires_enough_windows() -> None:
    n_epochs = 2
    n_times = 60
    times = np.arange(n_times, dtype=float) / 100.0
    signal = np.exp(1j * 2.0 * np.pi * 10.0 * times)
    envelope_a = 1.0 + (0.2 * np.sin(2.0 * np.pi * 1.0 * times))
    envelope_b = 1.0 + (0.2 * np.cos(2.0 * np.pi * 1.0 * times))
    analytic_epoch = np.stack(
        [
            envelope_a * signal,
            envelope_b * signal * np.exp(1j * np.pi / 4.0),
        ],
        axis=0,
    )
    analytic = np.repeat(analytic_epoch[None, :, :], n_epochs, axis=0)
    precomputed = PrecomputedData(
        data=np.real(analytic),
        times=times,
        sfreq=100.0,
        ch_names=["C3", "C4"],
        picks=np.array([0, 1], dtype=int),
        windows=TimeWindows(
            masks={"active": np.ones(n_times, dtype=bool)},
            ranges={"active": (0.0, float(times[-1]))},
            times=times,
        ),
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=np.real(analytic),
                analytic=analytic,
                envelope=np.abs(analytic),
                phase=np.angle(analytic),
                power=np.ones_like(np.real(analytic)),
            )
        },
        config=DotConfig(
            {
                "feature_engineering": {
                    "connectivity": {
                        "measures": [],
                        "dynamic_enabled": True,
                        "dynamic_measures": ["aec"],
                        "dynamic_state_enabled": True,
                        "dynamic_state_n_states": 10,
                        "dynamic_state_min_windows": 3,
                        "dynamic_min_windows": 3,
                        "sliding_window_len": 0.2,
                        "sliding_window_step": 0.1,
                        "min_segment_samples": 5,
                        "min_segment_sec": 0.1,
                    }
                },
                "time_frequency_analysis": {"bands": {"alpha": [8.0, 12.0]}},
            }
        ),
        logger=logging.getLogger("strict-dynamic-state-count"),
        spatial_modes=["global"],
        frequency_bands={"alpha": [8.0, 12.0]},
    )

    with pytest.raises(ValueError, match="dynamic_state_n_states"):
        extract_connectivity_from_precomputed(precomputed, bands=["alpha"])


def test_tfr_config_rejects_invalid_ranges_and_decimation() -> None:
    with pytest.raises(ValueError, match="freq_max"):
        get_tfr_config(
            DotConfig({"time_frequency_analysis": {"tfr": {"freq_min": 30.0, "freq_max": 10.0}}})
        )

    with pytest.raises(ValueError, match="n_freqs"):
        get_tfr_config(DotConfig({"time_frequency_analysis": {"tfr": {"n_freqs": 1}}}))

    with pytest.raises(ValueError, match="decim_phase"):
        get_tfr_decim(
            DotConfig({"time_frequency_analysis": {"tfr": {"decim_phase": 0}}}),
            mode="phase",
        )

    with pytest.raises(ValueError, match="n_cycles"):
        compute_adaptive_n_cycles(
            np.array([4.0, 8.0]),
            min_cycles=3.0,
            max_cycles=2.0,
        )


def test_spectral_line_noise_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="preprocessing.line_freq"):
        _resolve_line_noise_freqs(
            {"line_noise_freqs": None},
            DotConfig({"preprocessing": {"line_freq": "bad"}}),
        )

    with pytest.raises(ValueError, match="line_noise_freqs"):
        _resolve_line_noise_freqs(
            {"line_noise_freqs": ["bad"]},
            DotConfig({"preprocessing": {"line_freq": 60.0}}),
        )


def test_precomputed_psd_config_rejects_invalid_line_noise_values() -> None:
    with pytest.raises(ValueError, match="preprocessing.line_freq"):
        _get_psd_config(
            DotConfig({"preprocessing": {"line_freq": "bad"}}),
            sfreq=100.0,
        )

    with pytest.raises(ValueError, match="line_noise_freqs"):
        _get_psd_config(
            DotConfig(
                {
                    "preprocessing": {"line_freq": 60.0},
                    "feature_engineering": {"spectral": {"line_noise_freqs": ["bad"]}},
                }
            ),
            sfreq=100.0,
        )


def test_burst_threshold_percentile_rejects_out_of_range_config() -> None:
    with pytest.raises(ValueError, match="threshold_percentile"):
        _parse_burst_config(
            DotConfig({"feature_engineering": {"bursts": {"threshold_percentile": 10.0}}}),
            ["alpha"],
        )


def test_bursts_fail_when_requested_segment_is_too_short() -> None:
    n_epochs = 2
    n_times = 140
    times = np.arange(n_times, dtype=float) / 100.0
    baseline_mask = np.zeros(n_times, dtype=bool)
    baseline_mask[:40] = True
    short_mask = np.zeros(n_times, dtype=bool)
    short_mask[40:45] = True
    long_mask = np.zeros(n_times, dtype=bool)
    long_mask[45:140] = True
    envelope = np.ones((n_epochs, 1, n_times), dtype=float)
    precomputed = PrecomputedData(
        data=envelope.copy(),
        times=times,
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        windows=TimeWindows(
            baseline_mask=baseline_mask,
            masks={
                "baseline": baseline_mask,
                "short": short_mask,
                "long": long_mask,
            },
            ranges={
                "baseline": (0.0, 0.39),
                "short": (0.4, 0.44),
                "long": (0.45, 1.39),
            },
            times=times,
        ),
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=envelope.copy(),
                analytic=envelope.astype(complex),
                envelope=envelope,
                phase=np.zeros_like(envelope),
                power=envelope**2,
            )
        },
        config=DotConfig({}),
        spatial_modes=["global"],
    )
    ctx = SimpleNamespace(
        precomputed=precomputed,
        config=DotConfig(
            {
                "feature_engineering": {
                    "bursts": {
                        "threshold_reference": "trial",
                        "min_duration_ms": 100.0,
                        "min_cycles": 1.0,
                    }
                }
            }
        ),
        logger=logging.getLogger("strict-burst-short-segment"),
        spatial_modes=["global"],
    )

    with pytest.raises(ValueError, match="Bursts.*too short"):
        extract_burst_features(ctx, ["alpha"])


def test_bursts_extract_requested_baseline_segment() -> None:
    n_epochs = 2
    n_times = 120
    times = np.arange(n_times, dtype=float) / 100.0
    baseline_mask = np.zeros(n_times, dtype=bool)
    baseline_mask[:80] = True
    active_mask = ~baseline_mask
    envelope = np.ones((n_epochs, 1, n_times), dtype=float)
    precomputed = PrecomputedData(
        data=envelope.copy(),
        times=times,
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        windows=TimeWindows(
            baseline_mask=baseline_mask,
            active_mask=active_mask,
            masks={"baseline": baseline_mask, "active": active_mask},
            ranges={"baseline": (0.0, 0.79), "active": (0.8, 1.19)},
            times=times,
            name="baseline",
        ),
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=envelope.copy(),
                analytic=envelope.astype(complex),
                envelope=envelope,
                phase=np.zeros_like(envelope),
                power=envelope**2,
            )
        },
        config=DotConfig({}),
        spatial_modes=["global"],
    )
    ctx = SimpleNamespace(
        precomputed=precomputed,
        config=DotConfig(
            {
                "feature_engineering": {
                    "bursts": {
                        "threshold_reference": "trial",
                        "min_duration_ms": 10.0,
                        "min_cycles": 1.0,
                    }
                }
            }
        ),
        logger=logging.getLogger("strict-burst-baseline"),
        spatial_modes=["global"],
    )

    df, cols = extract_burst_features(ctx, ["alpha"])

    expected_col = "bursts_baseline_alpha_global_count"
    assert expected_col in cols
    assert expected_col in df.columns


def _phase_precomputed_for_short_segment() -> PrecomputedData:
    n_epochs = 3
    n_times = 220
    times = np.arange(n_times, dtype=float) / 100.0
    short_mask = np.zeros(n_times, dtype=bool)
    short_mask[:20] = True
    long_mask = np.zeros(n_times, dtype=bool)
    long_mask[20:] = True
    phase = np.zeros((n_epochs, 1, n_times), dtype=float)
    analytic = np.exp(1j * phase)
    power = np.ones((n_epochs, 1, n_times), dtype=float)
    return PrecomputedData(
        data=np.real(analytic),
        times=times,
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        windows=TimeWindows(
            masks={"short": short_mask, "long": long_mask},
            ranges={"short": (0.0, 0.19), "long": (0.2, 2.19)},
            times=times,
        ),
        band_data={
            "theta": BandData(
                band="theta",
                fmin=4.0,
                fmax=8.0,
                filtered=np.real(analytic),
                analytic=analytic,
                envelope=np.ones_like(power),
                phase=phase,
                power=power,
            ),
            "gamma": BandData(
                band="gamma",
                fmin=30.0,
                fmax=80.0,
                filtered=np.real(analytic),
                analytic=analytic,
                envelope=np.ones_like(power),
                phase=phase,
                power=power,
            ),
        },
        config=DotConfig(
            {
                "feature_engineering": {
                    "itpc": {
                        "method": "global",
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                    },
                    "pac": {
                        "method": "mvl",
                        "pairs": [["theta", "gamma"]],
                        "n_surrogates": 0,
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                        "allow_harmonic_overlap": True,
                    },
                    "spatial_modes": ["global"],
                },
                "time_frequency_analysis": {"bands": {"theta": [4.0, 8.0], "gamma": [30.0, 80.0]}},
            }
        ),
        logger=logging.getLogger("strict-phase-short-segment"),
        spatial_modes=["global"],
        frequency_bands={"theta": [4.0, 8.0], "gamma": [30.0, 80.0]},
    )


def test_itpc_fails_when_requested_segment_is_too_short() -> None:
    precomputed = _phase_precomputed_for_short_segment()

    with pytest.raises(ValueError, match="ITPC.*too short"):
        extract_itpc_from_precomputed(precomputed, n_jobs=1)


def test_pac_fails_when_requested_segment_is_too_short() -> None:
    precomputed = _phase_precomputed_for_short_segment()

    with pytest.raises(ValueError, match="PAC.*too short"):
        extract_pac_from_precomputed(precomputed, precomputed.config)


def test_complexity_fails_when_requested_segment_is_too_short() -> None:
    n_epochs = 2
    n_times = 260
    times = np.arange(n_times, dtype=float) / 100.0
    short_mask = np.zeros(n_times, dtype=bool)
    short_mask[:50] = True
    long_mask = np.zeros(n_times, dtype=bool)
    long_mask[50:] = True
    data = np.sin(2.0 * np.pi * 10.0 * times)[None, None, :]
    data = np.repeat(data, n_epochs, axis=0)
    precomputed = PrecomputedData(
        data=data,
        times=times,
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        windows=TimeWindows(
            masks={"short": short_mask, "long": long_mask},
            ranges={"short": (0.0, 0.49), "long": (0.5, 2.59)},
            times=times,
        ),
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=data,
                analytic=data.astype(complex),
                envelope=np.abs(data),
                phase=np.zeros_like(data),
                power=data**2,
            )
        },
        config=DotConfig(
            {
                "feature_engineering": {
                    "complexity": {
                        "min_samples": 200,
                        "min_segment_sec": 1.0,
                        "mse_scale_max": 20,
                    }
                }
            }
        ),
        logger=logging.getLogger("strict-complexity-short-segment"),
        spatial_modes=["global"],
    )

    with pytest.raises(ValueError, match="Complexity.*too short"):
        extract_complexity_from_precomputed(precomputed, n_jobs=1)


class _DummyTFR:
    def __init__(self) -> None:
        self.times = np.array([0.0, 0.5, 1.0], dtype=float)
        self.data = np.ones((2, 1, 1, 3), dtype=float)
        self.comment = "dummy"

    def copy(self) -> "_DummyTFR":
        new = _DummyTFR()
        new.times = self.times.copy()
        new.data = self.data.copy()
        new.comment = self.comment
        return new

    def crop(self, tmin: float, tmax: float) -> "_DummyTFR":
        self.times = self.times[(self.times >= tmin) & (self.times <= tmax)]
        return self

    def save(self, *_args, **_kwargs) -> None:
        return None


def test_tfr_feature_crop_rejects_out_of_range_request(monkeypatch, tmp_path) -> None:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(np.ones((2, 1, 64), dtype=float), info, verbose=False)

    def fake_compute_tfr_for_subject(*_args, **_kwargs):
        return _DummyTFR(), None, [], None, None

    monkeypatch.setattr(
        "eeg_pipeline.analysis.features.api.compute_tfr_for_subject",
        fake_compute_tfr_for_subject,
    )
    ctx = SimpleNamespace(
        _original_epochs=epochs,
        epochs=epochs,
        aligned_events=pd.DataFrame(index=range(2)),
        subject="sub-01",
        task="task",
        config=DotConfig({"feature_engineering": {"save_tfr_with_sidecar": False}}),
        deriv_root=tmp_path,
        logger=logging.getLogger("strict-tfr-feature-crop"),
        windows=None,
        feature_categories=[],
        frequency_bands=None,
        tfr=None,
    )

    with pytest.raises(ValueError, match="outside available TFR range"):
        _compute_tfr_for_features(ctx, tmin=-0.5, tmax=0.5)


def test_aperiodic_line_noise_rejects_invalid_preprocessing_fallback() -> None:
    with pytest.raises(ValueError, match="preprocessing.line_freq"):
        _parse_line_noise_config(
            DotConfig(
                {
                    "preprocessing": {"line_freq": "bad"},
                    "feature_engineering": {"aperiodic": {"exclude_line_noise": True}},
                }
            )
        )


def test_aperiodic_line_noise_rejects_invalid_explicit_frequencies() -> None:
    with pytest.raises(ValueError, match="line_noise_freqs"):
        _parse_line_noise_config(
            DotConfig(
                {
                    "preprocessing": {"line_freq": 60.0},
                    "feature_engineering": {
                        "aperiodic": {
                            "exclude_line_noise": True,
                            "line_noise_freqs": [60.0, np.nan],
                        }
                    },
                }
            )
        )


def test_raw_aperiodic_fails_when_requested_segment_is_too_short(monkeypatch) -> None:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types="eeg")
    data = np.sin(2.0 * np.pi * 10.0 * np.arange(260) / 100.0)
    epochs = mne.EpochsArray(data[None, None, :].repeat(2, axis=0), info, verbose=False)
    short_mask = np.zeros(260, dtype=bool)
    short_mask[:50] = True
    long_mask = np.zeros(260, dtype=bool)
    long_mask[50:] = True
    windows = TimeWindows(
        masks={"short": short_mask, "long": long_mask},
        ranges={"short": (0.0, 0.49), "long": (0.5, 2.59)},
        times=epochs.times,
    )

    def fake_segment(*_args, **_kwargs):
        return {"fake": np.ones(2, dtype=float), "__qc__": {}}

    monkeypatch.setattr(
        "eeg_pipeline.analysis.features.aperiodic._extract_aperiodic_for_segment",
        fake_segment,
    )
    ctx = SimpleNamespace(
        config=DotConfig(
            {
                "preprocessing": {"line_freq": 60.0},
                "feature_engineering": {
                    "aperiodic": {"min_segment_sec": 1.0},
                    "spatial_modes": ["global"],
                },
                "time_frequency_analysis": {"bands": {"alpha": [8.0, 12.0]}},
            }
        ),
        epochs=epochs,
        logger=logging.getLogger("strict-raw-aperiodic-short"),
        windows=windows,
        name=None,
        spatial_modes=["global"],
        frequency_bands={"alpha": [8.0, 12.0]},
        train_mask=None,
        analysis_mode="group_stats",
    )

    with pytest.raises(ValueError, match="Aperiodic.*too short"):
        extract_aperiodic_features(ctx, ["alpha"])


def _precomputed_for_partial_strict_segments() -> PrecomputedData:
    n_epochs = 2
    n_times = 320
    times = np.arange(n_times, dtype=float) / 100.0
    short_mask = np.zeros(n_times, dtype=bool)
    short_mask[:50] = True
    long_mask = np.zeros(n_times, dtype=bool)
    long_mask[50:] = True
    analytic_epoch = np.stack(
        [
            (1.0 + 0.2 * np.sin(2.0 * np.pi * times)) * np.exp(1j * 2.0 * np.pi * 10.0 * times),
            (1.0 + 0.2 * np.cos(2.0 * np.pi * times))
            * np.exp(1j * (2.0 * np.pi * 10.0 * times + 0.5)),
        ],
        axis=0,
    )
    analytic = np.repeat(analytic_epoch[None, :, :], n_epochs, axis=0)
    return PrecomputedData(
        data=np.real(analytic),
        times=times,
        sfreq=100.0,
        ch_names=["C3", "C4"],
        picks=np.array([0, 1], dtype=int),
        windows=TimeWindows(
            masks={"short": short_mask, "long": long_mask},
            ranges={"short": (0.0, 0.49), "long": (0.5, 3.19)},
            times=times,
        ),
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=np.real(analytic),
                analytic=analytic,
                envelope=np.abs(analytic),
                phase=np.angle(analytic),
                power=np.abs(analytic) ** 2,
            )
        },
        config=DotConfig(
            {
                "preprocessing": {"line_freq": 60.0},
                "feature_engineering": {
                    "aperiodic": {
                        "min_segment_sec": 1.0,
                        "psd_method": "welch",
                        "fmin": 1.0,
                        "fmax": 40.0,
                    },
                    "connectivity": {
                        "measures": ["aec"],
                        "dynamic_enabled": False,
                        "min_segment_sec": 1.0,
                        "min_segment_samples": 20,
                    },
                    "spatial_modes": ["global"],
                },
                "time_frequency_analysis": {"bands": {"alpha": [8.0, 12.0]}},
            }
        ),
        logger=logging.getLogger("strict-partial-segments"),
        spatial_modes=["global"],
        frequency_bands={"alpha": [8.0, 12.0]},
    )


def test_precomputed_aperiodic_fails_when_requested_segment_is_too_short() -> None:
    precomputed = _precomputed_for_partial_strict_segments()

    with pytest.raises(ValueError, match="Aperiodic.*too short"):
        extract_aperiodic_from_precomputed(precomputed, ["alpha"])


def test_precomputed_connectivity_fails_when_requested_segment_is_too_short() -> None:
    precomputed = _precomputed_for_partial_strict_segments()

    with pytest.raises(ValueError, match="Connectivity.*too short"):
        extract_connectivity_from_precomputed(precomputed, bands=["alpha"])


def test_dynamic_connectivity_fails_when_requested_windows_are_insufficient() -> None:
    precomputed = _precomputed_for_partial_strict_segments()
    mask = np.ones(precomputed.data.shape[-1], dtype=bool)
    precomputed.windows = TimeWindows(
        masks={"active": mask},
        ranges={"active": (0.0, float(precomputed.times[-1]))},
        times=precomputed.times,
    )
    precomputed.config = DotConfig(
        {
            "feature_engineering": {
                "connectivity": {
                    "measures": ["aec"],
                    "dynamic_enabled": True,
                    "dynamic_measures": ["aec"],
                    "dynamic_min_windows": 3,
                    "sliding_window_len": 2.0,
                    "sliding_window_step": 2.0,
                    "min_segment_sec": 0.5,
                    "min_segment_samples": 20,
                },
                "spatial_modes": ["global"],
            },
            "time_frequency_analysis": {"bands": {"alpha": [8.0, 12.0]}},
        }
    )

    with pytest.raises(ValueError, match="dynamic.*windows"):
        extract_connectivity_from_precomputed(precomputed, bands=["alpha"])


def test_requested_roi_aggregation_requires_roi_definitions() -> None:
    with pytest.raises(ValueError, match="ROI.*definitions"):
        build_roi_map_if_needed(
            ["roi", "global"],
            ["Cz", "Pz"],
            DotConfig({}),
        )


def test_epoch_crop_rejects_reversed_and_out_of_range_windows() -> None:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(np.ones((2, 1, 100), dtype=float), info, verbose=False)

    with pytest.raises(ValueError, match="tmin.*<=.*tmax"):
        crop_epochs_to_time_range(
            epochs,
            0.5,
            0.1,
            logging.getLogger("strict-epoch-crop-reversed"),
        )

    with pytest.raises(ValueError, match="outside available data range"):
        crop_epochs_to_time_range(
            epochs,
            -0.1,
            0.5,
            logging.getLogger("strict-epoch-crop-outside"),
        )


def test_connectivity_wavelet_cycles_reject_invalid_or_unviable_requests() -> None:
    with pytest.raises(ValueError, match="n_cycles"):
        _safe_n_cycles_for_segment(
            "bad",
            np.array([8.0, 12.0], dtype=float),
            sfreq_hz=100.0,
            n_times=100,
        )

    with pytest.raises(ValueError, match="n_cycles"):
        _safe_n_cycles_for_segment(
            20.0,
            np.array([4.0, 5.0], dtype=float),
            sfreq_hz=100.0,
            n_times=30,
        )


def test_across_epoch_connectivity_replacement_fails_when_requested_segment_is_short() -> None:
    n_epochs = 2
    n_times = 40
    times = np.arange(n_times, dtype=float) / 100.0
    mask = np.zeros(n_times, dtype=bool)
    mask[:20] = True
    precomputed = PrecomputedData(
        data=np.ones((n_epochs, 2, n_times), dtype=float),
        times=times,
        sfreq=100.0,
        ch_names=["C3", "C4"],
        picks=np.array([0, 1], dtype=int),
        windows=TimeWindows(
            masks={"short": mask},
            ranges={"short": (0.0, 0.19)},
            times=times,
        ),
        band_data={},
        config=DotConfig({}),
        frequency_bands={"alpha": [8.0, 12.0]},
    )
    df = pd.DataFrame({"conn_short_alpha_global_wpli_mean": [0.0, 0.0]})

    with pytest.raises(ValueError, match="too short"):
        _apply_across_epochs_phase_estimates_inplace(
            df,
            precomputed=precomputed,
            segments=["short"],
            bands=["alpha"],
            epoch_groups={"all": np.array([0, 1], dtype=int)},
            config=DotConfig(
                {
                    "feature_engineering": {
                        "connectivity": {
                            "measures": ["wpli"],
                            "min_segment_sec": 1.0,
                        }
                    }
                }
            ),
            logger=logging.getLogger("strict-across-epoch-short"),
        )


def test_pac_requested_pairs_reject_missing_bands_and_harmonic_overlap() -> None:
    tf_bands = {
        "theta": [4.0, 8.0],
        "alpha": [8.0, 12.0],
        "gamma": [30.0, 80.0],
        "harmonic": [16.0, 24.0],
    }

    with pytest.raises(ValueError, match="missing"):
        _get_valid_pac_pairs(
            [("theta", "gamma"), ("missing", "gamma")],
            tf_bands,
            allow_harmonic_overlap=True,
            max_harm=6,
            tol_hz=1.0,
            logger=logging.getLogger("strict-pac-missing-band"),
        )

    with pytest.raises(ValueError, match="harmonic overlap"):
        _get_valid_pac_pairs(
            [("theta", "harmonic"), ("alpha", "gamma")],
            tf_bands,
            allow_harmonic_overlap=False,
            max_harm=6,
            tol_hz=1.0,
            logger=logging.getLogger("strict-pac-harmonic"),
        )


def test_precomputed_pac_fails_when_requested_pair_band_data_is_missing() -> None:
    n_epochs = 3
    n_times = 220
    times = np.arange(n_times, dtype=float) / 100.0
    phase = np.zeros((n_epochs, 1, n_times), dtype=float)
    analytic = np.exp(1j * phase)
    power = np.ones((n_epochs, 1, n_times), dtype=float)
    precomputed = PrecomputedData(
        data=np.real(analytic),
        times=times,
        sfreq=100.0,
        ch_names=["Cz"],
        picks=np.array([0], dtype=int),
        windows=TimeWindows(
            masks={"active": np.ones(n_times, dtype=bool)},
            ranges={"active": (0.0, float(times[-1]))},
            times=times,
        ),
        band_data={
            "theta": BandData(
                band="theta",
                fmin=4.0,
                fmax=8.0,
                filtered=np.real(analytic),
                analytic=analytic,
                envelope=np.ones_like(power),
                phase=phase,
                power=power,
            ),
            "gamma": BandData(
                band="gamma",
                fmin=30.0,
                fmax=80.0,
                filtered=np.real(analytic),
                analytic=analytic,
                envelope=np.ones_like(power),
                phase=phase,
                power=power,
            ),
        },
        config=DotConfig(
            {
                "feature_engineering": {
                    "pac": {
                        "method": "mvl",
                        "pairs": [["theta", "gamma"], ["alpha", "gamma"]],
                        "n_surrogates": 0,
                        "min_segment_sec": 0.0,
                        "min_cycles_at_fmin": 1.0,
                        "allow_harmonic_overlap": True,
                    },
                    "spatial_modes": ["global"],
                },
                "time_frequency_analysis": {
                    "bands": {
                        "theta": [4.0, 8.0],
                        "alpha": [8.0, 12.0],
                        "gamma": [30.0, 80.0],
                    }
                },
            }
        ),
        logger=logging.getLogger("strict-precomputed-pac-missing-band"),
        spatial_modes=["global"],
        frequency_bands={
            "theta": [4.0, 8.0],
            "alpha": [8.0, 12.0],
            "gamma": [30.0, 80.0],
        },
    )

    with pytest.raises(ValueError, match="missing precomputed"):
        extract_pac_from_precomputed(precomputed, precomputed.config)


def test_directed_connectivity_rejects_mvar_order_reduction(monkeypatch) -> None:
    n_epochs = 1
    n_times = 120
    times = np.arange(n_times, dtype=float) / 100.0
    data = np.ones((n_epochs, 2, n_times), dtype=float)
    precomputed = PrecomputedData(
        data=data,
        times=times,
        sfreq=100.0,
        ch_names=["C3", "C4"],
        picks=np.array([0, 1], dtype=int),
        windows=TimeWindows(
            masks={"active": np.ones(n_times, dtype=bool)},
            ranges={"active": (0.0, float(times[-1]))},
            times=times,
        ),
        band_data={
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.0,
                filtered=data,
                analytic=data.astype(complex),
                envelope=np.ones_like(data),
                phase=np.zeros_like(data),
                power=np.ones_like(data),
            )
        },
        config=DotConfig(
            {
                "feature_engineering": {
                    "directedconnectivity": {
                        "enable_psi": True,
                        "mvar_order": 20,
                        "min_samples_per_mvar_parameter": 10,
                        "min_segment_samples": 20,
                    }
                },
                "time_frequency_analysis": {"bands": {"alpha": [8.0, 12.0]}},
            }
        ),
        logger=logging.getLogger("strict-directed-mvar-order"),
    )

    monkeypatch.setattr(
        "eeg_pipeline.analysis.features.connectivity._compute_directed_connectivity_epoch",
        lambda *_args, **_kwargs: {"psi": np.ones((2, 2), dtype=float)},
    )

    with pytest.raises(ValueError, match="mvar_order"):
        extract_directed_connectivity_from_precomputed(precomputed, bands=["alpha"])


def test_erp_component_masks_reject_invalid_requested_components() -> None:
    times = np.linspace(0.0, 1.0, 101)

    with pytest.raises(ValueError, match="ERP component"):
        _build_component_masks(
            times,
            {
                "components": [
                    {"name": "p300", "start": 0.2, "end": 0.4},
                    {"name": "bad", "start": 0.6, "end": 0.5},
                ]
            },
        )

    with pytest.raises(ValueError, match="no samples"):
        _build_component_masks(
            times,
            {"components": [{"name": "late", "start": 1.5, "end": 2.0}]},
        )
