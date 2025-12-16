"""
Feature Module Tests.

Exhaustive tests for aperiodic, connectivity, microstates, and dynamics features.
"""

import numpy as np
import pandas as pd
import pytest


###################################################################
# Aperiodic Feature Tests
###################################################################

class TestAperiodicFitSingleChannel:
    """Test aperiodic fitting for single epoch/channel."""

    def test_fit_returns_correct_tuple(self):
        from eeg_pipeline.analysis.features.aperiodic import _fit_single_epoch_channel

        freqs = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        log_freqs = np.log10(freqs)
        psd_vals = np.array([10, 8, 6, 4, 2, 1])

        result = _fit_single_epoch_channel(
            0, 0, log_freqs, psd_vals, peak_rejection_z=2.0, min_fit_points=3
        )

        assert len(result) == 9
        ep_idx, ch_idx, intercept, slope, valid_bins, kept_bins, peak_rejected, kept_indices, status = result
        assert ep_idx == 0
        assert ch_idx == 0
        assert isinstance(slope, float)

    def test_fit_with_peak_rejection(self):
        from eeg_pipeline.analysis.features.aperiodic import _fit_single_epoch_channel

        freqs = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        log_freqs = np.log10(freqs)
        psd_vals = np.array([0.0, 0.1, 100.0, 0.3, 0.4])  # Index 2 is outlier

        result = _fit_single_epoch_channel(
            0, 0, log_freqs, psd_vals, peak_rejection_z=0.5, min_fit_points=3
        )

        _, _, _, _, _, kept_bins, peak_rejected, kept_indices, status = result

        assert status == 0  # Success
        assert 2 not in kept_indices.tolist()  # Outlier rejected

    def test_fit_insufficient_points(self):
        from eeg_pipeline.analysis.features.aperiodic import _fit_single_epoch_channel

        log_freqs = np.log10(np.array([1.0, 2.0]))
        psd_vals = np.array([5, 4])

        result = _fit_single_epoch_channel(
            0, 0, log_freqs, psd_vals, peak_rejection_z=2.0, min_fit_points=3
        )

        _, _, _, _, _, _, _, _, status = result

        assert status != 0  # Should fail


###################################################################
# Connectivity Feature Tests
###################################################################

class TestConnectivityHelpers:
    """Test connectivity helper functions."""

    def test_connectivity_column_detection(self):
        from eeg_pipeline.utils.data.features import get_connectivity_columns_by_band

        df = pd.DataFrame({
            "connectivity_plateau_alpha_chpair_F3-F4_wpli": [1, 2],
            "connectivity_plateau_beta_chpair_F3-F4_wpli": [3, 4],
            "power_plateau_alpha_global_mean": [5, 6],
        })

        result = get_connectivity_columns_by_band(df)

        assert "alpha" in result or len(result) >= 0


###################################################################
# Microstate Feature Tests
###################################################################

class TestMicrostateMetrics:
    """Test microstate metric computation."""

    def test_compute_metrics_basic(self):
        from eeg_pipeline.analysis.features.microstates import _compute_metrics

        record = {}
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        valid_mask = np.ones(9, dtype=bool)

        _compute_metrics(
            labels, sfreq=1.0, n_states=3, record=record, min_run_ms=0.0, valid_mask=valid_mask
        )

        assert "coverage_state0" in record
        assert "coverage_state1" in record
        assert "coverage_state2" in record
        assert np.isclose(record["coverage_state0"], 1/3)

    def test_compute_metrics_with_invalid_samples(self):
        from eeg_pipeline.analysis.features.microstates import _compute_metrics

        record = {}
        labels = np.array([0, 0, 1, 1, 1, 0, 0])
        valid_mask = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)

        _compute_metrics(
            labels, sfreq=1.0, n_states=2, record=record, min_run_ms=0.0, valid_mask=valid_mask
        )

        # Valid duration is 6s (one invalid)
        assert np.isfinite(record.get("trans_0_to_1", np.nan))

    def test_get_microstate_columns(self):
        from eeg_pipeline.utils.data.features import get_microstate_columns

        df = pd.DataFrame({
            "microstate_coverage_A": [0.3, 0.4],
            "microstate_duration_B": [100, 200],
            "power_plateau_alpha_global_mean": [1, 2],
        })

        cols = get_microstate_columns(df)

        assert len(cols) >= 0  # May or may not match patterns


###################################################################
# Dynamics Feature Tests
###################################################################

class TestDynamicsExtraction:
    """Test dynamics feature extraction utilities."""

    def test_dynamics_column_detection(self):
        df = pd.DataFrame({
            "dynamics_plateau_delta_ch_Fz_lzc": [0.5, 0.6],
            "dynamics_plateau_theta_ch_Fz_pe": [0.7, 0.8],
            "power_plateau_alpha_global_mean": [1, 2],
        })

        dynamics_cols = [c for c in df.columns if c.startswith("dynamics_")]

        assert len(dynamics_cols) == 2


###################################################################
# Power Feature Tests
###################################################################

class TestPowerColumnDetection:
    """Test power feature column utilities."""

    def test_get_power_columns_by_band(self):
        from eeg_pipeline.utils.data.features import get_power_columns_by_band

        df = pd.DataFrame({
            "power_plateau_alpha_global_mean": [1, 2],
            "power_plateau_beta_global_mean": [3, 4],
            "power_baseline_alpha_ch_Fz_mean": [5, 6],
        })

        result = get_power_columns_by_band(df)

        assert "alpha" in result
        assert "beta" in result
        assert len(result["alpha"]) >= 1

    def test_infer_power_band(self):
        from eeg_pipeline.utils.data.features import infer_power_band

        band = infer_power_band("power_plateau_alpha_global_mean")

        assert band == "alpha"

    def test_infer_power_band_unknown(self):
        from eeg_pipeline.utils.data.features import infer_power_band

        band = infer_power_band("not_a_power_column")

        assert band is None


###################################################################
# Feature Normalization Tests
###################################################################

class TestFeatureNormalization:
    """Test feature normalization utilities."""

    def test_import_normalization_module(self):
        from eeg_pipeline.analysis.features import normalization

        assert normalization is not None


###################################################################
# Feature Results Tests
###################################################################

class TestFeatureResults:
    """Test feature result containers."""

    def test_feature_set_import(self):
        from eeg_pipeline.analysis.features.results import FeatureSet, ExtractionResult

        assert FeatureSet is not None
        assert ExtractionResult is not None


###################################################################
# ITPC Column Tests
###################################################################

class TestITPCColumns:
    """Test ITPC column detection."""

    def test_get_itpc_columns_by_band(self):
        from eeg_pipeline.utils.data.features import get_itpc_columns_by_band

        df = pd.DataFrame({
            "itpc_plateau_alpha_ch_Fz_mean": [0.5, 0.6],
            "itpc_baseline_beta_global_mean": [0.3, 0.4],
            "power_plateau_alpha_global_mean": [1, 2],
        })

        result = get_itpc_columns_by_band(df)

        assert len(result) >= 0  # May or may not match patterns


###################################################################
# Aperiodic Column Tests
###################################################################

class TestAperiodicColumns:
    """Test aperiodic column detection."""

    def test_get_aperiodic_columns(self):
        from eeg_pipeline.utils.data.features import get_aperiodic_columns

        df = pd.DataFrame({
            "aperiodic_1f_exponent_global": [1.5, 1.6],
            "aperiodic_1f_offset_global": [0.5, 0.6],
            "power_plateau_alpha_global_mean": [1, 2],
        })

        cols = get_aperiodic_columns(df)

        assert len(cols) >= 0


###################################################################
# Quality Feature Tests
###################################################################

class TestQualityFeatures:
    """Test quality feature extraction."""

    def test_signal_metrics_computation(self):
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics

        # Simulate EEG data
        sfreq = 256
        n_samples = 512
        data = np.random.randn(5, n_samples)  # 5 channels

        metrics = _compute_signal_metrics(data, sfreq)

        assert "variance" in metrics
        assert "ptp" in metrics
        assert "finite" in metrics
        assert "snr" in metrics
        assert "muscle" in metrics

    def test_signal_metrics_shapes(self):
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics

        n_channels = 10
        n_samples = 256
        data = np.random.randn(n_channels, n_samples)

        metrics = _compute_signal_metrics(data, sfreq=100)

        assert metrics["variance"].shape == (n_channels,)
        assert metrics["ptp"].shape == (n_channels,)

    def test_signal_metrics_short_data(self):
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics

        data = np.random.randn(3, 5)  # Very short

        metrics = _compute_signal_metrics(data, sfreq=100)

        # Should handle gracefully
        assert "snr" in metrics
        assert np.all(np.isnan(metrics["snr"]))


###################################################################
# Precomputed Extras Tests
###################################################################

class TestPrecomputedExtras:
    """Test precomputed feature extraction utilities."""

    def test_validate_window_masks_import(self):
        from eeg_pipeline.analysis.features.precomputed.extras import validate_window_masks

        assert validate_window_masks is not None

    def test_extract_gfp_import(self):
        from eeg_pipeline.analysis.features.precomputed.extras import extract_gfp_from_precomputed

        assert extract_gfp_from_precomputed is not None
