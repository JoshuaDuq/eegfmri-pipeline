"""
Quality Assessment Tests.

Tests for data quality metrics and reporting utilities.
"""

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.analysis.features.quality import generate_quality_report


class TestQualityReport:
    """Test quality report generation."""

    def test_empty_dataframe_returns_status(self):
        df = pd.DataFrame()

        report = generate_quality_report(df)

        assert report["status"] == "empty"
        assert report["n_rows"] == 0

    def test_report_counts_features(self):
        df = pd.DataFrame({
            "quality_baseline_snr": [10, 20, 30],
            "quality_plateau_snr": [15, 25, 35],
            "epoch": [0, 1, 2],
        })

        report = generate_quality_report(df)

        assert report["n_rows"] == 3
        assert report["n_features"] == 2  # excludes 'epoch'

    def test_report_identifies_missing(self):
        df = pd.DataFrame({
            "feat1": [1, np.nan, 3],
            "feat2": [np.nan, np.nan, np.nan],
        })

        report = generate_quality_report(df)

        assert report["missing_fraction"]["feat1"] == pytest.approx(1/3)
        assert report["missing_fraction"]["feat2"] == pytest.approx(1.0)

    def test_report_identifies_constant_features(self):
        df = pd.DataFrame({
            "constant": [5, 5, 5, 5, 5],
            "varying": [1, 2, 3, 4, 5],
        })

        report = generate_quality_report(df)

        assert "constant" in report["constant_features"]
        assert "varying" not in report["constant_features"]

    def test_report_mean_missing_fraction(self):
        df = pd.DataFrame({
            "feat1": [1, np.nan, 3],  # 1/3 missing
            "feat2": [1, 2, 3],       # 0 missing
        })

        report = generate_quality_report(df)

        expected = (1/3 + 0) / 2
        assert report["mean_missing_fraction"] == pytest.approx(expected)


class TestQualityMetrics:
    """Test quality metric computation helpers."""

    def test_snr_calculation_logic(self):
        # Test that SNR makes sense: higher signal power = higher SNR
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics
        
        # Create data with clear signal (1-30 Hz) vs noise (50+ Hz)
        sfreq = 256
        n_samples = 512
        t = np.arange(n_samples) / sfreq

        # Pure 10 Hz signal (should have high SNR)
        signal_data = np.sin(2 * np.pi * 10 * t).reshape(1, -1)

        metrics = _compute_signal_metrics(signal_data, sfreq)

        assert "snr" in metrics
        assert "variance" in metrics
        assert "ptp" in metrics

    def test_variance_calculation(self):
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics

        # Known variance
        data = np.array([[1, 2, 3, 4, 5]])
        sfreq = 100

        metrics = _compute_signal_metrics(data, sfreq)

        assert np.isclose(metrics["variance"][0], np.var(data))

    def test_ptp_calculation(self):
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics

        data = np.array([[0, 10, 5]])
        sfreq = 100

        metrics = _compute_signal_metrics(data, sfreq)

        assert metrics["ptp"][0] == 10


class TestFiniteFraction:
    """Test finite data fraction calculation."""

    def test_all_finite(self):
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics

        data = np.array([[1, 2, 3, 4, 5]])
        sfreq = 100

        metrics = _compute_signal_metrics(data, sfreq)

        assert metrics["finite"][0] == 1.0

    def test_some_nan(self):
        from eeg_pipeline.analysis.features.quality import _compute_signal_metrics

        data = np.array([[1, np.nan, 3, np.nan, 5]])
        sfreq = 100

        metrics = _compute_signal_metrics(data, sfreq)

        assert metrics["finite"][0] == pytest.approx(0.6)
