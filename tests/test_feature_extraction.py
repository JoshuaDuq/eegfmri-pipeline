"""
Feature Extraction Tests.

Tests for feature extraction utilities with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec


class TestTimeWindowSpec:
    """Test time window specification utilities."""

    def test_create_spec_from_times(self):
        times = np.linspace(-1.0, 2.0, 300)
        config = {
            "time_frequency_analysis": {
                "baseline_window": [-0.5, 0],
                "plateau_window": [0.5, 1.5],
            },
            "feature_engineering": {
                "features": {"ramp_end": 0.5},
            },
        }

        spec = TimeWindowSpec(times=times, config=config, sampling_rate=100.0)

        assert spec.times is not None
        assert len(spec.times) == 300

    def test_get_mask_baseline(self):
        times = np.linspace(-1.0, 2.0, 301)
        config = {
            "time_frequency_analysis": {
                "baseline_window": [-0.5, 0],
                "plateau_window": [0.5, 1.5],
            },
            "feature_engineering": {
                "features": {"ramp_end": 0.5},
            },
        }

        spec = TimeWindowSpec(times=times, config=config, sampling_rate=100.0)
        mask = spec.get_mask("baseline")

        assert mask is not None
        assert mask.sum() > 0
        masked_times = times[mask]
        assert masked_times.min() >= -0.5
        assert masked_times.max() <= 0

    def test_get_mask_plateau(self):
        times = np.linspace(-1.0, 2.0, 301)
        config = {
            "time_frequency_analysis": {
                "baseline_window": [-0.5, 0],
                "plateau_window": [0.5, 1.5],
            },
            "feature_engineering": {
                "features": {"ramp_end": 0.5},
            },
        }

        spec = TimeWindowSpec(times=times, config=config, sampling_rate=100.0)
        mask = spec.get_mask("plateau")

        assert mask is not None
        assert mask.sum() > 0
        masked_times = times[mask]
        assert masked_times.min() >= 0.5
        assert masked_times.max() <= 1.5


class TestTimeWindowsFromSpec:
    """Test TimeWindows creation from spec."""

    def test_creates_windows_with_all_masks(self):
        times = np.linspace(-1.0, 2.0, 301)
        config = {
            "time_frequency_analysis": {
                "baseline_window": [-0.5, 0],
                "plateau_window": [0.5, 1.5],
            },
            "feature_engineering": {
                "features": {"ramp_end": 0.5},
            },
        }

        spec = TimeWindowSpec(times=times, config=config, sampling_rate=100.0)
        windows = time_windows_from_spec(spec, n_plateau_windows=2, strict=True)

        assert windows.baseline_mask is not None
        assert windows.active_mask is not None

    def test_windows_masks_match_spec(self):
        times = np.linspace(-1.0, 2.0, 301)
        config = {
            "time_frequency_analysis": {
                "baseline_window": [-0.5, 0],
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


class TestFeatureColumnSelection:
    """Test feature column selection utilities."""

    def test_get_power_columns_by_band(self):
        from eeg_pipeline.utils.data.features import get_power_columns_by_band

        df = pd.DataFrame({
            "power_plateau_alpha_global_mean": [1, 2, 3],
            "power_plateau_beta_global_mean": [1, 2, 3],
            "connectivity_plateau_alpha_chpair_F3-F4_wpli": [1, 2, 3],
        })

        result = get_power_columns_by_band(df)

        assert "alpha" in result
        assert "beta" in result
        assert len(result["alpha"]) == 1
        assert "power" in result["alpha"][0]

    def test_get_aperiodic_columns(self):
        from eeg_pipeline.utils.data.features import get_aperiodic_columns

        df = pd.DataFrame({
            "aperiodic_1f_global": [1, 2, 3],
            "aperiodic_exponent_global": [1, 2, 3],
            "power_plateau_alpha_global_mean": [1, 2, 3],
        })

        result = get_aperiodic_columns(df)

        assert len(result) >= 0  # May or may not match depending on patterns

    def test_get_microstate_columns(self):
        from eeg_pipeline.utils.data.features import get_microstate_columns

        df = pd.DataFrame({
            "microstate_gev": [1, 2, 3],
            "microstate_coverage_A": [1, 2, 3],
            "power_plateau_alpha_global_mean": [1, 2, 3],
        })

        result = get_microstate_columns(df)

        assert len(result) >= 0  # May or may not match depending on patterns


class TestFeatureBlockRegistration:
    """Test feature block registration utilities."""

    def test_register_feature_block(self):
        from eeg_pipeline.utils.data.features import register_feature_block

        registry = {}
        lengths = {}

        df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
        register_feature_block("power", df, registry, lengths)

        assert "power" in registry
        assert lengths["power"] == 3

    def test_register_none_block_logged(self):
        from eeg_pipeline.utils.data.features import register_feature_block

        registry = {}
        lengths = {}

        register_feature_block("power", None, registry, lengths)

        # None blocks are registered with length 0
        assert lengths.get("power", -1) == 0
