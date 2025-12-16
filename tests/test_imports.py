"""
Import Verification Tests.

Ensures all major modules import without error.
This is critical for catching circular imports and missing dependencies.
"""

import pytest


class TestCoreImports:
    """Test that core pipeline modules import cleanly."""

    def test_import_types(self):
        from eeg_pipeline.types import PrecomputedData, BandData, TimeWindows
        assert PrecomputedData is not None
        assert BandData is not None
        assert TimeWindows is not None

    def test_import_feature_extraction_api(self):
        from eeg_pipeline.analysis.features.api import (
            extract_precomputed_features,
            extract_fmri_prediction_features,
            extract_all_features,
        )
        assert extract_precomputed_features is not None
        assert extract_fmri_prediction_features is not None
        assert extract_all_features is not None

    def test_import_pipeline_class(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline
        assert FeaturePipeline is not None

    def test_import_naming_schema(self):
        from eeg_pipeline.domain.features.naming import NamingSchema
        assert NamingSchema is not None
        assert hasattr(NamingSchema, "build")
        assert hasattr(NamingSchema, "parse")

    def test_import_feature_registry(self):
        from eeg_pipeline.domain.features.registry import (
            FeatureRegistry,
            classify_feature,
        )
        assert FeatureRegistry is not None
        assert classify_feature is not None


class TestAnalysisImports:
    """Test that analysis modules import cleanly."""

    def test_import_behavior_analysis(self):
        from eeg_pipeline.analysis.behavior.feature_correlator import (
            FeatureBehaviorCorrelator,
            CorrelationConfig,
        )
        assert FeatureBehaviorCorrelator is not None

    def test_import_decoding_modules(self):
        from eeg_pipeline.analysis.decoding.cv import (
            create_loso_folds,
            create_stratified_cv_by_binned_targets,
            safe_pearsonr,
        )
        assert create_loso_folds is not None
        assert safe_pearsonr is not None

    def test_import_decoding_classification(self):
        from eeg_pipeline.analysis.decoding.classification import (
            decode_pain_binary,
            ClassificationResult,
        )
        assert decode_pain_binary is not None
        assert ClassificationResult is not None

    def test_import_precomputed_extractors(self):
        from eeg_pipeline.analysis.features.precomputed.extras import (
            extract_gfp_from_precomputed,
            extract_temporal_features_from_precomputed,
            extract_band_ratios_from_precomputed,
            extract_asymmetry_from_precomputed,
            extract_roi_features_from_precomputed,
            validate_window_masks,
        )
        assert extract_gfp_from_precomputed is not None

    def test_import_spectral_extractors(self):
        from eeg_pipeline.analysis.features.precomputed.spectral import (
            extract_power_from_precomputed,
            extract_spectral_extras_from_precomputed,
            extract_segment_power_from_precomputed,
        )
        assert extract_power_from_precomputed is not None

    def test_import_quality_features(self):
        from eeg_pipeline.analysis.features.quality import (
            extract_quality_features,
            generate_quality_report,
            compute_trial_quality_metrics,
        )
        assert extract_quality_features is not None


class TestPlottingImports:
    """Test that plotting modules import cleanly."""

    def test_import_plot_config(self):
        from eeg_pipeline.plotting.config import get_plot_config
        assert get_plot_config is not None

    def test_import_decoding_plotting(self):
        from eeg_pipeline.plotting.decoding.helpers import (
            despine,
            calculate_axis_limits,
            plot_residual_diagnostics,
        )
        assert despine is not None
        assert plot_residual_diagnostics is not None

    def test_import_behavioral_scatter(self):
        from eeg_pipeline.plotting.behavioral.scatter import (
            plot_power_roi_scatter,
        )
        assert plot_power_roi_scatter is not None


class TestUtilityImports:
    """Test that utility modules import cleanly."""

    def test_import_config_loader(self):
        from eeg_pipeline.utils.config.loader import (
            load_settings,
            get_config_value,
            get_frequency_bands,
        )
        assert load_settings is not None

    def test_import_stats_utilities(self):
        from eeg_pipeline.utils.analysis.stats import (
            extract_finite_mask,
        )
        assert extract_finite_mask is not None

    def test_import_feature_io(self):
        from eeg_pipeline.utils.data.feature_io import (
            load_feature_bundle,
            save_all_features,
        )
        assert load_feature_bundle is not None

    def test_import_windowing(self):
        from eeg_pipeline.utils.analysis.windowing import (
            TimeWindowSpec,
            time_windows_from_spec,
        )
        assert TimeWindowSpec is not None
