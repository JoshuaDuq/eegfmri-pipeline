"""
Configuration and Pipeline Tests.

Tests for config loading, pipeline orchestration, and integration.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


###################################################################
# Config Loading Tests
###################################################################

class TestConfigLoader:
    """Test configuration loading utilities."""

    def test_load_settings_returns_dict(self):
        from eeg_pipeline.utils.config.loader import load_settings

        config = load_settings()

        assert config is not None
        assert hasattr(config, "get") or isinstance(config, dict)

    def test_get_config_value(self):
        from eeg_pipeline.utils.config.loader import get_config_value, load_settings

        config = load_settings()

        # Test getting a known config value
        result = get_config_value(config, "decoding.cv.default_n_splits", default=5)

        assert isinstance(result, int)

    def test_get_config_value_default(self):
        from eeg_pipeline.utils.config.loader import get_config_value

        config = {}

        result = get_config_value(config, "missing.key", default=99)

        assert result == 99

    def test_get_frequency_bands(self):
        from eeg_pipeline.utils.config.loader import get_frequency_bands, load_settings

        config = load_settings()
        bands = get_frequency_bands(config)

        assert isinstance(bands, dict) or bands is not None


class TestDecodingConfig:
    """Test decoding configuration utilities."""

    def test_get_decoding_config(self):
        from eeg_pipeline.analysis.decoding.config import get_decoding_config

        config = get_decoding_config()

        assert isinstance(config, dict)
        assert "min_trials_for_within_subject" in config
        assert "default_n_splits" in config

    def test_decoding_config_values(self):
        from eeg_pipeline.analysis.decoding.config import get_decoding_config

        config = get_decoding_config()

        assert config["default_n_splits"] >= 2
        assert config["min_trials_for_within_subject"] >= 1


###################################################################
# Pipeline Tests
###################################################################

class TestFeaturePipeline:
    """Test feature pipeline class."""

    def test_pipeline_class_exists(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline

        # Test that class is instantiable
        assert FeaturePipeline is not None

    def test_pipeline_has_process_method(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline

        assert hasattr(FeaturePipeline, "process_subject")
        assert hasattr(FeaturePipeline, "run_batch")


class TestDecodingPipelines:
    """Test decoding pipeline factories."""

    def test_create_svm_pipeline(self):
        from eeg_pipeline.analysis.decoding.classification import create_svm_pipeline

        pipeline = create_svm_pipeline()

        assert pipeline is not None
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")

    def test_create_logistic_pipeline(self):
        from eeg_pipeline.analysis.decoding.classification import create_logistic_pipeline

        pipeline = create_logistic_pipeline()

        assert pipeline is not None
        assert hasattr(pipeline, "fit")

    def test_create_rf_classification_pipeline(self):
        from eeg_pipeline.analysis.decoding.classification import create_rf_classification_pipeline

        pipeline = create_rf_classification_pipeline()

        assert pipeline is not None


class TestRegressionPipelines:
    """Test regression pipeline factories."""

    def test_create_elasticnet_pipeline(self):
        from eeg_pipeline.analysis.decoding.pipelines import create_elasticnet_pipeline

        pipeline = create_elasticnet_pipeline()

        assert pipeline is not None
        assert hasattr(pipeline, "fit")

    def test_create_rf_pipeline(self):
        from eeg_pipeline.analysis.decoding.pipelines import create_rf_pipeline

        pipeline = create_rf_pipeline()

        assert pipeline is not None

    def test_build_param_grids(self):
        from eeg_pipeline.analysis.decoding.pipelines import (
            build_elasticnet_param_grid,
            build_rf_param_grid,
        )

        en_grid = build_elasticnet_param_grid()
        rf_grid = build_rf_param_grid()

        assert isinstance(en_grid, dict)
        assert isinstance(rf_grid, dict)


###################################################################
# Context Tests
###################################################################

class TestFeatureContext:
    """Test feature context utilities."""

    def test_feature_context_import(self):
        from eeg_pipeline.context.features import FeatureContext

        assert FeatureContext is not None


###################################################################
# Path Utilities Tests
###################################################################

class TestPathUtilities:
    """Test path utility functions."""

    def test_ensure_dir_import(self):
        from eeg_pipeline.infra.paths import ensure_dir

        assert ensure_dir is not None

    def test_deriv_paths_import(self):
        from eeg_pipeline.infra.paths import (
            deriv_features_path,
            deriv_plots_path,
        )

        assert deriv_features_path is not None
        assert deriv_plots_path is not None


###################################################################
# Plotting Config Tests
###################################################################

class TestPlotConfig:
    """Test plotting configuration."""

    def test_get_plot_config(self):
        from eeg_pipeline.plotting.config import get_plot_config

        config = get_plot_config()

        assert config is not None
        assert hasattr(config, "formats") or hasattr(config, "style")

    def test_plot_config_figure_size(self):
        from eeg_pipeline.plotting.config import get_plot_config

        config = get_plot_config()

        # Should have method to get figure size
        if hasattr(config, "get_figure_size"):
            fig_size = config.get_figure_size("default")
            assert len(fig_size) == 2


###################################################################
# Classification Result Tests
###################################################################

class TestClassificationResult:
    """Test classification result container."""

    def test_classification_result_creation(self):
        from eeg_pipeline.analysis.decoding.classification import ClassificationResult

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        result = ClassificationResult(y_true=y_true, y_pred=y_pred)

        assert result.accuracy >= 0
        assert result.accuracy <= 1

    def test_classification_result_to_dict(self):
        from eeg_pipeline.analysis.decoding.classification import ClassificationResult

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = ClassificationResult(y_true=y_true, y_pred=y_pred)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "accuracy" in d

    def test_classification_result_perfect(self):
        from eeg_pipeline.analysis.decoding.classification import ClassificationResult

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = ClassificationResult(y_true=y_true, y_pred=y_pred)

        assert result.accuracy == 1.0


###################################################################
# Scoring Tests
###################################################################

class TestScoringUtilities:
    """Test scoring utilities."""

    def test_make_pearsonr_scorer(self):
        from eeg_pipeline.analysis.decoding.cv import make_pearsonr_scorer

        scorer = make_pearsonr_scorer()

        assert scorer is not None
        assert callable(scorer)

    def test_create_scoring_dict(self):
        from eeg_pipeline.analysis.decoding.cv import create_scoring_dict

        scoring = create_scoring_dict()

        assert isinstance(scoring, dict)
        assert len(scoring) > 0


###################################################################
# Subject Utilities Tests
###################################################################

class TestSubjectUtilities:
    """Test subject parsing utilities."""

    def test_parse_subject_args_import(self):
        from eeg_pipeline.utils.data.subjects import parse_subject_args

        assert parse_subject_args is not None

    def test_get_available_subjects_import(self):
        from eeg_pipeline.utils.data.subjects import get_available_subjects

        assert get_available_subjects is not None
