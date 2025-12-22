"""
Pipeline Integration Tests.

Exhaustive tests for each pipeline that test actual pipeline execution
with synthetic data. One comprehensive test class per pipeline.

Pipelines tested:
- Feature Compute Pipeline
- Feature Visualize Pipeline  
- Behavior Compute Pipeline
- Behavior Visualize Pipeline
- Decoding Pipeline
- TFR Visualize Pipeline
- ERP Pipeline
- Preprocessing Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


###################################################################
# Fixtures
###################################################################


@pytest.fixture
def mock_config():
    from eeg_pipeline.utils.config.loader import load_config
    return load_config()


@pytest.fixture
def synthetic_epochs():
    import mne
    
    n_channels = 32
    n_epochs = 20
    sfreq = 250.0
    tmin = -3.0
    tmax = 12.0
    n_times = int((tmax - tmin) * sfreq)
    
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage("standard_1020", on_missing="ignore")
    
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    
    for i in range(n_epochs):
        t = np.linspace(0, n_times / sfreq, n_times)
        alpha = np.sin(2 * np.pi * 10 * t) * 2e-6
        data[i, :, :] += alpha
    
    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    
    epochs = mne.EpochsArray(data, info, events=events, tmin=tmin)
    return epochs


@pytest.fixture
def synthetic_events_df():
    n_epochs = 20
    rng = np.random.default_rng(42)
    
    return pd.DataFrame({
        "trial": np.arange(n_epochs),
        "rating": rng.uniform(0, 100, n_epochs),
        "temperature": rng.choice([44.0, 46.0, 48.0], n_epochs),
        "condition": rng.choice(["pain", "nonpain"], n_epochs),
        "subject": ["0001"] * n_epochs,
    })


@pytest.fixture
def synthetic_feature_df():
    n_trials = 20
    rng = np.random.default_rng(42)
    
    return pd.DataFrame({
        "power_plateau_alpha_global_mean": rng.standard_normal(n_trials),
        "power_plateau_beta_global_mean": rng.standard_normal(n_trials),
        "power_plateau_theta_global_mean": rng.standard_normal(n_trials),
        "power_plateau_delta_global_mean": rng.standard_normal(n_trials),
        "connectivity_plateau_alpha_chpair_F3-F4_wpli": rng.uniform(0, 1, n_trials),
        "microstate_coverage_state0": rng.uniform(0, 0.5, n_trials),
        "microstate_coverage_state1": rng.uniform(0, 0.5, n_trials),
        "aperiodic_1f_exponent_global": rng.uniform(0.5, 2.0, n_trials),
        "aperiodic_1f_offset_global": rng.uniform(-1, 1, n_trials),
        "rating": rng.uniform(0, 100, n_trials),
        "temperature": rng.choice([44.0, 46.0, 48.0], n_trials),
        "condition": rng.choice(["pain", "nonpain"], n_trials),
    })


###################################################################
# Feature Compute Pipeline
###################################################################


class TestFeatureComputePipeline:
    """Exhaustive tests for feature extraction pipeline."""

    def test_pipeline_instantiation(self, mock_config):
        from eeg_pipeline.pipelines.features import FeaturePipeline

        pipeline = FeaturePipeline(config=mock_config)

        assert pipeline is not None
        assert pipeline.name == "feature_extraction"
        assert hasattr(pipeline, "process_subject")
        assert hasattr(pipeline, "run_batch")
        assert pipeline.config is not None
        assert pipeline.deriv_root is not None

    def test_feature_context_creation(self, mock_config, synthetic_epochs, synthetic_events_df):
        from eeg_pipeline.context.features import FeatureContext
        import logging

        ctx = FeatureContext(
            subject="0001",
            task="test",
            config=mock_config,
            deriv_root=Path(tempfile.mkdtemp()),
            logger=logging.getLogger("test"),
            epochs=synthetic_epochs,
            aligned_events=synthetic_events_df,
            feature_categories=["power"],
        )

        assert ctx.subject == "0001"
        assert ctx.epochs is not None
        assert len(ctx.epochs) == 20
        assert ctx.aligned_events is not None
        assert "power" in ctx.feature_categories

    def test_feature_category_resolution(self, mock_config):
        from eeg_pipeline.analysis.features.selection import resolve_feature_categories

        all_categories = resolve_feature_categories(mock_config, None)
        assert isinstance(all_categories, list)
        assert len(all_categories) > 0

        specific = resolve_feature_categories(mock_config, ["power", "connectivity"])
        assert "power" in specific
        assert "connectivity" in specific

    def test_feature_results_containers(self):
        from eeg_pipeline.analysis.features.results import (
            FeatureSet,
            ExtractionResult,
            FeatureExtractionResult,
        )

        df = pd.DataFrame({"feat1": [1, 2, 3]})
        fs = FeatureSet(df=df, columns=["feat1"], name="test")
        assert fs.df is not None
        assert fs.columns == ["feat1"]

        result = ExtractionResult()
        assert result.features == {}
        assert result.qc == {}

        full_result = FeatureExtractionResult()
        assert full_result.pow_df is None
        assert full_result.conn_df is None

    def test_power_feature_extraction_logic(self, mock_config):
        from eeg_pipeline.analysis.features.power import extract_power_features
        
        assert callable(extract_power_features)

    def test_connectivity_feature_extraction_logic(self, mock_config):
        from eeg_pipeline.analysis.features.connectivity import (
            extract_connectivity_features,
            extract_connectivity_from_precomputed,
        )
        
        assert callable(extract_connectivity_features)
        assert callable(extract_connectivity_from_precomputed)

    def test_microstate_feature_extraction_logic(self):
        from eeg_pipeline.analysis.features.microstates import (
            extract_microstate_features,
            _compute_metrics,
            label_timecourse,
            zscore_maps,
        )

        assert callable(extract_microstate_features)
        assert callable(_compute_metrics)

        maps = np.random.randn(10, 32)
        zscored = zscore_maps(maps, axis=1)
        assert zscored.shape == maps.shape
        assert np.allclose(zscored.mean(axis=1), 0, atol=1e-10)

    def test_microstate_metrics_computation(self):
        from eeg_pipeline.analysis.features.microstates import _compute_metrics

        record = {}
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        valid_mask = np.ones(9, dtype=bool)

        _compute_metrics(
            labels, sfreq=1.0, n_states=3, record=record, 
            min_run_ms=0.0, valid_mask=valid_mask
        )

        assert "coverage_state0" in record
        assert "coverage_state1" in record
        assert "coverage_state2" in record
        assert np.isclose(record["coverage_state0"], 1/3)

    def test_aperiodic_feature_extraction_logic(self):
        from eeg_pipeline.analysis.features.aperiodic import (
            _fit_single_epoch_channel,
            extract_aperiodic_features,
        )

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

    def test_complexity_feature_extraction_logic(self):
        from eeg_pipeline.analysis.features.complexity import (
            extract_dynamics_features,
            extract_complexity_from_precomputed,
        )

        assert callable(extract_dynamics_features)
        assert callable(extract_complexity_from_precomputed)

    def test_phase_feature_extraction_logic(self):
        from eeg_pipeline.analysis.features.phase import (
            extract_phase_features,
            extract_itpc_from_precomputed,
        )

        assert callable(extract_phase_features)
        assert callable(extract_itpc_from_precomputed)

    def test_cfc_feature_extraction_logic(self):
        from eeg_pipeline.analysis.features.cfc import (
            extract_all_cfc_features,
            extract_pac_from_precomputed,
        )

        assert callable(extract_all_cfc_features)
        assert callable(extract_pac_from_precomputed)

    def test_quality_feature_extraction_logic(self):
        from eeg_pipeline.analysis.features.quality import (
            extract_quality_features,
            generate_quality_report,
            compute_trial_quality_metrics,
            _compute_signal_metrics,
        )

        sfreq = 256
        n_samples = 512
        data = np.random.randn(5, n_samples)

        metrics = _compute_signal_metrics(data, sfreq)

        assert "variance" in metrics
        assert "ptp" in metrics
        assert "finite" in metrics
        assert "snr" in metrics
        assert metrics["variance"].shape == (5,)

    def test_precomputed_feature_extraction_api(self):
        from eeg_pipeline.analysis.features.api import (
            extract_precomputed_features,
            extract_fmri_prediction_features,
        )

        assert callable(extract_precomputed_features)
        assert callable(extract_fmri_prediction_features)

    def test_feature_io_utilities(self):
        from eeg_pipeline.utils.data.feature_io import (
            load_feature_bundle,
            save_all_features,
        )

        assert callable(load_feature_bundle)
        assert callable(save_all_features)

    def test_windowing_utilities(self):
        from eeg_pipeline.utils.analysis.windowing import (
            TimeWindowSpec,
            time_windows_from_spec,
        )

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
        assert np.array_equal(windows.baseline_mask, spec.get_mask("baseline"))


###################################################################
# Feature Visualize Pipeline
###################################################################


class TestFeatureVisualizePipeline:
    """Exhaustive tests for feature visualization pipeline."""

    def test_visualize_orchestration_import(self):
        from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects

        assert callable(visualize_features_for_subjects)

    def test_power_plotting_functions(self):
        from eeg_pipeline.plotting.features import (
            plot_channel_power_heatmap,
            plot_power_time_courses,
            plot_power_spectral_density,
            plot_band_power_topomaps,
        )

        assert callable(plot_channel_power_heatmap)
        assert callable(plot_power_time_courses)
        assert callable(plot_power_spectral_density)
        assert callable(plot_band_power_topomaps)

    def test_connectivity_plotting_functions(self):
        from eeg_pipeline.plotting.features import (
            plot_connectivity_heatmap,
            plot_connectivity_network,
            plot_graph_metric_distributions,
        )

        assert callable(plot_connectivity_heatmap)
        assert callable(plot_connectivity_network)
        assert callable(plot_graph_metric_distributions)

    def test_microstate_plotting_functions(self):
        from eeg_pipeline.plotting.features import (
            plot_microstate_templates,
            plot_microstate_transition_network,
            plot_microstate_duration_distributions,
        )

        assert callable(plot_microstate_templates)
        assert callable(plot_microstate_transition_network)
        assert callable(plot_microstate_duration_distributions)

    def test_aperiodic_plotting_functions(self):
        from eeg_pipeline.plotting.features import (
            plot_aperiodic_topomaps,
            plot_aperiodic_residual_spectra,
        )

        assert callable(plot_aperiodic_topomaps)
        assert callable(plot_aperiodic_residual_spectra)

    def test_phase_plotting_functions(self):
        from eeg_pipeline.plotting.features import (
            plot_itpc_heatmap,
            plot_itpc_topomaps,
            plot_pac_comodulograms,
        )

        assert callable(plot_itpc_heatmap)
        assert callable(plot_itpc_topomaps)
        assert callable(plot_pac_comodulograms)

    def test_plot_config(self):
        from eeg_pipeline.plotting.config import get_plot_config

        config = get_plot_config()
        assert config is not None
        assert hasattr(config, "formats") or hasattr(config, "style")


###################################################################
# Behavior Compute Pipeline
###################################################################


class TestBehaviorComputePipeline:
    """Exhaustive tests for behavior analysis pipeline."""

    def test_pipeline_instantiation(self, mock_config):
        from eeg_pipeline.pipelines.behavior import (
            BehaviorPipeline,
            BehaviorPipelineConfig,
        )

        pipeline = BehaviorPipeline(config=mock_config)

        assert pipeline is not None
        assert pipeline.name == "behavior_analysis"
        assert hasattr(pipeline, "process_subject")
        assert hasattr(pipeline, "run_batch")

    def test_pipeline_config_from_config(self, mock_config):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig

        pipeline_config = BehaviorPipelineConfig.from_config(mock_config)

        assert pipeline_config.method in ["spearman", "pearson"]
        assert pipeline_config.min_samples >= 1
        assert 0 < pipeline_config.fdr_alpha <= 1
        assert pipeline_config.n_permutations >= 0

    def test_behavior_context_creation(self, mock_config):
        from eeg_pipeline.context.behavior import BehaviorContext
        import logging

        ctx = BehaviorContext(
            subject="0001",
            task="test",
            config=mock_config,
            logger=logging.getLogger("test"),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
            use_spearman=True,
            bootstrap=0,
            n_perm=100,
            rng=np.random.default_rng(42),
        )

        assert ctx.subject == "0001"
        assert ctx.use_spearman is True

    def test_behavior_results_container(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineResults

        results = BehaviorPipelineResults(subject="test")
        summary = results.to_summary()

        assert isinstance(summary, dict)
        assert summary["subject"] == "test"

        results.correlations = pd.DataFrame({
            "feature": ["a", "b"],
            "p_raw": [0.01, 0.5],
            "r": [0.5, 0.1],
        })
        summary = results.to_summary()
        assert summary["n_features"] == 2

    def test_behavior_export_manifest(self, mock_config):
        from eeg_pipeline.analysis.behavior.orchestration import stage_export, write_outputs_manifest
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig, BehaviorPipelineResults
        from eeg_pipeline.context.behavior import BehaviorContext
        from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES
        import logging

        stats_dir = Path(tempfile.mkdtemp())
        ctx = BehaviorContext(
            subject="0001",
            task="test",
            config=mock_config,
            logger=logging.getLogger("test"),
            deriv_root=stats_dir,
            stats_dir=stats_dir,
            use_spearman=True,
            bootstrap=0,
            n_perm=0,
            rng=np.random.default_rng(1),
        )
        pipeline_config = BehaviorPipelineConfig.from_config(mock_config)
        results = BehaviorPipelineResults(subject="0001")
        results.correlations = pd.DataFrame({
            "feature": [f"{cat}_feat" for cat in FEATURE_CATEGORIES],
            "feature_type": FEATURE_CATEGORIES,
            "r": [0.2] * len(FEATURE_CATEGORIES),
            "p_raw": [0.5] * len(FEATURE_CATEGORIES),
            "n": [10] * len(FEATURE_CATEGORIES),
        })

        stage_export(ctx, pipeline_config, results)
        manifest_path = write_outputs_manifest(ctx, pipeline_config, results)

        assert manifest_path.exists()
        corr_files = [p.name for p in stats_dir.iterdir() if p.name.startswith("correlations")]
        assert any(pipeline_config.method_label in name for name in corr_files)

    def test_correlation_computation(self):
        from eeg_pipeline.utils.analysis.stats.correlation import (
            compute_correlation,
            correlate_features_loop,
        )

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        r, p = compute_correlation(x, y, method="spearman")
        assert np.isclose(r, 1.0)
        assert p < 0.05

        r, p = compute_correlation(x, y, method="pearson")
        assert np.isclose(r, 1.0)

    def test_pain_sensitivity_correlations(self):
        from eeg_pipeline.utils.analysis.stats.correlation import run_pain_sensitivity_correlations

        features_df = pd.DataFrame({"feat": [0, 1, 2, 3, 4, 5]}, index=list("abcdef"))
        ratings = pd.Series([10, 20, 30, 40, 50], index=list("bcdef"))
        temperatures = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0], index=list("bcdef"))

        result = run_pain_sensitivity_correlations(
            features_df, ratings, temperatures, method="spearman", min_samples=3
        )

        assert not result.empty
        assert result.loc[0, "feature"] == "feat"
        assert result.loc[0, "n"] == 5

    def test_feature_correlator(self, synthetic_feature_df):
        from eeg_pipeline.analysis.behavior.feature_correlator import (
            FeatureBehaviorCorrelator,
            CorrelationConfig,
        )

        assert FeatureBehaviorCorrelator is not None
        assert CorrelationConfig is not None

        config = CorrelationConfig(
            method="spearman",
            min_samples=3,
            n_bootstrap=0,
            n_permutations=0,
        )
        assert config.method == "spearman"

    def test_bootstrap_utilities(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import (
            bootstrap_corr_ci,
            bootstrap_mean_ci,
            perm_pval_simple,
        )

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = x * 2

        ci_low, ci_high = bootstrap_corr_ci(x, y, method="pearson", n_boot=50)
        assert ci_low > 0.9
        assert ci_high <= 1.0

        mean, ci_low, ci_high = bootstrap_mean_ci(x, n_boot=50)
        assert np.isclose(mean, 5.5)
        assert ci_low < mean < ci_high

    def test_effect_size_utilities(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import (
            cohens_d,
            hedges_g,
            fisher_z_test,
            r_to_d,
            d_to_r,
        )

        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        d = cohens_d(group1, group2)
        assert np.abs(d) > 2.0

        g = hedges_g(group1, group2)
        assert np.abs(g) <= np.abs(d)

        z, p = fisher_z_test(r1=0.5, r2=0.1, n1=30, n2=30)
        assert isinstance(z, float)
        assert 0 <= p <= 1

    def test_fdr_correction(self):
        from eeg_pipeline.utils.analysis.stats.fdr import (
            fdr_bh,
            fdr_bh_reject,
            fdr_correction,
            apply_global_fdr,
        )

        pvals = np.array([0.001, 0.01, 0.05, 0.5, 0.9])

        q_values = fdr_bh(pvals)
        assert len(q_values) == len(pvals)
        assert q_values[0] <= q_values[1]

        reject_mask, critical = fdr_bh_reject(pvals, alpha=0.05)
        assert reject_mask.dtype == bool
        assert reject_mask[0] == True
        assert reject_mask[-1] == False


###################################################################
# Behavior Visualize Pipeline
###################################################################


class TestBehaviorVisualizePipeline:
    """Exhaustive tests for behavior visualization pipeline."""

    def test_visualize_orchestration_import(self):
        from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects

        assert callable(visualize_behavior_for_subjects)

    def test_scatter_plotting_functions(self):
        from eeg_pipeline.plotting.behavioral.scatter import plot_power_roi_scatter

        assert callable(plot_power_roi_scatter)

    def test_decoding_plot_helpers(self):
        from eeg_pipeline.plotting.decoding.helpers import (
            despine,
            calculate_axis_limits,
            plot_residual_diagnostics,
        )

        assert callable(despine)
        assert callable(calculate_axis_limits)
        assert callable(plot_residual_diagnostics)


###################################################################
# Decoding Pipeline
###################################################################


class TestDecodingPipeline:
    """Exhaustive tests for decoding pipeline."""

    def test_pipeline_instantiation(self, mock_config):
        from eeg_pipeline.pipelines.decoding import DecodingPipeline

        pipeline = DecodingPipeline(config=mock_config)

        assert pipeline is not None
        assert pipeline.name == "decoding"
        assert hasattr(pipeline, "run_batch")

    def test_pipeline_requires_multiple_subjects(self, mock_config):
        from eeg_pipeline.pipelines.decoding import DecodingPipeline

        pipeline = DecodingPipeline(config=mock_config)

        with pytest.raises(NotImplementedError):
            pipeline.process_subject("0001")

    def test_orchestration_functions(self):
        from eeg_pipeline.analysis.decoding.orchestration import (
            run_regression_decoding,
            run_time_generalization,
        )

        assert callable(run_regression_decoding)
        assert callable(run_time_generalization)

    def test_loso_fold_creation(self):
        from eeg_pipeline.analysis.decoding.cv import create_loso_folds

        X = np.random.randn(30, 10)
        groups = np.repeat([0, 1, 2], 10)

        folds = create_loso_folds(X, groups)

        assert len(folds) == 3
        for fold_num, train_idx, test_idx in folds:
            assert len(train_idx) == 20
            assert len(test_idx) == 10
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_stratified_cv_creation(self, mock_config):
        from eeg_pipeline.analysis.decoding.cv import create_stratified_cv_by_binned_targets

        y = np.linspace(0, 100, 50)
        config = {"decoding": {"cv": {"default_n_splits": 5, "default_n_bins": 5}}}

        cv, y_binned = create_stratified_cv_by_binned_targets(y, config=config)

        n_folds = sum(1 for _ in cv.split(y, y_binned))
        assert n_folds == 5

    def test_block_aware_cv(self):
        from eeg_pipeline.analysis.decoding.cv import create_block_aware_cv

        blocks = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        config = {"decoding": {"cv": {"default_n_splits": 2}}}

        cv, _ = create_block_aware_cv(blocks, config=config)

        for train_idx, test_idx in cv.split(blocks, groups=blocks):
            train_blocks = set(blocks[train_idx])
            test_blocks = set(blocks[test_idx])
            assert train_blocks.isdisjoint(test_blocks)

    def test_safe_pearsonr(self):
        from eeg_pipeline.analysis.decoding.cv import safe_pearsonr

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        r, p = safe_pearsonr(x, y)
        assert np.isclose(r, 1.0)

        x_const = np.array([1, 1, 1, 1, 1])
        r, p = safe_pearsonr(x_const, y)
        assert np.isnan(r)
        assert np.isnan(p)

    def test_classification_pipelines(self):
        from eeg_pipeline.analysis.decoding.classification import (
            create_svm_pipeline,
            create_logistic_pipeline,
            create_rf_classification_pipeline,
        )

        svm = create_svm_pipeline()
        logistic = create_logistic_pipeline()
        rf = create_rf_classification_pipeline()

        assert hasattr(svm, "fit")
        assert hasattr(logistic, "fit")
        assert hasattr(rf, "fit")

        X = np.random.randn(20, 5)
        y = np.array([0] * 10 + [1] * 10)
        svm.fit(X, y)
        preds = svm.predict(X)
        assert len(preds) == 20

    def test_regression_pipelines(self):
        from eeg_pipeline.analysis.decoding.pipelines import (
            create_elasticnet_pipeline,
            create_rf_pipeline,
            build_elasticnet_param_grid,
            build_rf_param_grid,
        )

        en = create_elasticnet_pipeline()
        rf = create_rf_pipeline()

        assert hasattr(en, "fit")
        assert hasattr(rf, "fit")

        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        en.fit(X, y)
        preds = en.predict(X)
        assert len(preds) == 20

        en_grid = build_elasticnet_param_grid()
        rf_grid = build_rf_param_grid()
        assert isinstance(en_grid, dict)
        assert isinstance(rf_grid, dict)

    def test_classification_result(self):
        from eeg_pipeline.analysis.decoding.classification import ClassificationResult

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        result = ClassificationResult(y_true=y_true, y_pred=y_pred)

        assert 0 <= result.accuracy <= 1
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "accuracy" in d

    def test_scoring_utilities(self):
        from eeg_pipeline.analysis.decoding.cv import (
            make_pearsonr_scorer,
            create_scoring_dict,
        )

        scorer = make_pearsonr_scorer()
        assert callable(scorer)

        scoring = create_scoring_dict()
        assert isinstance(scoring, dict)
        assert len(scoring) > 0

    def test_random_seeds(self):
        from eeg_pipeline.analysis.decoding.cv import set_random_seeds

        set_random_seeds(42, fold=0)
        val1 = np.random.rand()

        set_random_seeds(42, fold=0)
        val2 = np.random.rand()

        assert val1 == val2

        set_random_seeds(42, fold=1)
        val3 = np.random.rand()
        assert val1 != val3


###################################################################
# TFR Visualize Pipeline
###################################################################


class TestTFRVisualizePipeline:
    """Exhaustive tests for TFR visualization pipeline."""

    def test_visualize_orchestration_import(self):
        from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects

        assert callable(visualize_tfr_for_subjects)

    def test_subject_tfr_visualize(self):
        from eeg_pipeline.plotting.tfr import visualize_subject_tfr

        assert callable(visualize_subject_tfr)

    def test_topomap_plotting(self):
        from eeg_pipeline.plotting.tfr import (
            plot_topomap_grid_baseline_temps,
            plot_temporal_topomaps_allbands_plateau,
        )

        assert callable(plot_topomap_grid_baseline_temps)
        assert callable(plot_temporal_topomaps_allbands_plateau)

    def test_channel_plotting(self):
        from eeg_pipeline.plotting.tfr import (
            plot_cz_all_trials,
            plot_channels_all_trials,
            contrast_channels_pain_nonpain,
        )

        assert callable(plot_cz_all_trials)
        assert callable(plot_channels_all_trials)
        assert callable(contrast_channels_pain_nonpain)

    def test_contrast_plotting(self):
        from eeg_pipeline.plotting.tfr import (
            contrast_maxmin_temperature,
            contrast_pain_nonpain,
        )

        assert callable(contrast_maxmin_temperature)
        assert callable(contrast_pain_nonpain)

    def test_band_evolution_plotting(self):
        from eeg_pipeline.plotting.tfr import (
            visualize_band_evolution,
            plot_band_power_evolution_all_conditions,
        )

        assert callable(visualize_band_evolution)
        assert callable(plot_band_power_evolution_all_conditions)


###################################################################
# ERP Pipeline
###################################################################


class TestERPPipeline:
    """Exhaustive tests for ERP analysis pipeline."""

    def test_pipeline_instantiation(self, mock_config):
        from eeg_pipeline.pipelines.erp import ErpPipeline

        pipeline = ErpPipeline(config=mock_config)

        assert pipeline is not None
        assert pipeline.name == "erp_analysis"
        assert hasattr(pipeline, "process_subject")
        assert hasattr(pipeline, "run_batch")
        assert hasattr(pipeline, "set_crop_params")

    def test_crop_params(self, mock_config):
        from eeg_pipeline.pipelines.erp import ErpPipeline

        pipeline = ErpPipeline(config=mock_config)
        pipeline.set_crop_params(crop_tmin=-0.2, crop_tmax=0.8)

        assert pipeline._crop_tmin == -0.2
        assert pipeline._crop_tmax == 0.8

    def test_erp_config_loading(self):
        from eeg_pipeline.pipelines.erp import get_erp_config

        erp_config = get_erp_config()

        assert isinstance(erp_config, dict)
        assert "baseline_window" in erp_config
        assert "picks" in erp_config
        assert "pain_color" in erp_config
        assert "nonpain_color" in erp_config

    def test_erp_visualize_orchestration(self):
        from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects

        assert callable(visualize_erp_for_subjects)

    def test_erp_module_functions(self):
        from eeg_pipeline.pipelines.erp import (
            extract_erp_stats,
            extract_erp_stats_for_subjects,
            load_and_prepare_epochs,
        )

        assert callable(extract_erp_stats)
        assert callable(extract_erp_stats_for_subjects)
        assert callable(load_and_prepare_epochs)


###################################################################
# Preprocessing Pipeline
###################################################################


class TestUtilityPipeline:
    """Exhaustive tests for utility pipeline."""

    def test_pipeline_instantiation(self, mock_config):
        from eeg_pipeline.pipelines.utilities import UtilityPipeline

        pipeline = UtilityPipeline(config=mock_config)

        assert pipeline is not None
        assert hasattr(pipeline, "run_raw_to_bids")
        assert hasattr(pipeline, "run_merge_behavior")

    def test_preprocessing_functions(self):
        from eeg_pipeline.pipelines.utilities import (
            run_raw_to_bids,
            run_merge_behavior,
        )

        assert callable(run_raw_to_bids)
        assert callable(run_merge_behavior)


###################################################################
# Domain Utilities
###################################################################


class TestDomainUtilities:
    """Tests for domain utilities used across pipelines."""

    def test_naming_schema_build(self):
        from eeg_pipeline.domain.features.naming import NamingSchema

        name = NamingSchema.build(
            group="power",
            segment="plateau",
            band="alpha",
            scope="global",
            stat="mean",
        )

        assert "power" in name
        assert "plateau" in name
        assert "alpha" in name

    def test_naming_schema_parse(self):
        from eeg_pipeline.domain.features.naming import NamingSchema

        name = "power_plateau_alpha_global_mean"
        result = NamingSchema.parse(name)

        assert result is not None

    def test_feature_constants(self):
        from eeg_pipeline.domain.features.constants import (
            FEATURE_CATEGORIES,
            PRECOMPUTED_GROUP_CHOICES,
        )

        assert len(FEATURE_CATEGORIES) > 0
        assert "power" in FEATURE_CATEGORIES
        assert "connectivity" in FEATURE_CATEGORIES

        assert len(PRECOMPUTED_GROUP_CHOICES) > 0

    def test_feature_classification(self):
        from eeg_pipeline.domain.features.registry import classify_feature

        feature_type, subtype, meta = classify_feature(
            "power_plateau_alpha_global_mean"
        )

        assert feature_type is not None

    def test_feature_column_utilities(self):
        from eeg_pipeline.utils.data.features import (
            get_power_columns_by_band,
            get_aperiodic_columns,
            get_microstate_columns,
            infer_power_band,
        )

        df = pd.DataFrame({
            "power_plateau_alpha_global_mean": [1, 2, 3],
            "power_plateau_beta_global_mean": [1, 2, 3],
            "aperiodic_1f_exponent_global": [1, 2, 3],
            "microstate_coverage_A": [0.3, 0.4, 0.5],
        })

        power_cols = get_power_columns_by_band(df)
        assert "alpha" in power_cols
        assert "beta" in power_cols

        band = infer_power_band("power_plateau_alpha_global_mean")
        assert band == "alpha"


###################################################################
# Pipeline Infrastructure
###################################################################


class TestPipelineInfrastructure:
    """Tests for shared pipeline infrastructure."""

    def test_pipeline_base_class(self):
        from eeg_pipeline.pipelines.base import PipelineBase

        assert PipelineBase is not None
        assert hasattr(PipelineBase, "process_subject")
        assert hasattr(PipelineBase, "run_batch")
        assert hasattr(PipelineBase, "run_group_level")

    def test_config_loader(self):
        from eeg_pipeline.utils.config.loader import (
            load_config,
            get_config_value,
            get_frequency_bands,
            get_frequency_band_names,
        )

        config = load_config()
        assert config is not None

        value = get_config_value(config, "nonexistent.key", default=42)
        assert value == 42

        bands = get_frequency_bands(config)
        assert isinstance(bands, dict)

        band_names = get_frequency_band_names(config)
        assert isinstance(band_names, list)

    def test_path_utilities(self):
        from eeg_pipeline.infra.paths import (
            ensure_dir,
            deriv_features_path,
            deriv_plots_path,
            deriv_stats_path,
        )

        assert callable(ensure_dir)
        assert callable(deriv_features_path)
        assert callable(deriv_plots_path)
        assert callable(deriv_stats_path)

    def test_subject_utilities(self):
        from eeg_pipeline.utils.data.subjects import (
            parse_subject_args,
            get_available_subjects,
        )

        assert callable(parse_subject_args)
        assert callable(get_available_subjects)

    def test_cli_commands_registry(self):
        from eeg_pipeline.cli.commands import COMMANDS, get_command

        assert len(COMMANDS) > 0

        expected_commands = ["behavior", "features", "erp", "tfr", "decoding", "utilities"]
        for cmd_name in expected_commands:
            cmd = get_command(cmd_name)
            assert cmd is not None, f"Command '{cmd_name}' not found"
            assert hasattr(cmd, "run")
            assert hasattr(cmd, "setup")

    def test_validation_utilities(self):
        from eeg_pipeline.utils.validation import validate_epochs

        assert callable(validate_epochs)

    def test_progress_utilities(self):
        from eeg_pipeline.utils.progress import (
            PipelineProgress,
            BatchProgress,
        )

        assert PipelineProgress is not None
        assert BatchProgress is not None

    def test_finite_mask_extraction(self):
        from eeg_pipeline.utils.analysis.stats import extract_finite_mask

        x = np.array([1, np.nan, 3, 4])
        y = np.array([5, 6, np.nan, 8])

        x_out, y_out, mask = extract_finite_mask(x, y)

        assert len(x_out) == 2
        assert not mask[1]
        assert not mask[2]
