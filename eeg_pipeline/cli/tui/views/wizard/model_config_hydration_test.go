package wizard

import (
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestApplyConfigKeys_HydratesFeatureRestFlagFromFeatureConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"preprocessing.task_is_rest":       false,
		"feature_engineering.task_is_rest": true,
	})

	if !m.prepTaskIsRest {
		t.Fatalf("expected feature pipeline to hydrate task_is_rest from feature_engineering")
	}
}

func TestApplyConfigKeys_HydratesFeatureEngineeringDefaultsFromConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"feature_engineering.analysis_mode":                     "trial_ml_safe",
		"feature_engineering.constants.min_epochs_for_features": float64(14),
		"feature_engineering.compute_change_scores":             false,
		"feature_engineering.save_tfr_with_sidecar":             true,
		"feature_engineering.output.also_save_csv":              true,
		"feature_engineering.feature_categories":                []interface{}{"power", "connectivity"},
		"feature_engineering.spatial_modes":                     []interface{}{"roi", "channels"},
	})

	if m.featAnalysisMode != 1 {
		t.Fatalf("expected featAnalysisMode=1 (trial_ml_safe), got %d", m.featAnalysisMode)
	}
	if m.minEpochsForFeatures != 14 {
		t.Fatalf("expected minEpochsForFeatures=14, got %d", m.minEpochsForFeatures)
	}
	if m.featComputeChangeScores {
		t.Fatalf("expected featComputeChangeScores=false")
	}
	if !m.featSaveTfrWithSidecar {
		t.Fatalf("expected featSaveTfrWithSidecar=true")
	}
	if !m.featAlsoSaveCsv {
		t.Fatalf("expected featAlsoSaveCsv=true")
	}

	if got := m.SelectedCategories(); len(got) != 2 || got[0] != "connectivity" || got[1] != "power" {
		t.Fatalf("expected selected feature categories [connectivity power], got %#v", got)
	}
	if got := m.SelectedSpatialModes(); len(got) != 2 || got[0] != "roi" || got[1] != "channels" {
		t.Fatalf("expected selected spatial modes [roi channels], got %#v", got)
	}
}

func TestApplyConfigKeys_HydratesPreprocessingRestFlagFromPreprocessingConfig(t *testing.T) {
	m := New(types.PipelinePreprocessing, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"preprocessing.task_is_rest":       true,
		"feature_engineering.task_is_rest": false,
	})

	if !m.prepTaskIsRest {
		t.Fatalf("expected preprocessing pipeline to hydrate task_is_rest from preprocessing")
	}
}

func TestApplyConfigKeys_HydratesBidsRestRoot(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	m.ApplyConfigKeys(map[string]interface{}{
		"paths.bids_rest_root":  "/data/bids/rest",
		"paths.deriv_rest_root": "/data/derivatives/rest",
	})

	if m.bidsRestRoot != "/data/bids/rest" {
		t.Fatalf("expected bidsRestRoot='/data/bids/rest', got %q", m.bidsRestRoot)
	}
	if m.derivRestRoot != "/data/derivatives/rest" {
		t.Fatalf("expected derivRestRoot='/data/derivatives/rest', got %q", m.derivRestRoot)
	}
}

func TestApplyConfigKeys_HydratesSourceLocFmriContrastConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	values := map[string]interface{}{
		"feature_engineering.sourcelocalization.mode":                                 "fmri_informed",
		"feature_engineering.sourcelocalization.fmri.enabled":                         true,
		"feature_engineering.sourcelocalization.fmri.contrast.type":                   "custom",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_a.column":     "event_group",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_b.column":     "group_label",
		"feature_engineering.sourcelocalization.fmri.contrast.events_to_model":        []interface{}{"stim", "rating"},
		"feature_engineering.sourcelocalization.fmri.contrast.events_to_model_column": "event_class",
		"feature_engineering.sourcelocalization.fmri.contrast.phase_scope_column":     "phase_scope_col",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_scope_column": "scope_col",
		"feature_engineering.sourcelocalization.fmri.contrast.stim_phases_to_model":   []interface{}{"ramp", "plateau"},
		"feature_engineering.sourcelocalization.fmri.contrast.condition_scope_trial_types": []interface{}{
			"stim",
			"outcome",
		},
	}

	m.ApplyConfigKeys(values)

	if m.sourceLocMode != 1 {
		t.Fatalf("expected sourceLocMode=1 (fmri_informed), got %d", m.sourceLocMode)
	}
	if !m.sourceLocFmriEnabled {
		t.Fatalf("expected sourceLocFmriEnabled=true")
	}
	if m.sourceLocFmriContrastType != 1 {
		t.Fatalf("expected sourceLocFmriContrastType=1 (custom), got %d", m.sourceLocFmriContrastType)
	}
	if m.sourceLocFmriCondAColumn != "event_group" {
		t.Fatalf("expected condition A column event_group, got %q", m.sourceLocFmriCondAColumn)
	}
	if m.sourceLocFmriCondBColumn != "group_label" {
		t.Fatalf("expected condition B column group_label, got %q", m.sourceLocFmriCondBColumn)
	}
	if m.sourceLocFmriPhaseScopeColumn != "phase_scope_col" {
		t.Fatalf("expected phase scope column phase_scope_col, got %q", m.sourceLocFmriPhaseScopeColumn)
	}
	if m.sourceLocFmriConditionScopeColumn != "scope_col" {
		t.Fatalf("expected condition scope column scope_col, got %q", m.sourceLocFmriConditionScopeColumn)
	}
	if m.sourceLocFmriEventsToModel != "stim rating" {
		t.Fatalf("expected events_to_model 'stim rating', got %q", m.sourceLocFmriEventsToModel)
	}
	if m.sourceLocFmriEventsToModelColumn != "event_class" {
		t.Fatalf("expected events_to_model_column event_class, got %q", m.sourceLocFmriEventsToModelColumn)
	}
	if m.sourceLocFmriStimPhasesToModel != "ramp plateau" {
		t.Fatalf("expected stim phases 'ramp plateau', got %q", m.sourceLocFmriStimPhasesToModel)
	}
	if m.sourceLocFmriConditionScopeTrialTypes != "stim outcome" {
		t.Fatalf(
			"expected condition scope values 'stim outcome', got %q",
			m.sourceLocFmriConditionScopeTrialTypes,
		)
	}
}

func TestApplyConfigKeys_HydratesFmriThreadConfig(t *testing.T) {
	m := New(types.PipelineFmri, ".")
	values := map[string]interface{}{
		"fmri_preprocessing.fmriprep.nthreads":     12,
		"fmri_preprocessing.fmriprep.omp_nthreads": float64(6),
	}

	m.ApplyConfigKeys(values)

	if m.fmriNThreads != 12 {
		t.Fatalf("expected fmriNThreads=12, got %d", m.fmriNThreads)
	}
	if m.fmriOmpNThreads != 6 {
		t.Fatalf("expected fmriOmpNThreads=6, got %d", m.fmriOmpNThreads)
	}
}

func TestApplyConfigKeys_HydratesExtendedMLConfig(t *testing.T) {
	m := New(types.PipelineML, ".")
	values := map[string]interface{}{
		"machine_learning.preprocessing.imputer_strategy":                     "most_frequent",
		"machine_learning.preprocessing.pca.enabled":                          true,
		"machine_learning.preprocessing.pca.svd_solver":                       "randomized",
		"machine_learning.preprocessing.pca.n_components":                     0.9,
		"machine_learning.classification.resampler":                           "smote",
		"machine_learning.models.svm.kernel":                                  "poly",
		"machine_learning.models.svm.C_grid":                                  []interface{}{0.1, 1.0, 10.0},
		"machine_learning.models.random_forest.class_weight":                  "none",
		"machine_learning.analysis.time_generalization.min_subjects_per_cell": float64(7),
		"machine_learning.targets.strict_regression_target_continuous":        true,
		"machine_learning.interpretability.grouped_outputs":                   false,
		"machine_learning.classification.max_failed_fold_fraction":            0.33,
		"machine_learning.evaluation.bootstrap_iterations":                    float64(2500),
	}

	m.ApplyConfigKeys(values)

	if m.mlImputer != 2 {
		t.Fatalf("expected mlImputer=2 (most_frequent), got %d", m.mlImputer)
	}
	if !m.mlPCAEnabled {
		t.Fatalf("expected mlPCAEnabled=true")
	}
	if m.mlPCASvdSolver != 2 {
		t.Fatalf("expected mlPCASvdSolver=2 (randomized), got %d", m.mlPCASvdSolver)
	}
	if m.mlPCANComponents != 0.9 {
		t.Fatalf("expected mlPCANComponents=0.9, got %v", m.mlPCANComponents)
	}
	if m.mlClassificationResampler != 2 {
		t.Fatalf("expected mlClassificationResampler=2 (smote), got %d", m.mlClassificationResampler)
	}
	if m.mlSvmKernel != 2 {
		t.Fatalf("expected mlSvmKernel=2 (poly), got %d", m.mlSvmKernel)
	}
	if m.mlSvmCGrid != "0.1,1,10" {
		t.Fatalf("expected mlSvmCGrid='0.1,1,10', got %q", m.mlSvmCGrid)
	}
	if m.mlRfClassWeight != 2 {
		t.Fatalf("expected mlRfClassWeight=2 (none), got %d", m.mlRfClassWeight)
	}
	if m.mlTimeGenMinSubjects != 7 {
		t.Fatalf("expected mlTimeGenMinSubjects=7, got %d", m.mlTimeGenMinSubjects)
	}
	if !m.mlTargetsStrictRegressionCont {
		t.Fatalf("expected mlTargetsStrictRegressionCont=true")
	}
	if m.mlInterpretabilityGroupedOutputs {
		t.Fatalf("expected mlInterpretabilityGroupedOutputs=false")
	}
	if m.mlClassMaxFailedFoldFraction != 0.33 {
		t.Fatalf("expected mlClassMaxFailedFoldFraction=0.33, got %v", m.mlClassMaxFailedFoldFraction)
	}
	if m.mlEvalBootstrapIterations != 2500 {
		t.Fatalf("expected mlEvalBootstrapIterations=2500, got %d", m.mlEvalBootstrapIterations)
	}
}

func TestApplyConfigKeys_HydratesAdditionalFeatureAndPreprocessingConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")
	values := map[string]interface{}{
		"alignment.allow_misaligned_trim":                                         true,
		"alignment.min_alignment_samples":                                         float64(11),
		"alignment.fmri_onset_reference":                                          "first_iti_start",
		"eeg.ecg_channels":                                                        []interface{}{"ECG", "EKG"},
		"epochs.autoreject_n_interpolate":                                         []interface{}{4, 8, 16},
		"preprocessing.clean_events_qc.enabled":                                   false,
		"preprocessing.clean_events_qc.ecg_coupling.enabled":                      false,
		"preprocessing.clean_events_qc.ecg_coupling.output_column":                "ecg_coupling_qc",
		"preprocessing.clean_events_qc.ecg_coupling.channels":                     []interface{}{"ECG", "EKG"},
		"preprocessing.clean_events_qc.ecg_coupling.window":                       []interface{}{-1.5, 0.25},
		"preprocessing.clean_events_qc.peripheral_low_gamma.enabled":              false,
		"preprocessing.clean_events_qc.peripheral_low_gamma.output_column":        "gamma_qc",
		"preprocessing.clean_events_qc.peripheral_low_gamma.channels":             []interface{}{"Fp1", "Fp2"},
		"preprocessing.clean_events_qc.peripheral_low_gamma.band":                 []interface{}{28.0, 44.0},
		"preprocessing.clean_events_qc.peripheral_low_gamma.window":               []interface{}{-1.0, 0.5},
		"feature_engineering.spatial_transform_per_family.connectivity":           "laplacian",
		"feature_engineering.change_scores.transform":                             "log_ratio",
		"feature_engineering.change_scores.window_pairs":                          []interface{}{[]interface{}{"baseline", "active"}},
		"feature_engineering.pac.surrogate_method":                                "time_shift",
		"feature_engineering.aperiodic.max_freq_resolution_hz":                    0.5,
		"feature_engineering.directedconnectivity.min_samples_per_mvar_parameter": float64(25),
		"feature_engineering.erds.laterality_columns":                             []interface{}{"stim_side", "hand"},
		"feature_engineering.microstates.assign_from_gfp_peaks":                   false,
	}

	m.ApplyConfigKeys(values)

	if !m.alignAllowMisalignedTrim {
		t.Fatalf("expected alignAllowMisalignedTrim=true")
	}
	if m.alignMinAlignmentSamples != 11 {
		t.Fatalf("expected alignMinAlignmentSamples=11, got %d", m.alignMinAlignmentSamples)
	}
	if m.alignFmriOnsetReference != 1 {
		t.Fatalf("expected alignFmriOnsetReference=1 (first_volume/first_iti_start), got %d", m.alignFmriOnsetReference)
	}
	if m.prepEcgChannels != "ECG,EKG" {
		t.Fatalf("expected prepEcgChannels='ECG,EKG', got %q", m.prepEcgChannels)
	}
	if m.prepAutorejectNInterpolate != "4,8,16" {
		t.Fatalf("expected prepAutorejectNInterpolate='4,8,16', got %q", m.prepAutorejectNInterpolate)
	}
	if m.prepCleanEventsQCEnabled {
		t.Fatalf("expected prepCleanEventsQCEnabled=false")
	}
	if m.prepCleanEventsQCEcgVarianceEnabled {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceEnabled=false")
	}
	if m.prepCleanEventsQCEcgVarianceOutputColumn != "ecg_coupling_qc" {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceOutputColumn='ecg_coupling_qc', got %q", m.prepCleanEventsQCEcgVarianceOutputColumn)
	}
	if m.prepCleanEventsQCEcgVarianceChannels != "[\"ECG\",\"EKG\"]" {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceChannels JSON array, got %q", m.prepCleanEventsQCEcgVarianceChannels)
	}
	if m.prepCleanEventsQCEcgVarianceWindow != "[-1.5,0.25]" {
		t.Fatalf("expected prepCleanEventsQCEcgVarianceWindow JSON array, got %q", m.prepCleanEventsQCEcgVarianceWindow)
	}
	if m.prepCleanEventsQCPeripheralLowGammaEnabled {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaEnabled=false")
	}
	if m.prepCleanEventsQCPeripheralLowGammaOutputColumn != "gamma_qc" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaOutputColumn='gamma_qc', got %q", m.prepCleanEventsQCPeripheralLowGammaOutputColumn)
	}
	if m.prepCleanEventsQCPeripheralLowGammaChannels != "[\"Fp1\",\"Fp2\"]" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaChannels JSON array, got %q", m.prepCleanEventsQCPeripheralLowGammaChannels)
	}
	if m.prepCleanEventsQCPeripheralLowGammaBand != "[28,44]" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaBand JSON array, got %q", m.prepCleanEventsQCPeripheralLowGammaBand)
	}
	if m.prepCleanEventsQCPeripheralLowGammaWindow != "[-1,0.5]" {
		t.Fatalf("expected prepCleanEventsQCPeripheralLowGammaWindow JSON array, got %q", m.prepCleanEventsQCPeripheralLowGammaWindow)
	}
	if m.spatialTransformPerFamilyConnectivity != 3 {
		t.Fatalf("expected spatialTransformPerFamilyConnectivity=3 (laplacian), got %d", m.spatialTransformPerFamilyConnectivity)
	}
	if m.changeScoresTransform != 2 {
		t.Fatalf("expected changeScoresTransform=2 (log_ratio), got %d", m.changeScoresTransform)
	}
	if m.changeScoresWindowPairs != "baseline:active" {
		t.Fatalf("expected changeScoresWindowPairs='baseline:active', got %q", m.changeScoresWindowPairs)
	}
	if m.pacSurrogateMethod != 3 {
		t.Fatalf("expected pacSurrogateMethod=3 (time_shift), got %d", m.pacSurrogateMethod)
	}
	if m.aperiodicMaxFreqResolutionHz != 0.5 {
		t.Fatalf("expected aperiodicMaxFreqResolutionHz=0.5, got %v", m.aperiodicMaxFreqResolutionHz)
	}
	if m.directedConnMinSamplesPerMvarParam != 25 {
		t.Fatalf("expected directedConnMinSamplesPerMvarParam=25, got %d", m.directedConnMinSamplesPerMvarParam)
	}
	if m.erdsLateralityColumns != "stim_side,hand" {
		t.Fatalf("expected erdsLateralityColumns='stim_side,hand', got %q", m.erdsLateralityColumns)
	}
	if m.microstatesAssignFromGfpPeaks {
		t.Fatalf("expected microstatesAssignFromGfpPeaks=false")
	}
}

func TestApplyConfigKeys_HydratesConnectivityFeatureConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	values := map[string]interface{}{
		"feature_engineering.connectivity.measures":                     []interface{}{"imcoh", "plv"},
		"feature_engineering.connectivity.output_level":                 "global_only",
		"feature_engineering.connectivity.enable_graph_metrics":         true,
		"feature_engineering.connectivity.graph_top_prop":               0.25,
		"feature_engineering.connectivity.sliding_window_len":           1.5,
		"feature_engineering.connectivity.sliding_window_step":          0.75,
		"feature_engineering.connectivity.aec_mode":                     "sym",
		"feature_engineering.connectivity.mode":                         "multitaper",
		"feature_engineering.connectivity.aec_absolute":                 false,
		"feature_engineering.connectivity.enable_aec":                   false,
		"feature_engineering.connectivity.n_freqs_per_band":             float64(11),
		"feature_engineering.connectivity.n_cycles":                     6.5,
		"feature_engineering.connectivity.decim":                        float64(3),
		"feature_engineering.connectivity.min_segment_samples":          float64(80),
		"feature_engineering.connectivity.small_world_n_rand":           float64(150),
		"feature_engineering.connectivity.aec_output":                   []interface{}{"r", "z"},
		"feature_engineering.connectivity.force_within_epoch_for_ml":    false,
		"feature_engineering.connectivity.granularity":                  "condition",
		"feature_engineering.connectivity.condition_column":             "trial_type",
		"feature_engineering.connectivity.condition_values":             []interface{}{"pain", "nonpain"},
		"feature_engineering.connectivity.min_epochs_per_group":         float64(7),
		"feature_engineering.connectivity.min_cycles_per_band":          4.5,
		"feature_engineering.connectivity.warn_if_no_spatial_transform": false,
		"feature_engineering.connectivity.phase_estimator":              "across_epochs",
		"feature_engineering.connectivity.min_segment_sec":              2.5,
		"feature_engineering.connectivity.dynamic_enabled":              true,
		"feature_engineering.connectivity.dynamic_measures":             []interface{}{"aec"},
		"feature_engineering.connectivity.dynamic_autocorr_lag":         float64(2),
		"feature_engineering.connectivity.dynamic_min_windows":          float64(5),
		"feature_engineering.connectivity.dynamic_include_roi_pairs":    false,
		"feature_engineering.connectivity.dynamic_state_enabled":        false,
		"feature_engineering.connectivity.dynamic_state_n_states":       float64(4),
		"feature_engineering.connectivity.dynamic_state_min_windows":    float64(10),
		"feature_engineering.connectivity.dynamic_state_random_state":   float64(17),
	}

	m.ApplyConfigKeys(values)

	if got := m.selectedConnectivityMeasures(); len(got) != 2 || got[0] != "imcoh" || got[1] != "plv" {
		t.Fatalf("expected connectivity measures [imcoh plv], got %#v", got)
	}
	if m.connOutputLevel != 1 {
		t.Fatalf("expected connOutputLevel=1 (global_only), got %d", m.connOutputLevel)
	}
	if !m.connGraphMetrics {
		t.Fatalf("expected connGraphMetrics=true")
	}
	if m.connGraphProp != 0.25 {
		t.Fatalf("expected connGraphProp=0.25, got %v", m.connGraphProp)
	}
	if m.connWindowLen != 1.5 {
		t.Fatalf("expected connWindowLen=1.5, got %v", m.connWindowLen)
	}
	if m.connWindowStep != 0.75 {
		t.Fatalf("expected connWindowStep=0.75, got %v", m.connWindowStep)
	}
	if m.connAECMode != 2 {
		t.Fatalf("expected connAECMode=2 (sym), got %d", m.connAECMode)
	}
	if m.connMode != 1 {
		t.Fatalf("expected connMode=1 (multitaper), got %d", m.connMode)
	}
	if m.connAECAbsolute {
		t.Fatalf("expected connAECAbsolute=false")
	}
	if m.connEnableAEC {
		t.Fatalf("expected connEnableAEC=false")
	}
	if m.connNFreqsPerBand != 11 {
		t.Fatalf("expected connNFreqsPerBand=11, got %d", m.connNFreqsPerBand)
	}
	if m.connNCycles != 6.5 {
		t.Fatalf("expected connNCycles=6.5, got %v", m.connNCycles)
	}
	if m.connDecim != 3 {
		t.Fatalf("expected connDecim=3, got %d", m.connDecim)
	}
	if m.connMinSegSamples != 80 {
		t.Fatalf("expected connMinSegSamples=80, got %d", m.connMinSegSamples)
	}
	if m.connSmallWorldNRand != 150 {
		t.Fatalf("expected connSmallWorldNRand=150, got %d", m.connSmallWorldNRand)
	}
	if m.connAECOutput != 2 {
		t.Fatalf("expected connAECOutput=2 (r+z), got %d", m.connAECOutput)
	}
	if m.connForceWithinEpochML {
		t.Fatalf("expected connForceWithinEpochML=false")
	}
	if m.connGranularity != 1 {
		t.Fatalf("expected connGranularity=1 (condition), got %d", m.connGranularity)
	}
	if m.connConditionColumn != "trial_type" {
		t.Fatalf("expected connConditionColumn='trial_type', got %q", m.connConditionColumn)
	}
	if m.connConditionValues != "pain nonpain" {
		t.Fatalf("expected connConditionValues='pain nonpain', got %q", m.connConditionValues)
	}
	if m.connMinEpochsPerGroup != 7 {
		t.Fatalf("expected connMinEpochsPerGroup=7, got %d", m.connMinEpochsPerGroup)
	}
	if m.connMinCyclesPerBand != 4.5 {
		t.Fatalf("expected connMinCyclesPerBand=4.5, got %v", m.connMinCyclesPerBand)
	}
	if m.connWarnNoSpatialTransform {
		t.Fatalf("expected connWarnNoSpatialTransform=false")
	}
	if m.connPhaseEstimator != 1 {
		t.Fatalf("expected connPhaseEstimator=1 (across_epochs), got %d", m.connPhaseEstimator)
	}
	if m.connMinSegmentSec != 2.5 {
		t.Fatalf("expected connMinSegmentSec=2.5, got %v", m.connMinSegmentSec)
	}
	if !m.connDynamicEnabled {
		t.Fatalf("expected connDynamicEnabled=true")
	}
	if m.connDynamicMeasures != 2 {
		t.Fatalf("expected connDynamicMeasures=2 (aec), got %d", m.connDynamicMeasures)
	}
	if m.connDynamicAutocorrLag != 2 {
		t.Fatalf("expected connDynamicAutocorrLag=2, got %d", m.connDynamicAutocorrLag)
	}
	if m.connDynamicMinWindows != 5 {
		t.Fatalf("expected connDynamicMinWindows=5, got %d", m.connDynamicMinWindows)
	}
	if m.connDynamicIncludeROIPairs {
		t.Fatalf("expected connDynamicIncludeROIPairs=false")
	}
	if m.connDynamicStateEnabled {
		t.Fatalf("expected connDynamicStateEnabled=false")
	}
	if m.connDynamicStateNStates != 4 {
		t.Fatalf("expected connDynamicStateNStates=4, got %d", m.connDynamicStateNStates)
	}
	if m.connDynamicStateMinWindows != 10 {
		t.Fatalf("expected connDynamicStateMinWindows=10, got %d", m.connDynamicStateMinWindows)
	}
	if m.connDynamicStateRandomSeed != 17 {
		t.Fatalf("expected connDynamicStateRandomSeed=17, got %d", m.connDynamicStateRandomSeed)
	}
}

func TestApplyConfigKeys_HydratesPlottingSourceLocalizationAndComparisons(t *testing.T) {
	m := New(types.PipelinePlotting, ".")
	values := map[string]interface{}{
		"feature_engineering.sourcelocalization.subjects_dir": "/fs/subjects",
		"plotting.plots.features.sourcelocalization.hemi":     "both",
		"plotting.plots.features.sourcelocalization.views":    []interface{}{"lateral", "medial"},
		"plotting.plots.features.sourcelocalization.cortex":   "classic",
		"plotting.comparisons.compare_windows":                true,
		"plotting.comparisons.comparison_windows":             []interface{}{"baseline", "plateau"},
		"plotting.comparisons.compare_columns":                false,
		"plotting.comparisons.comparison_segment":             "plateau",
		"plotting.comparisons.comparison_column":              "pain_binary_coded",
		"plotting.comparisons.comparison_values":              []interface{}{"0", "1"},
		"plotting.comparisons.comparison_labels":              []interface{}{"nonpain", "pain"},
		"plotting.comparisons.comparison_rois":                []interface{}{"ParOccipital_Left", "ParOccipital_Right"},
		"plotting.overwrite":                                  true,
	}

	m.ApplyConfigKeys(values)

	if m.plotSourceSubjectsDir != "/fs/subjects" {
		t.Fatalf("expected plotSourceSubjectsDir '/fs/subjects', got %q", m.plotSourceSubjectsDir)
	}
	if m.plotSourceHemi != "both" {
		t.Fatalf("expected plotSourceHemi='both', got %q", m.plotSourceHemi)
	}
	if m.plotSourceViews != "lateral medial" {
		t.Fatalf("expected plotSourceViews='lateral medial', got %q", m.plotSourceViews)
	}
	if m.plotSourceCortex != "classic" {
		t.Fatalf("expected plotSourceCortex='classic', got %q", m.plotSourceCortex)
	}
	if m.plotCompareWindows == nil || !*m.plotCompareWindows {
		t.Fatalf("expected plotCompareWindows=true")
	}
	if m.plotComparisonWindowsSpec != "baseline plateau" {
		t.Fatalf("expected plotComparisonWindowsSpec='baseline plateau', got %q", m.plotComparisonWindowsSpec)
	}
	if m.plotCompareColumns == nil || *m.plotCompareColumns {
		t.Fatalf("expected plotCompareColumns=false")
	}
	if m.plotComparisonSegment != "plateau" {
		t.Fatalf("expected plotComparisonSegment='plateau', got %q", m.plotComparisonSegment)
	}
	if m.plotComparisonColumn != "pain_binary_coded" {
		t.Fatalf("expected plotComparisonColumn='pain_binary_coded', got %q", m.plotComparisonColumn)
	}
	if m.plotComparisonValuesSpec != "0 1" {
		t.Fatalf("expected plotComparisonValuesSpec='0 1', got %q", m.plotComparisonValuesSpec)
	}
	if m.plotComparisonLabelsSpec != "nonpain pain" {
		t.Fatalf("expected plotComparisonLabelsSpec='nonpain pain', got %q", m.plotComparisonLabelsSpec)
	}
	if m.plotComparisonROIsSpec != "ParOccipital_Left ParOccipital_Right" {
		t.Fatalf(
			"expected plotComparisonROIsSpec='ParOccipital_Left ParOccipital_Right', got %q",
			m.plotComparisonROIsSpec,
		)
	}
	if m.plotOverwrite == nil || !*m.plotOverwrite {
		t.Fatalf("expected plotOverwrite=true")
	}
}

func TestApplyConfigKeys_HydratesPlottingConnectivityAndSelectionOverrides(t *testing.T) {
	m := New(types.PipelinePlotting, ".")

	values := map[string]interface{}{
		"plotting.plots.connectivity.width_per_circle":              10.5,
		"plotting.plots.connectivity.width_per_band":                6.75,
		"plotting.plots.connectivity.height_per_measure":            4.25,
		"plotting.plots.features.connectivity.circle_top_fraction":  0.2,
		"plotting.plots.features.connectivity.circle_min_lines":     float64(12),
		"plotting.plots.features.connectivity.network_top_fraction": 0.35,
		"plotting.plots.features.pac_pairs":                         []interface{}{"theta_gamma", "alpha_beta"},
		"plotting.plots.features.connectivity.measures":             []interface{}{"imcoh", "pli"},
		"plotting.plots.features.spectral.metrics":                  []interface{}{"peak_frequency", "iaf"},
		"plotting.plots.features.bursts.metrics":                    []interface{}{"rate", "duration"},
		"plotting.plots.features.asymmetry.stat":                    "effect_size_d",
		"plotting.plots.features.temporal.time_bins":                []interface{}{"early", "late"},
		"plotting.plots.features.temporal.time_labels":              []interface{}{"Early", "Late"},
	}

	m.ApplyConfigKeys(values)

	if m.plotConnectivityWidthPerCircle != 10.5 {
		t.Fatalf("expected plotConnectivityWidthPerCircle=10.5, got %v", m.plotConnectivityWidthPerCircle)
	}
	if m.plotConnectivityWidthPerBand != 6.75 {
		t.Fatalf("expected plotConnectivityWidthPerBand=6.75, got %v", m.plotConnectivityWidthPerBand)
	}
	if m.plotConnectivityHeightPerMeasure != 4.25 {
		t.Fatalf("expected plotConnectivityHeightPerMeasure=4.25, got %v", m.plotConnectivityHeightPerMeasure)
	}
	if m.plotConnectivityCircleTopFraction != 0.2 {
		t.Fatalf("expected plotConnectivityCircleTopFraction=0.2, got %v", m.plotConnectivityCircleTopFraction)
	}
	if m.plotConnectivityCircleMinLines != 12 {
		t.Fatalf("expected plotConnectivityCircleMinLines=12, got %d", m.plotConnectivityCircleMinLines)
	}
	if m.plotConnectivityNetworkTopFraction != 0.35 {
		t.Fatalf("expected plotConnectivityNetworkTopFraction=0.35, got %v", m.plotConnectivityNetworkTopFraction)
	}
	if m.plotPacPairsSpec != "theta_gamma alpha_beta" {
		t.Fatalf("expected plotPacPairsSpec='theta_gamma alpha_beta', got %q", m.plotPacPairsSpec)
	}
	if m.plotConnectivityMeasuresSpec != "imcoh pli" {
		t.Fatalf("expected plotConnectivityMeasuresSpec='imcoh pli', got %q", m.plotConnectivityMeasuresSpec)
	}
	if m.plotSpectralMetricsSpec != "peak_frequency iaf" {
		t.Fatalf("expected plotSpectralMetricsSpec='peak_frequency iaf', got %q", m.plotSpectralMetricsSpec)
	}
	if m.plotBurstsMetricsSpec != "rate duration" {
		t.Fatalf("expected plotBurstsMetricsSpec='rate duration', got %q", m.plotBurstsMetricsSpec)
	}
	if m.plotAsymmetryStatSpec != "effect_size_d" {
		t.Fatalf("expected plotAsymmetryStatSpec='effect_size_d', got %q", m.plotAsymmetryStatSpec)
	}
	if m.plotTemporalTimeBinsSpec != "early late" {
		t.Fatalf("expected plotTemporalTimeBinsSpec='early late', got %q", m.plotTemporalTimeBinsSpec)
	}
	if m.plotTemporalTimeLabelsSpec != "Early Late" {
		t.Fatalf("expected plotTemporalTimeLabelsSpec='Early Late', got %q", m.plotTemporalTimeLabelsSpec)
	}
}

func TestApplyConfigKeys_HydratesPlottingDefaultsStylingAndTfrConfig(t *testing.T) {
	m := New(types.PipelinePlotting, ".")

	values := map[string]interface{}{
		"plotting.defaults.formats":                                      []interface{}{"png", "pdf"},
		"plotting.defaults.dpi":                                          float64(150),
		"plotting.defaults.savefig_dpi":                                  float64(600),
		"plotting.defaults.bbox_inches":                                  "tight",
		"plotting.defaults.pad_inches":                                   0.15,
		"plotting.defaults.font.family":                                  "Arial",
		"plotting.defaults.font.weight":                                  "bold",
		"plotting.defaults.font.sizes.small":                             float64(9),
		"plotting.defaults.font.sizes.ylabel":                            float64(13),
		"plotting.defaults.font.sizes.figure_title":                      float64(21),
		"plotting.defaults.layout.tight_rect":                            []interface{}{0.0, 0.02, 1.0, 0.98},
		"plotting.defaults.layout.gridspec.width_ratios":                 []interface{}{1.0, 2.0, 1.0},
		"plotting.defaults.layout.gridspec.hspace":                       0.4,
		"plotting.figure_sizes.standard":                                 []interface{}{12.0, 8.0},
		"plotting.styling.colors.condition_1":                            "#111111",
		"plotting.styling.colors.network_node":                           "#333333",
		"plotting.styling.alpha.grid":                                    0.25,
		"plotting.styling.scatter.marker_size.small":                     float64(12),
		"plotting.styling.scatter.edgecolor":                             "black",
		"plotting.styling.bar.capsize_large":                             float64(8),
		"plotting.styling.line.width.bold":                               3.5,
		"plotting.styling.histogram.bins_tfr":                            float64(30),
		"plotting.styling.kde.points":                                    float64(256),
		"plotting.styling.errorbar.capsize":                              float64(4),
		"plotting.styling.text_position.title_y":                         0.97,
		"plotting.styling.text_position.residual_qc_title_y":             0.88,
		"plotting.validation.min_bins_for_calibration":                   float64(4),
		"plotting.plots.itpc.shared_colorbar":                            false,
		"plotting.plots.topomap.contours":                                float64(8),
		"plotting.plots.topomap.colormap":                                "RdBu_r",
		"plotting.plots.topomap.diff_annotation_enabled":                 false,
		"plotting.plots.topomap.sig_mask_params.markersize":              6.5,
		"plotting.plots.tfr.log_base":                                    10.0,
		"plotting.plots.tfr.percentage_multiplier":                       100.0,
		"time_frequency_analysis.topomap.temporal.window_size_ms":        75.0,
		"time_frequency_analysis.topomap.temporal.window_count":          float64(5),
		"plotting.plots.tfr.topomap.label_x_position":                    0.33,
		"plotting.plots.tfr.topomap.title_pad":                           float64(12),
		"time_frequency_analysis.topomap.temporal.single_subject.hspace": 0.2,
		"time_frequency_analysis.topomap.temporal.single_subject.wspace": 0.15,
		"plotting.plots.roi.width_per_band":                              4.0,
		"plotting.plots.power.height_per_segment":                        5.5,
		"plotting.plots.itpc.width_per_bin":                              2.2,
		"plotting.plots.pac.cmap":                                        "magma",
		"plotting.plots.aperiodic.n_perm":                                float64(2000),
		"plotting.plots.complexity.width_per_measure":                    3.3,
	}

	m.ApplyConfigKeys(values)

	if !m.plotFormatSelected["png"] || !m.plotFormatSelected["pdf"] || m.plotFormatSelected["svg"] {
		t.Fatalf("expected plot formats png/pdf selected and svg unselected, got %#v", m.plotFormatSelected)
	}
	if m.plotDpiIndex != 0 {
		t.Fatalf("expected plotDpiIndex=0 (150 dpi), got %d", m.plotDpiIndex)
	}
	if m.plotSavefigDpiIndex != 2 {
		t.Fatalf("expected plotSavefigDpiIndex=2 (600 dpi), got %d", m.plotSavefigDpiIndex)
	}
	if m.plotBboxInches != "tight" {
		t.Fatalf("expected plotBboxInches='tight', got %q", m.plotBboxInches)
	}
	if m.plotPadInches != 0.15 {
		t.Fatalf("expected plotPadInches=0.15, got %v", m.plotPadInches)
	}
	if m.plotFontFamily != "Arial" || m.plotFontWeight != "bold" {
		t.Fatalf("expected font settings hydrated, got family=%q weight=%q", m.plotFontFamily, m.plotFontWeight)
	}
	if m.plotFontSizeSmall != 9 || m.plotFontSizeYLabel != 13 || m.plotFontSizeFigureTitle != 21 {
		t.Fatalf("expected font sizes hydrated, got small=%d ylabel=%d figureTitle=%d", m.plotFontSizeSmall, m.plotFontSizeYLabel, m.plotFontSizeFigureTitle)
	}
	if m.plotLayoutTightRectSpec != "0 0.02 1 0.98" {
		t.Fatalf("expected plotLayoutTightRectSpec='0 0.02 1 0.98', got %q", m.plotLayoutTightRectSpec)
	}
	if m.plotGridSpecWidthRatiosSpec != "1 2 1" || m.plotGridSpecHspace != 0.4 {
		t.Fatalf("expected gridspec settings hydrated, got ratios=%q hspace=%v", m.plotGridSpecWidthRatiosSpec, m.plotGridSpecHspace)
	}
	if m.plotFigureSizeStandardSpec != "12 8" {
		t.Fatalf("expected plotFigureSizeStandardSpec='12 8', got %q", m.plotFigureSizeStandardSpec)
	}
	if m.plotColorCondA != "#111111" || m.plotColorNetworkNode != "#333333" {
		t.Fatalf("expected color settings hydrated, got condA=%q network=%q", m.plotColorCondA, m.plotColorNetworkNode)
	}
	if m.plotAlphaGrid != 0.25 {
		t.Fatalf("expected plotAlphaGrid=0.25, got %v", m.plotAlphaGrid)
	}
	if m.plotScatterMarkerSizeSmall != 12 || m.plotScatterEdgeColor != "black" {
		t.Fatalf("expected scatter settings hydrated, got size=%d edge=%q", m.plotScatterMarkerSizeSmall, m.plotScatterEdgeColor)
	}
	if m.plotBarCapsizeLarge != 8 || m.plotLineWidthBold != 3.5 {
		t.Fatalf("expected bar/line settings hydrated, got capsizeLarge=%d lineBold=%v", m.plotBarCapsizeLarge, m.plotLineWidthBold)
	}
	if m.plotHistBinsTFR != 30 || m.plotKdePoints != 256 || m.plotErrorbarCapsize != 4 {
		t.Fatalf("expected hist/kde/errorbar settings hydrated, got histTFR=%d kde=%d err=%d", m.plotHistBinsTFR, m.plotKdePoints, m.plotErrorbarCapsize)
	}
	if m.plotTextTitleY != 0.97 || m.plotTextResidualQcTitleY != 0.88 {
		t.Fatalf("expected text positions hydrated, got titleY=%v residual=%v", m.plotTextTitleY, m.plotTextResidualQcTitleY)
	}
	if m.plotValidationMinBinsForCalibration != 4 {
		t.Fatalf("expected plotValidationMinBinsForCalibration=4, got %d", m.plotValidationMinBinsForCalibration)
	}
	if m.plotSharedColorbar {
		t.Fatalf("expected plotSharedColorbar=false")
	}
	if m.plotTopomapContours != 8 || m.plotTopomapColormap != "RdBu_r" || m.plotTopomapDiffAnnotation == nil || *m.plotTopomapDiffAnnotation {
		t.Fatalf("expected topomap settings hydrated, got contours=%d cmap=%q diff=%v", m.plotTopomapContours, m.plotTopomapColormap, m.plotTopomapDiffAnnotation)
	}
	if m.plotTopomapSigMaskMarkerSize != 6.5 {
		t.Fatalf("expected plotTopomapSigMaskMarkerSize=6.5, got %v", m.plotTopomapSigMaskMarkerSize)
	}
	if m.plotTFRLogBase != 10.0 || m.plotTFRPercentageMultiplier != 100.0 {
		t.Fatalf("expected TFR settings hydrated, got logBase=%v pct=%v", m.plotTFRLogBase, m.plotTFRPercentageMultiplier)
	}
	if m.plotTFRTopomapWindowSizeMs != 75.0 || m.plotTFRTopomapWindowCount != 5 {
		t.Fatalf("expected TFR topomap window settings hydrated, got size=%v count=%d", m.plotTFRTopomapWindowSizeMs, m.plotTFRTopomapWindowCount)
	}
	if m.plotTFRTopomapLabelXPosition != 0.33 || m.plotTFRTopomapTitlePad != 12 {
		t.Fatalf("expected TFR topomap label/title settings hydrated, got labelX=%v titlePad=%d", m.plotTFRTopomapLabelXPosition, m.plotTFRTopomapTitlePad)
	}
	if m.plotTFRTopomapTemporalHspace != 0.2 || m.plotTFRTopomapTemporalWspace != 0.15 {
		t.Fatalf("expected TFR topomap spacing hydrated, got hspace=%v wspace=%v", m.plotTFRTopomapTemporalHspace, m.plotTFRTopomapTemporalWspace)
	}
	if m.plotRoiWidthPerBand != 4.0 || m.plotPowerHeightPerSegment != 5.5 || m.plotItpcWidthPerBin != 2.2 {
		t.Fatalf("expected sizing settings hydrated, got roi=%v power=%v itpc=%v", m.plotRoiWidthPerBand, m.plotPowerHeightPerSegment, m.plotItpcWidthPerBin)
	}
	if m.plotPacCmap != "magma" || m.plotAperiodicNPerm != 2000 || m.plotComplexityWidthPerMeasure != 3.3 {
		t.Fatalf("expected pac/aperiodic/complexity settings hydrated, got pac=%q aper=%d comp=%v", m.plotPacCmap, m.plotAperiodicNPerm, m.plotComplexityWidthPerMeasure)
	}
}
