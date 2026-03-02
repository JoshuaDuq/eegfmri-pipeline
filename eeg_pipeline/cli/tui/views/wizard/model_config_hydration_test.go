package wizard

import (
	"testing"

	"github.com/eeg-pipeline/tui/types"
)

func TestApplyConfigKeys_HydratesSourceLocFmriContrastConfig(t *testing.T) {
	m := New(types.PipelineFeatures, ".")

	values := map[string]interface{}{
		"feature_engineering.sourcelocalization.mode":                                 "fmri_informed",
		"feature_engineering.sourcelocalization.fmri.enabled":                         true,
		"feature_engineering.sourcelocalization.fmri.contrast.type":                   "custom",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_a.column":     "event_group",
		"feature_engineering.sourcelocalization.fmri.contrast.condition_b.column":     "group_label",
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
	if m.sourceLocFmriContrastType != 3 {
		t.Fatalf("expected sourceLocFmriContrastType=3 (custom), got %d", m.sourceLocFmriContrastType)
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
